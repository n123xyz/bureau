"""Unified hierarchical work decomposition.

Every task goes through work_node():
  - Agent executes partial work within a configurable budget
  - A unified critic reviews the work
  - Agent optionally delegates remaining subtasks to child nodes
  - Child nodes read parent's output as context and continue

Scheduling uses separate read/write file sets:
  - Tasks whose WRITES overlap another task's WRITES or READS are serialized
  - Tasks that only READ the same files can run in parallel
"""

from __future__ import annotations

import asyncio
import json
import logging
import textwrap
import traceback
import uuid
from pathlib import Path
from typing import Any

from bureau_mod.agents import run_agent
from bureau_mod.config import Config, Critic, Phase
from bureau_mod.context import make_context
from bureau_mod.git_utils import git_commit, repo_file_listing
from bureau_mod.prompts import (
    HIERARCHY_CONTEXT,
    PARALLELISM_RULES,
    SUBTASKS_FILE,
    SUBTASKS_SCHEMA_DOC,
)
from bureau_mod.revision import critique_and_revise
from bureau_mod.state import STATE, TaskNode, TaskStatus, emit_event, emit_task_output
from bureau_mod.worktree import WorktreeManager

log = logging.getLogger("bureau")

# (description, reads, writes)
PlanItem = tuple[str, set[str], set[str]]


# ═══════════════════════════════════════════════════════════════════════════
# Concurrency partitioning
# ═══════════════════════════════════════════════════════════════════════════

def _partition_by_deps(
    tasks: list[PlanItem],
) -> list[list[tuple[int, PlanItem]]]:
    """Partition tasks into sequential groups that can run internally parallel.

    Rules:
    - Two tasks CONFLICT if one's writes overlap the other's writes or reads.
    - Tasks with no writes and no reads go into their own sequential group.
    - Within a group, all tasks can run concurrently (no conflicts).
    """
    indexed = list(enumerate(tasks))
    groups: list[list[tuple[int, PlanItem]]] = []
    current_group: list[tuple[int, PlanItem]] = []
    # Track the union of all writes and all reads+writes in current group
    current_writes: set[str] = set()
    current_all: set[str] = set()  # reads ∪ writes of current group

    for i, (desc, reads, writes) in indexed:
        if not reads and not writes:
            # No dependency info — isolate to avoid hidden conflicts
            if current_group:
                groups.append(current_group)
                current_group = []
                current_writes = set()
                current_all = set()
            groups.append([(i, (desc, reads, writes))])
        elif (writes & current_all) or (reads & current_writes):
            # Conflict: this task's writes touch something in the group,
            # or this task reads something the group writes
            if current_group:
                groups.append(current_group)
            current_group = [(i, (desc, reads, writes))]
            current_writes = set(writes)
            current_all = reads | writes
        else:
            # No conflict — add to current parallel group
            current_group.append((i, (desc, reads, writes)))
            current_writes |= writes
            current_all |= reads | writes

    if current_group:
        groups.append(current_group)

    return groups


# ═══════════════════════════════════════════════════════════════════════════
# Schemas
# ═══════════════════════════════════════════════════════════════════════════

_ITEMS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "reads": {
                "type": "array", "items": {"type": "string"},
                "description": "Files this task needs to read (inputs/deps)",
            },
            "writes": {
                "type": "array", "items": {"type": "string"},
                "description": "Files this task will create or modify",
            },
        },
        "required": ["description"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _parse_plan_items(result: Any) -> list[PlanItem]:
    """Parse structured delegation/plan result into [(desc, reads, writes)]."""
    raw_items: list = []
    if isinstance(result, dict):
        raw_items = result.get("subtasks",
                               result.get("items",
                                          result.get("tasks", [])))
    elif isinstance(result, list):
        raw_items = result

    items: list[PlanItem] = []
    for item in raw_items:
        if isinstance(item, dict):
            desc = item.get("description", str(item))
            reads = set(item.get("reads", []))
            writes = set(item.get("writes", []))
            items.append((desc, reads, writes))
        elif isinstance(item, str):
            items.append((item, set(), set()))
        else:
            items.append((str(item), set(), set()))
    return items


def _read_subtasks_file(cwd: str) -> list[PlanItem]:
    """Read and clean up the subtasks file written by the work agent.

    Returns parsed subtask list (empty if file doesn't exist or is invalid).
    Deletes the file after reading.
    """
    path = Path(cwd) / SUBTASKS_FILE
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return _parse_plan_items(data)
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Failed to read {SUBTASKS_FILE}: {e}")
        return []
    finally:
        try:
            path.unlink()
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

async def work_node(
    *,
    problem: str,
    phase: Phase,
    prev_phases: list[str],
    task_description: str,
    critics: list[Critic],
    cwd: str,
    cfg: Config,
    label: str,
    depth: int = 0,
    parent_task_id: str | None = None,
    worktree_mgr: WorktreeManager | None = None,
    file_reads: set[str] | None = None,
    file_writes: set[str] | None = None,
    concurrency_group: int | None = None,
    task_id: str | None = None,
) -> None:
    """Unified recursive work node.

    Each node:
    1. Executes partial work within a configurable budget
    2. Gets a unified critique/revise cycle
    3. Optionally delegates remaining subtasks to child nodes
    """

    # ── Task node ──────────────────────────────────────────────────────
    if task_id and (existing := STATE.get_task(task_id)):
        task_node = existing
    else:
        task_id = f"{label}-{uuid.uuid4().hex[:8]}"
        task_node = TaskNode(
            id=task_id,
            label=label,
            description=task_description,
            parent_id=parent_task_id,
            max_revision_rounds=cfg.max_revision_rounds,
            file_reads=sorted(file_reads) if file_reads else [],
            file_writes=sorted(file_writes) if file_writes else [],
            concurrency_group=concurrency_group,
        )
        STATE.add_task(task_node)

    # ── Working directory (worktree or main) ───────────────────────────
    work_cwd = cwd
    if cfg.use_worktrees and worktree_mgr and depth > 0:
        work_cwd = str(await worktree_mgr.create_worktree(task_id))
        task_node.worktree_path = work_cwd
        emit_event("task", task_node.to_dict())

    try:
        STATE.update_task_status(task_id, TaskStatus.RUNNING)

        context = make_context(
            problem, phase, prev_phases,
            tree_path=f"{label} (depth {depth})",
            cwd=work_cwd,
        )

        # ── Step 1: Execute work (and write subtasks file if needed) ─
        can_delegate = depth < cfg.max_depth
        log.info(f"  [{label}] executing work "
                 f"(budget ~{cfg.work_budget} lines)")
        emit_task_output(task_id,
                         f"executing work (budget ~{cfg.work_budget} lines)")

        await _execute_work(
            context=context,
            task_description=task_description,
            task_node=task_node,
            cwd=work_cwd, cfg=cfg,
            phase=phase, label=label,
            can_delegate=can_delegate,
        )
        git_commit(work_cwd, f"bureau: {label} work")

        # Gate check
        listing = repo_file_listing(work_cwd)
        if "(empty repository" in listing:
            log.warning(f"  [{label}] produced no files — skipping")
            STATE.update_task_status(task_id, TaskStatus.COMPLETED)
            return

        # ── Step 2: Critique and revise ────────────────────────────────
        # The critic sees all files including _bureau_subtasks.json
        # (if the work agent wrote one), so the revision cycle covers
        # both the work done and the proposed delegation plan.
        await critique_and_revise(
            context=context,
            task_description=task_description,
            task_node=task_node,
            critics=critics,
            cwd=work_cwd, cfg=cfg,
            phase_name=phase.name, label=label,
        )

        # ── Step 3: Read (possibly revised) subtasks and run them ─────
        subtasks = _read_subtasks_file(work_cwd)
        if subtasks:
            log.info(f"  [{label}] delegating "
                     f"{len(subtasks)} subtasks")
            emit_task_output(
                task_id,
                f"delegating {len(subtasks)} subtasks")
            for i, (desc, reads, writes) in enumerate(subtasks):
                rstr = (f" R[{', '.join(sorted(reads)[:3])}]"
                        if reads else "")
                wstr = (f" W[{', '.join(sorted(writes)[:3])}]"
                        if writes else "")
                log.info(f"    {i+1}. {desc}{rstr}{wstr}")
                emit_task_output(
                    task_id, f"  {i+1}. {desc}{rstr}{wstr}")

            await _run_children(
                plan_items=subtasks,
                problem=problem, phase=phase,
                prev_phases=prev_phases,
                critics=critics, cwd=work_cwd, cfg=cfg,
                label=label, depth=depth, task_id=task_id,
                worktree_mgr=worktree_mgr,
            )

        STATE.update_task_status(task_id, TaskStatus.COMPLETED)

    except asyncio.CancelledError:
        STATE.update_task_status(task_id, TaskStatus.PAUSED, "cancelled")
        raise
    except Exception as e:
        STATE.update_task_status(task_id, TaskStatus.FAILED, str(e))
        log.error(f"  [{label}] failed: {e}")
        log.debug(traceback.format_exc())
    finally:
        if cfg.use_worktrees and worktree_mgr and work_cwd != cwd:
            await worktree_mgr.merge_and_cleanup(
                task_id, f"bureau: {label} complete"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Work execution
# ═══════════════════════════════════════════════════════════════════════════

async def _execute_work(
    *,
    context: str,
    task_description: str,
    task_node: TaskNode,
    cwd: str,
    cfg: Config,
    phase: Phase,
    label: str,
    can_delegate: bool,
) -> None:
    """Execute work within the configured budget.

    If the agent has remaining work to delegate, it writes a
    _bureau_subtasks.json file which the critic will review alongside
    all other output.
    """
    writes = task_node.file_writes
    reads = task_node.file_reads
    file_constraint = ""
    if writes:
        wlist = ", ".join(writes)
        file_constraint += f"\n        You MUST write these files: {wlist}"
        file_constraint += "\n        Do NOT create or modify any other files."
    if reads:
        rlist = ", ".join(reads)
        file_constraint += f"\n        You may read (but not modify): {rlist}"
    if not writes and not reads:
        file_constraint = ("\n        Only create or modify files within your"
                           " working directory.")

    work_budget = cfg.work_budget
    delegation_section = ""
    if can_delegate:
        delegation_section = textwrap.dedent(f"""\

            {SUBTASKS_SCHEMA_DOC}
            Maximum {cfg.max_split_pieces} subtasks. Each child node gets the
            same ~{work_budget}-line budget, so size subtasks accordingly.

            {PARALLELISM_RULES}
            Do NOT write `{SUBTASKS_FILE}` if all work is complete at this
            level — only if sub-levels exist to delegate.
        """)
    else:
        delegation_section = textwrap.dedent("""\

            You are at maximum decomposition depth. Complete ALL work for
            this task — no further delegation is possible.
        """)

    work_prompt = context + textwrap.dedent(f"""\
        ## Your role: WORKER

        {HIERARCHY_CONTEXT}
        Your budget at this level: **~{work_budget} lines total across all
        files you write** (combined line count of every file you create or
        modify).

        ### Task
        {task_description}
{file_constraint}

        ### Guidelines
        - Read existing project files first to understand the current state.
        - Do the work that belongs at your level of the hierarchy — create
          the structures, interfaces, and content for this granularity.
        - Write real, complete code/content — not stubs or placeholders.
        - **Prefer many small files over few large ones.** Each file should
          cover one module, component, or subject. This enables subtasks to
          work on separate files in parallel without conflicts.
        - Stay within your ~{work_budget}-line budget. When you hit it, or
          when natural sub-levels exist, delegate via `{SUBTASKS_FILE}`.
        - Delegation is not just for overflow — use it whenever the work has
          natural subdivisions (chapters→sections, packages→modules, etc.).
          Do your level well and let children handle the details.
{delegation_section}
    """)

    task_node.task_type = "executor"
    emit_event("task", task_node.to_dict())

    await run_agent(
        prompt=work_prompt, cwd=cwd, cfg=cfg, label=f"{label}/work",
        phase_name=phase.name, task_type="executor",
        task_id=task_node.id,
    )



# ═══════════════════════════════════════════════════════════════════════════
# Child execution
# ═══════════════════════════════════════════════════════════════════════════

async def _run_children(
    *,
    plan_items: list[PlanItem],
    problem: str,
    phase: Phase,
    prev_phases: list[str],
    critics: list[Critic],
    cwd: str,
    cfg: Config,
    label: str,
    depth: int,
    task_id: str,
    worktree_mgr: WorktreeManager | None,
) -> None:
    """Pre-create child task nodes, then execute with concurrency groups."""
    # Pre-create all children so the full tree is visible in the UI
    child_ids: dict[int, str] = {}
    for i, (desc, reads, writes) in enumerate(plan_items):
        child_label = f"{label}/{i+1}"
        child_id = f"{child_label}-{uuid.uuid4().hex[:8]}"
        child_node = TaskNode(
            id=child_id,
            label=child_label,
            description=desc,
            parent_id=task_id,
            max_revision_rounds=cfg.max_revision_rounds,
            file_reads=sorted(reads) if reads else [],
            file_writes=sorted(writes) if writes else [],
        )
        STATE.add_task(child_node)
        child_ids[i] = child_id

    # Execute with concurrency groups
    if cfg.parallel_subtasks and len(plan_items) > 1:
        groups = _partition_by_deps(plan_items)
        log.info(f"  [{label}] {len(plan_items)} items → {len(groups)} groups")
        emit_task_output(task_id,
                         f"{len(plan_items)} items → {len(groups)} "
                         f"concurrent groups")

        for gi, group in enumerate(groups):
            if len(group) == 1:
                i, (desc, reads, writes) = group[0]
                await work_node(
                    problem=problem, phase=phase, prev_phases=prev_phases,
                    task_description=desc, critics=critics,
                    cwd=cwd, cfg=cfg, label=f"{label}/{i+1}",
                    depth=depth + 1, parent_task_id=task_id,
                    worktree_mgr=worktree_mgr,
                    file_reads=reads, file_writes=writes,
                    concurrency_group=gi,
                    task_id=child_ids[i],
                )
            else:
                await asyncio.gather(*(
                    work_node(
                        problem=problem, phase=phase, prev_phases=prev_phases,
                        task_description=desc, critics=critics,
                        cwd=cwd, cfg=cfg, label=f"{label}/{i+1}",
                        depth=depth + 1, parent_task_id=task_id,
                        worktree_mgr=worktree_mgr,
                        file_reads=reads, file_writes=writes,
                        concurrency_group=gi,
                        task_id=child_ids[i],
                    )
                    for i, (desc, reads, writes) in group
                ))
            git_commit(cwd, f"bureau: {label} group done")
    else:
        for i, (desc, reads, writes) in enumerate(plan_items):
            await work_node(
                problem=problem, phase=phase, prev_phases=prev_phases,
                task_description=desc, critics=critics,
                cwd=cwd, cfg=cfg, label=f"{label}/{i+1}",
                depth=depth + 1, parent_task_id=task_id,
                worktree_mgr=worktree_mgr,
                file_reads=reads, file_writes=writes,
                task_id=child_ids[i],
            )

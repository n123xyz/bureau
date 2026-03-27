"""Unified hierarchical planning and execution.

Every task goes through plan_or_execute():
  - depth < max_depth → agent chooses plan vs execute
  - depth >= max_depth → always executes directly

Scheduling uses separate read/write file sets:
  - Tasks whose WRITES overlap another task's WRITES or READS are serialized
  - Tasks that only READ the same files can run in parallel
"""

from __future__ import annotations

import asyncio
import logging
import textwrap
import traceback
import uuid
from typing import Any

from bureau_mod.agents import run_structured_agent
from bureau_mod.config import Config, Critic, Phase
from bureau_mod.context import make_context
from bureau_mod.git_utils import git_commit
from bureau_mod.revision import revision_cycle
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

DECIDE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["plan", "execute"]},
        "tasks": _ITEMS_SCHEMA,
    },
    "required": ["action"],
}

PLAN_CRITIQUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string", "enum": ["accept", "revise", "execute"],
            "description": ("accept if plan is good, revise if it needs changes,"
                            " execute if items should be collapsed into a"
                            " single directly-executed task"),
        },
        "reason": {"type": "string"},
        "revised_plan": _ITEMS_SCHEMA,
    },
    "required": ["verdict"],
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _parse_plan_items(result: Any) -> list[PlanItem]:
    """Parse structured plan/decision result into [(desc, reads, writes)]."""
    raw_items: list = []
    if isinstance(result, dict):
        raw_items = result.get("items", result.get("tasks", []))
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


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

async def plan_or_execute(
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
    """Unified recursive planning and execution.

    - depth < max:  agent chooses plan or execute
    - depth >= max: always executes directly
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

        # ── Leaf: force execute ────────────────────────────────────────
        if depth >= cfg.max_depth:
            log.info(f"  [{label}] max depth — executing directly")
            emit_task_output(task_id, "max depth — executing directly")
            await revision_cycle(
                context=context,
                task_description=task_description,
                task_node=task_node,
                critics=critics,
                cwd=work_cwd, cfg=cfg,
                phase_name=phase.name, label=label,
            )
            return

        # ── Agent decides: plan (split) or execute directly ──────────
        action, plan_items = await _decide(
            context=context, task_description=task_description,
            cwd=work_cwd, cfg=cfg, label=label,
            phase=phase, task_id=task_id, depth=depth,
        )
        if action == "execute" or not plan_items:
            log.info(f"  [{label}] executing directly")
            emit_task_output(task_id, "decided: execute directly")
            await revision_cycle(
                context=context,
                task_description=task_description,
                task_node=task_node,
                critics=critics,
                cwd=work_cwd, cfg=cfg,
                phase_name=phase.name, label=label,
            )
            return
        plan_items, exec_desc = await _critique_plan(
            items=plan_items, context=context, cwd=work_cwd,
            cfg=cfg, label=label, phase=phase, task_id=task_id,
        )

        # Critic may collapse plan to execute-directly (returns None)
        if plan_items is None:
            final_desc = exec_desc or task_description
            log.info(f"  [{label}] plan critic: execute directly")
            emit_task_output(task_id, "plan critic: collapsed to execute")
            task_node.description = final_desc
            emit_event("task", task_node.to_dict())
            await revision_cycle(
                context=context,
                task_description=final_desc,
                task_node=task_node,
                critics=critics,
                cwd=work_cwd, cfg=cfg,
                phase_name=phase.name, label=label,
            )
            return

        # ── Execute the plan ───────────────────────────────────────────
        log.info(f"  [{label}] planned {len(plan_items)} sub-tasks")
        emit_task_output(task_id, f"planned {len(plan_items)} sub-tasks")
        for i, (desc, reads, writes) in enumerate(plan_items):
            rstr = f" R[{', '.join(sorted(reads)[:3])}]" if reads else ""
            wstr = f" W[{', '.join(sorted(writes)[:3])}]" if writes else ""
            log.info(f"    {i+1}. {desc}{rstr}{wstr}")
            emit_task_output(task_id, f"  {i+1}. {desc}{rstr}{wstr}")

        await _run_children(
            plan_items=plan_items,
            problem=problem, phase=phase, prev_phases=prev_phases,
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
# Planning stages
# ═══════════════════════════════════════════════════════════════════════════

PARALLELISM_INSRUCTIONS = textwrap.dedent("""\
        IMPORTANT — Parallelism and dependency rules:
    
        - Each sub-task must write to DIFFERENT files. No two sub-tasks may
          write to the same file.
    
        - Multiple sub-tasks MAY read the same file — readers don't block each
          other.
    
        - A sub-task that reads a file written by another sub-task will wait for
          that writer to finish first. Use this to express dependencies: if task
          B needs the output of task A, list A's output file in B's reads.
    
        - Prefer creating many small files over few large ones. More files
          allows expressing finer-grained dependencies and more parallelism, and
          is easier to manage and review without exhausting an agent.

        - You do not need to make separate _tasks_ to read or write separate
          _files_. A single agent can comfortably read, write and revise text on
          the order of 2000 lines (e.g. 5 files of 400 lines each). If the task
          would likely involve more reading/writing than that, it's a good
          candidate for splitting, but if it's below that scale, it's often
          better to keep it as one task to avoid unnecessary overhead.
    
        - All files must be within the project working directory. Do NOT create
          or modify files outside it.
""")

async def _decide(
    *, context: str, task_description: str, cwd: str, cfg: Config,
    label: str, phase: Phase, task_id: str, depth: int,
) -> tuple[str, list[PlanItem]]:
    """Intermediate-level decision: plan (split) or execute."""
    next_msg = ("the last (must execute)"
                if depth + 1 >= cfg.max_depth else "intermediate")
    prompt = context + textwrap.dedent(f"""\ ## Your role: PLANNER

        Decide how to handle the following task:

        ### Task {task_description}

        ### Options
         
        **plan** — This task involves multiple distinct parts (different files,
        different components). Break it into sub-tasks (max
        {cfg.max_split_pieces}). For each sub-task, describe what to do and list
        the files it will **read** (inputs/dependencies) and **write** (create
        or modify).

        **execute** — This task is small and focused enough to implement
        directly (e.g. likely to fit within ~2000 lines of text spread across
        one or a few files).

        {PARALLELISM_INSRUCTIONS}
    
        Depth: {depth}/{cfg.max_depth}. Next level is {next_msg}.
    """)
    decision = await run_structured_agent(
        prompt=prompt, schema=DECIDE_SCHEMA,
        cwd=cwd, cfg=cfg, label=f"{label}/plan",
        phase_name=phase.name, task_type="planner",
        task_id=task_id,
    )
    if decision is None or not isinstance(decision, dict):
        return ("execute", [])

    action = decision.get("action", "execute")
    items = _parse_plan_items(decision) if action == "plan" else []
    return (action, items)


# ═══════════════════════════════════════════════════════════════════════════
# Plan critique (in-memory)
# ═══════════════════════════════════════════════════════════════════════════

def _format_plan_for_review(items: list[PlanItem]) -> str:
    """Format plan items as numbered text for critique prompts."""
    lines = []
    for i, (desc, reads, writes) in enumerate(items):
        parts = [f"  {i+1}. {desc}"]
        if writes:
            parts.append(f"     writes: {', '.join(sorted(writes))}")
        if reads:
            parts.append(f"     reads: {', '.join(sorted(reads))}")
        lines.append("\n".join(parts))
    return "\n".join(lines)


async def _critique_plan(
    *, items: list[PlanItem], context: str, cwd: str, cfg: Config,
    label: str, phase: Phase, task_id: str,
) -> tuple[list[PlanItem] | None, str | None]:
    """Run one critique pass on a plan.

    Returns (revised_items, None) to continue planning,
    or (None, task_description) to collapse and execute directly.
    """
    plan_text = _format_plan_for_review(items)
    prompt = context + textwrap.dedent(f"""\ ## Your role: PLAN CRITIC

        Review the following plan and decide whether it is good or needs
        revision.

        ### Plan ({len(items)} items)
{plan_text}

        ### What to check

        - **Too-small tasks**: If a task is too big it will be re-split by a
          sub-agent, you don't have to worry about that. But if a task is too
          _small_ you might want to merge it with others to avoid unnecessary
          overhead. A single agent can comfortably read, write and revise text
          on the order of 2000 lines (e.g. 5 files of 400 lines each). If a task
          would likely involve more reading/writing than that, it's a good
          candidate for splitting, but if it's below that scale, it's often
          better to keep it as one task to avoid unnecessary overhead.

        - **Redundant tasks**: Are there tasks that do close-enough to "the same
          thing" that they should be merged?

        - **Dependent tasks**: If two tasks are strictly dependent (one reads a
          file the other writes), they will not be able to run in parallel, so
          the only reason to keep them separate is if the combined task would be
          too big for a single agent's work. Consider the size and decide on
          split vs. merge.

        - **Overlap**: Do multiple tasks write to the same files? That's a
          conflict — merge them or reassign files.

        - **Missing deps**: Does a task read a file that no other task writes?
          That's fine (it exists already). But if it reads a file another task
          creates, is that dependency listed?

        - **Scope creep**: Does any task go beyond the phase goal?

        ### Decision
        
          - **accept**: The plan is reasonable. Minor imperfections are fine.
        
          - **revise**: The plan has real problems. Provide a `revised_plan`
            with the corrected items. Keep the same format (description, reads,
            writes).

          - **execute**: The items should be collapsed into a single task and
            executed directly (e.g. you merged everything down to one item).
            You may optionally provide a single-item `revised_plan` with the
            merged task description; otherwise the original task description
            will be used.

        Be pragmatic. Only revise if there are genuine problems, not style nits.
    """)

    emit_task_output(task_id, "critiquing plan...")
    result = await run_structured_agent(
        prompt=prompt, schema=PLAN_CRITIQUE_SCHEMA,
        cwd=cwd, cfg=cfg, label=f"{label}/plan-critic",
        phase_name=phase.name, task_type="planner",
        task_id=task_id,
    )

    if not isinstance(result, dict):
        return items, None

    verdict = result.get("verdict", "accept")
    reason = result.get("reason", "")

    # Helper: extract a merged description from revised_plan if present
    def _extract_exec_desc() -> str | None:
        rp = result.get("revised_plan")
        if rp:
            parsed = _parse_plan_items({"items": rp})
            if parsed:
                return parsed[0][0]  # description of first item
        return None

    if verdict == "execute":
        merged_desc = _extract_exec_desc()
        log.info(f"  [{label}] plan critic: execute directly — {reason}")
        emit_task_output(task_id,
                         f"plan collapsed to execute: {reason}")
        return None, merged_desc

    if verdict == "revise" and result.get("revised_plan"):
        revised = _parse_plan_items({"items": result["revised_plan"]})
        if revised:
            # If revised down to 1 item, treat as execute
            if len(revised) == 1:
                log.info(f"  [{label}] plan revised to single item — "
                         f"will execute directly: {reason}")
                emit_task_output(task_id,
                                 f"plan revised to single task "
                                 f"(executing directly): {reason}")
                return None, revised[0][0]
            log.info(f"  [{label}] plan revised: {reason}")
            emit_task_output(task_id,
                             f"plan revised ({len(items)}→{len(revised)}): "
                             f"{reason}")
            return revised, None
        log.warning(f"  [{label}] critic said revise but gave empty plan")
    else:
        log.info(f"  [{label}] plan accepted: {reason}")
        emit_task_output(task_id, f"plan accepted: {reason}")

    return items, None


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
                await plan_or_execute(
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
                    plan_or_execute(
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
            await plan_or_execute(
                problem=problem, phase=phase, prev_phases=prev_phases,
                task_description=desc, critics=critics,
                cwd=cwd, cfg=cfg, label=f"{label}/{i+1}",
                depth=depth + 1, parent_task_id=task_id,
                worktree_mgr=worktree_mgr,
                file_reads=reads, file_writes=writes,
                task_id=child_ids[i],
            )

"""Critic/revision cycle: execute → critique → revise → judge."""

from __future__ import annotations

import asyncio
import logging
import textwrap
from dataclasses import dataclass
from typing import Any

from bureau_mod.agents import run_agent, run_structured_agent
from bureau_mod.config import Config, Critic
from bureau_mod.git_utils import git_commit, git_get_head, git_restore_to, repo_file_listing
from bureau_mod.state import (
    PAUSE_EVENT,
    STATE,
    TaskNode,
    TaskStatus,
    emit_event,
    emit_task_output,
)

log = logging.getLogger("bureau")


# ═══════════════════════════════════════════════════════════════════════════
# Schemas
# ═══════════════════════════════════════════════════════════════════════════

EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "clean": {"type": "boolean", "description": "true if no issues found"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "description": {"type": "string"},
                    "severity": {"type": "string",
                                 "enum": ["high", "medium", "low"]},
                },
                "required": ["file", "description", "severity"],
            },
        },
    },
    "required": ["clean"],
}

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["accept", "reject"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict"],
}


# ═══════════════════════════════════════════════════════════════════════════
# Critic types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CriticEdit:
    file: str
    description: str
    severity: str


@dataclass
class CriticResult:
    role: str
    clean: bool
    edits: list[CriticEdit]


async def run_critic_structured(
    *,
    context: str,
    task_description: str,
    critic: Critic,
    cwd: str,
    cfg: Config,
    phase_name: str,
    label: str,
    task_id: str | None = None,
) -> CriticResult:
    critic_prompt = context + textwrap.dedent(f"""\
        ## Your role: CRITIC — {critic.role}

        Review the current state of the project files in light of the
        task described below. Your specific review focus:

        {critic.prompt}

        ### Task that was supposed to be executed
        {task_description}

        Instructions:
        - Read all relevant project files.
        - For each issue, specify the exact file and what needs to change.
        - Be concrete: name functions, describe the fix, state expected behavior.
        - Severity: "high" = missing functionality or bugs, "medium" = significant
          improvement, "low" = cosmetic or style.
        - If everything is satisfactory, set clean=true and omit edits.
        - Do NOT fix anything. Only identify issues.
        - Only review files within the working directory.
    """)

    result = await run_structured_agent(
        prompt=critic_prompt, schema=EDIT_SCHEMA,
        cwd=cwd, cfg=cfg, label=label,
        phase_name=phase_name, task_type="critic",
        task_id=task_id,
    )

    if result is None or not isinstance(result, dict):
        log.warning(f"  [{label}] critic returned unparseable output")
        return CriticResult(role=critic.role, clean=True, edits=[])

    clean = result.get("clean", True)
    raw_edits = result.get("edits", [])
    edits = []
    for e in raw_edits:
        if isinstance(e, dict) and "description" in e:
            edits.append(CriticEdit(
                file=e.get("file", "unknown"),
                description=e["description"],
                severity=e.get("severity", "medium"),
            ))

    return CriticResult(role=critic.role, clean=clean, edits=edits)


# ═══════════════════════════════════════════════════════════════════════════
# Revision cycle
# ═══════════════════════════════════════════════════════════════════════════

async def revision_cycle(
    *,
    context: str,
    task_description: str,
    task_node: TaskNode,
    critics: list[Critic],
    cwd: str,
    cfg: Config,
    phase_name: str,
    label: str,
) -> None:
    """Execute → critique → revise cycle with task state tracking."""

    if task_node.status == TaskStatus.SKIPPED:
        log.info(f"  [{label}] skipped (user request)")
        return

    STATE.update_task_status(task_node.id, TaskStatus.RUNNING)

    # --- Step 1: Initial execution ---
    task_node.task_type = "executor"
    emit_event("task", task_node.to_dict())

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

    exec_prompt = context + textwrap.dedent(f"""\
        ## Your role: EXECUTOR

        Execute the following task thoroughly and completely. Read existing
        project files first to understand the current state. Write or modify
        all necessary files. Do NOT skip anything — implement every detail.

        ### Task
        {task_description}
{file_constraint}

        IMPORTANT: Do not leave stubs, placeholders, or TODOs. Every piece of
        functionality described above must be fully implemented.
    """)

    await run_agent(
        prompt=exec_prompt, cwd=cwd, cfg=cfg, label=f"{label}/exec",
        phase_name=phase_name, task_type="executor",
        task_id=task_node.id,
    )
    git_commit(cwd, f"bureau: {label} initial execution")

    # Gate check
    listing = repo_file_listing(cwd)
    if "(empty repository" in listing:
        log.warning(f"  [{label}] executor produced no files — skipping critics")
        STATE.update_task_status(task_node.id, TaskStatus.COMPLETED)
        return

    # --- Critique/revise rounds ---
    max_rounds = task_node.max_revision_rounds
    for round_num in range(1, max_rounds + 1):
        # Check for skip/pause/stop-revising
        if task_node.status == TaskStatus.SKIPPED:
            log.info(f"  [{label}] skipped mid-revision")
            return
        if task_node.stop_revising:
            log.info(f"  [{label}] stop-revising flag set, finishing")
            STATE.update_task_status(task_node.id, TaskStatus.COMPLETED)
            return
        await PAUSE_EVENT.wait()
        if STATE.stopping:
            raise asyncio.CancelledError("Stopping")

        task_node.revision_round = round_num
        task_node.task_type = f"revision r{round_num}/{max_rounds}"
        emit_event("task", task_node.to_dict())
        log.info(f"  [{label}] critique round {round_num}/{max_rounds}")
        emit_task_output(task_node.id,
                         f"critique round {round_num}/{max_rounds}")

        async def _run_one(critic: Critic) -> CriticResult:
            # Create a child task node for this critic
            critic_id = f"{task_node.id}-critic-{critic.role}-r{round_num}"
            critic_node = TaskNode(
                id=critic_id,
                label=f"critic ({critic.role})",
                description=f"Review focus: {critic.prompt[:200]}",
                task_type="critic",
                parent_id=task_node.id,
                worktree_path=task_node.worktree_path,
                max_revision_rounds=0,
            )
            STATE.add_task(critic_node)
            STATE.update_task_status(critic_id, TaskStatus.RUNNING)
            try:
                result = await run_critic_structured(
                    context=context, task_description=task_description,
                    critic=critic, cwd=cwd, cfg=cfg,
                    phase_name=phase_name,
                    label=f"{label}/critic-{critic.role}-r{round_num}",
                    task_id=critic_id,
                )
                STATE.update_task_status(critic_id, TaskStatus.COMPLETED)
                return result
            except Exception as e:
                STATE.update_task_status(
                    critic_id, TaskStatus.FAILED, str(e))
                raise

        if cfg.parallel_critics and len(critics) > 1:
            log.info(f"  [{label}] running {len(critics)} critics concurrently")
            critic_results: list[CriticResult] = list(await asyncio.gather(
                *(_run_one(c) for c in critics)
            ))
        else:
            critic_results = [await _run_one(c) for c in critics]

        any_substantive = False
        for cr in critic_results:
            n_high = sum(1 for e in cr.edits if e.severity == "high")
            n_med = sum(1 for e in cr.edits if e.severity == "medium")
            if cr.clean or not cr.edits:
                log.info(f"  [{label}] critic {cr.role}: clean")
            else:
                log.info(f"  [{label}] critic {cr.role}: "
                         f"{len(cr.edits)} edits (H={n_high} M={n_med})")
                if n_high > 0 or n_med > 0:
                    any_substantive = True

        if not any_substantive:
            log.info(f"  [{label}] all critics clean at round {round_num}")
            STATE.update_task_status(task_node.id, TaskStatus.COMPLETED)
            return

        # Check stop_revising again before applying edits
        if task_node.stop_revising:
            log.info(f"  [{label}] stop-revising flag set, finishing")
            STATE.update_task_status(task_node.id, TaskStatus.COMPLETED)
            return

        for cr in critic_results:
            substantive_edits = [e for e in cr.edits
                                 if e.severity in ("high", "medium")]
            if not substantive_edits:
                continue

            # Create a child task node for the reviser
            rev_id = f"{task_node.id}-reviser-{cr.role}-r{round_num}"
            rev_node = TaskNode(
                id=rev_id,
                label=f"reviser ({cr.role})",
                description=(f"Fix {len(substantive_edits)} issues from "
                             f"{cr.role} critic"),
                task_type="reviser",
                parent_id=task_node.id,
                worktree_path=task_node.worktree_path,
                max_revision_rounds=0,
            )
            STATE.add_task(rev_node)
            STATE.update_task_status(rev_id, TaskStatus.RUNNING)

            log.info(f"  [{label}] applying {len(substantive_edits)} edits "
                     f"from {cr.role}")
            pre_critic_commit = git_get_head(cwd)

            edit_list = "\n".join(
                f"  {j+1}. [{e.severity}] {e.file}: {e.description}"
                for j, e in enumerate(substantive_edits)
            )

            revise_prompt = context + textwrap.dedent(f"""\
                ## Your role: REVISER

                Address ALL of the following issues found by the {cr.role}
                reviewer.

                ### Original task
                {task_description}

                ### Issues to fix
{edit_list}

                Instructions:
                - Address every issue in the list above.
                - For each issue, make the minimum change needed.
                - Do NOT refactor unrelated code.
                - Only modify files listed in the original task's write set.
                  Do NOT create or modify any other files.
            """)

            await run_agent(
                prompt=revise_prompt, cwd=cwd, cfg=cfg,
                label=f"{label}/fix-{cr.role}-r{round_num}",
                phase_name=phase_name, task_type="reviser",
                task_id=rev_id,
            )
            git_commit(cwd, f"bureau: {label} {cr.role} edits r{round_num}")
            STATE.update_task_status(rev_id, TaskStatus.COMPLETED)

            # Judge — also as a child task node
            judge_id = f"{task_node.id}-judge-{cr.role}-r{round_num}"
            judge_node = TaskNode(
                id=judge_id,
                label=f"judge ({cr.role})",
                description=f"Evaluate {cr.role} revisions",
                task_type="judge",
                parent_id=task_node.id,
                worktree_path=task_node.worktree_path,
                max_revision_rounds=0,
            )
            STATE.add_task(judge_node)
            STATE.update_task_status(judge_id, TaskStatus.RUNNING)

            judge_prompt = context + textwrap.dedent(f"""\
                ## Your role: JUDGE

                A batch of edits was applied for the {cr.role} reviewer.
                Review the current state.

                ### Original task
                {task_description}

                ### Edits that were applied
            """)
            for j, edit in enumerate(substantive_edits):
                judge_prompt += (f"    {j+1}. [{edit.severity}] "
                                 f"{edit.file}: {edit.description}\n")
            judge_prompt += "\nDecide: accept (keep) or reject (revert)?"

            verdict = await run_structured_agent(
                prompt=judge_prompt, schema=JUDGE_SCHEMA,
                cwd=cwd, cfg=cfg,
                label=f"{label}/judge-{cr.role}-r{round_num}",
                phase_name=phase_name, task_type="judge",
                task_id=judge_id,
            )

            v = "accept"
            if isinstance(verdict, dict):
                v = verdict.get("verdict", "accept")
                reason = verdict.get("reason", "")
                log.info(f"  [{label}] judge({cr.role}): {v}"
                         f"{' — ' + reason[:80] if reason else ''}")

            STATE.update_task_status(judge_id, TaskStatus.COMPLETED)

            if v == "reject" and pre_critic_commit:
                log.info(f"  [{label}] reverting {cr.role} edits")
                git_restore_to(cwd, pre_critic_commit)

    log.info(f"  [{label}] revision cycle complete ({max_rounds} rounds)")
    STATE.update_task_status(task_node.id, TaskStatus.COMPLETED)

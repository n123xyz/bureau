"""Unified critique/revise cycle: critique → revise → judge.

All configured critic categories are merged into a single holistic review,
so critics can consider how issues relate to each other (e.g. a
simplification that makes a correctness issue moot).

Critic categories have optional file globs — only categories whose globs
match at least one file touched by the work node are included.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import textwrap
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from bureau_mod.agents import run_agent, run_structured_agent
from bureau_mod.config import Config, Critic
from bureau_mod.git_utils import git_commit, git_get_head, git_restore_to
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


# ═══════════════════════════════════════════════════════════════════════════
# Merged critic
# ═══════════════════════════════════════════════════════════════════════════


def _glob_matches(globs: list[str] | None, files: list[str]) -> bool:
    """Return True if any glob matches any file in the list.

    None or empty globs means "matches everything".
    """
    if not globs:
        return True
    for pattern in globs:
        for f in files:
            if fnmatch.fnmatch(f, pattern):
                return True
            # Also match against just the filename (not full path)
            if fnmatch.fnmatch(PurePosixPath(f).name, pattern):
                return True
    return False


def _filter_critics(critics: list[Critic], files: list[str]) -> list[Critic]:
    """Filter critics to those whose globs match at least one written file."""
    if not files:
        return list(critics)  # no file info → include all
    return [c for c in critics if _glob_matches(c.globs, files)]


def _merge_critic_prompts(critics: list[Critic]) -> str:
    """Merge multiple critic prompts into a single holistic review brief."""
    if not critics:
        return ("Review the work for completeness, correctness, "
                "and simplicity.")
    if len(critics) == 1:
        return f"**{critics[0].role}**: {critics[0].prompt}"
    parts = ["Consider all of the following aspects holistically — issues in "
             "one area often interact with others (e.g. a simplification "
             "might make a correctness issue moot, a completeness gap might "
             "make other issues irrelevant).\n"]
    for c in critics:
        parts.append(f"**{c.role}**: {c.prompt}")
    return "\n\n".join(parts)


async def _run_merged_critic(
    *,
    context: str,
    task_description: str,
    merged_prompt: str,
    cwd: str,
    cfg: Config,
    phase_name: str,
    label: str,
    task_id: str | None = None,
) -> CriticResult:
    """Run a single unified critic with merged review categories."""
    critic_prompt = context + textwrap.dedent(f"""\
        ## Your role: REVIEWER

        Review the current state of the project files in light of the
        task described below. Your review should consider all of the
        following aspects holistically:

        {merged_prompt}

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
        - Consider how issues relate to each other — a simplification might make
          another issue moot, or a missing feature might change other priorities.
        - If you see a `_bureau_subtasks.json` file, it describes work that
          will be done LATER by child agents — it is a delegation plan, not
          something that has run yet. This means:
          * Files listed in the subtasks' "writes" fields DO NOT EXIST YET.
            That is expected. Do NOT flag missing files as bugs if they are
            listed as outputs of a pending subtask.
          * Code at this level may contain forward references (imports,
            calls, type annotations) to things that subtasks will create.
            This is intentional — the subtasks will fill in those gaps.
            Likewise docs may make reference to not-yet-written chapters
            or sections. That is not a cause for concern if the forward
            reference is to work mentioned in _bureau_subtasks.json.
          * DO review the delegation plan itself: are subtasks well-scoped?
            Redundant? Missing coverage? Should some be merged or split?
            You may list issues against `_bureau_subtasks.json` and the
            reviser can modify it.
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
        return CriticResult(role="reviewer", clean=True, edits=[])

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

    return CriticResult(role="reviewer", clean=clean, edits=edits)


# ═══════════════════════════════════════════════════════════════════════════
# Critique and revise cycle
# ═══════════════════════════════════════════════════════════════════════════

async def critique_and_revise(
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
    """Unified critique/revise cycle (called after work is already done).

    Runs a single merged critic combining all configured review categories,
    then revises if issues are found, with a judge to accept/reject.
    """

    if task_node.status == TaskStatus.SKIPPED:
        log.info(f"  [{label}] skipped (user request)")
        return

    max_rounds = task_node.max_revision_rounds
    if max_rounds <= 0:
        return

    # Filter critics to those relevant to the files this node writes
    written_files = task_node.file_writes
    applicable = _filter_critics(critics, written_files)
    if not applicable:
        log.info(f"  [{label}] no applicable critics for files: "
                 f"{written_files[:5]}")
        return

    merged_prompt = _merge_critic_prompts(applicable)

    for round_num in range(1, max_rounds + 1):
        # Check for skip/pause/stop-revising
        if task_node.status == TaskStatus.SKIPPED:
            log.info(f"  [{label}] skipped mid-revision")
            return
        if task_node.stop_revising:
            log.info(f"  [{label}] stop-revising flag set, finishing")
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

        # ── Single merged critic ───────────────────────────────────────
        critic_id = f"{task_node.id}-critic-r{round_num}"
        critic_node = TaskNode(
            id=critic_id,
            label=f"critic r{round_num}",
            description=(f"Unified review: "
                         f"{', '.join(c.role for c in applicable)}"),
            task_type="critic",
            parent_id=task_node.id,
            worktree_path=task_node.worktree_path,
            max_revision_rounds=0,
        )
        STATE.add_task(critic_node)
        STATE.update_task_status(critic_id, TaskStatus.RUNNING)

        try:
            cr = await _run_merged_critic(
                context=context,
                task_description=task_description,
                merged_prompt=merged_prompt,
                cwd=cwd, cfg=cfg,
                phase_name=phase_name,
                label=f"{label}/critic-r{round_num}",
                task_id=critic_id,
            )
            STATE.update_task_status(critic_id, TaskStatus.COMPLETED)
        except Exception as e:
            STATE.update_task_status(critic_id, TaskStatus.FAILED, str(e))
            raise

        if cr.clean or not cr.edits:
            log.info(f"  [{label}] critic clean at round {round_num}")
            return

        n_high = sum(1 for e in cr.edits if e.severity == "high")
        n_med = sum(1 for e in cr.edits if e.severity == "medium")
        log.info(f"  [{label}] critic: {len(cr.edits)} edits "
                 f"(H={n_high} M={n_med})")

        substantive = [e for e in cr.edits
                       if e.severity in ("high", "medium")]
        if not substantive:
            log.info(f"  [{label}] no substantive issues at "
                     f"round {round_num}")
            return

        if task_node.stop_revising:
            log.info(f"  [{label}] stop-revising flag set, finishing")
            return

        # ── Revise ─────────────────────────────────────────────────────
        rev_id = f"{task_node.id}-reviser-r{round_num}"
        rev_node = TaskNode(
            id=rev_id,
            label=f"reviser r{round_num}",
            description=f"Fix {len(substantive)} issues",
            task_type="reviser",
            parent_id=task_node.id,
            worktree_path=task_node.worktree_path,
            max_revision_rounds=0,
        )
        STATE.add_task(rev_node)
        STATE.update_task_status(rev_id, TaskStatus.RUNNING)

        pre_commit = git_get_head(cwd)

        edit_list = "\n".join(
            f"  {j+1}. [{e.severity}] {e.file}: {e.description}"
            for j, e in enumerate(substantive)
        )

        revise_prompt = context + textwrap.dedent(f"""\
            ## Your role: REVISER

            Address ALL of the following issues found by the reviewer.

            ### Original task
            {task_description}

            ### Issues to fix
{edit_list}

            Instructions:
            - Address every issue in the list above.
            - For each issue, make the minimum change needed.
            - Do NOT refactor unrelated code.
            - Only modify files within your working directory.
        """)

        await run_agent(
            prompt=revise_prompt, cwd=cwd, cfg=cfg,
            label=f"{label}/revise-r{round_num}",
            phase_name=phase_name, task_type="reviser",
            task_id=rev_id,
        )
        git_commit(cwd, f"bureau: {label} revision r{round_num}")
        STATE.update_task_status(rev_id, TaskStatus.COMPLETED)

        # ── Judge ──────────────────────────────────────────────────────
        judge_id = f"{task_node.id}-judge-r{round_num}"
        judge_node = TaskNode(
            id=judge_id,
            label=f"judge r{round_num}",
            description="Evaluate revisions",
            task_type="judge",
            parent_id=task_node.id,
            worktree_path=task_node.worktree_path,
            max_revision_rounds=0,
        )
        STATE.add_task(judge_node)
        STATE.update_task_status(judge_id, TaskStatus.RUNNING)

        judge_prompt = context + textwrap.dedent(f"""\
            ## Your role: JUDGE

            A batch of edits was applied based on review feedback.
            Review the current state.

            ### Original task
            {task_description}

            ### Edits that were applied
        """)
        for j, edit in enumerate(substantive):
            judge_prompt += (f"    {j+1}. [{edit.severity}] "
                             f"{edit.file}: {edit.description}\n")
        judge_prompt += "\nDecide: accept (keep) or reject (revert)?"

        verdict = await run_structured_agent(
            prompt=judge_prompt, schema=JUDGE_SCHEMA,
            cwd=cwd, cfg=cfg,
            label=f"{label}/judge-r{round_num}",
            phase_name=phase_name, task_type="judge",
            task_id=judge_id,
        )

        v = "accept"
        if isinstance(verdict, dict):
            v = verdict.get("verdict", "accept")
            reason = verdict.get("reason", "")
            log.info(f"  [{label}] judge: {v}"
                     f"{' — ' + reason[:80] if reason else ''}")

        STATE.update_task_status(judge_id, TaskStatus.COMPLETED)

        if v == "reject" and pre_commit:
            log.info(f"  [{label}] reverting revision")
            git_restore_to(cwd, pre_commit)

    log.info(f"  [{label}] critique cycle complete ({max_rounds} rounds)")

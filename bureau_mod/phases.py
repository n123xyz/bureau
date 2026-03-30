"""Phase runner: execute phases sequentially."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from bureau_mod.config import Config, Critic, Phase
from bureau_mod.decompose import work_node
from bureau_mod.git_utils import git_commit
from bureau_mod.state import STATE, TaskNode, emit_event
from bureau_mod.worktree import WorktreeManager

log = logging.getLogger("bureau")


async def run_phase(
    *,
    problem: str,
    phase: Phase,
    prev_phases: list[str],
    critics: list[Critic],
    cwd: str,
    cfg: Config,
    worktree_mgr: WorktreeManager | None = None,
) -> None:
    STATE.current_phase = phase.name
    emit_event("phase", {"name": phase.name})

    log.info(f"{'='*60}")
    log.info(f"PHASE: {phase.name}")
    log.info(f"{'='*60}")

    _cleanup_bureau_temps(cwd)

    # Create root task for this phase — visible in the web UI immediately
    task_id = f"{phase.name}-{uuid.uuid4().hex[:8]}"
    root_task = TaskNode(
        id=task_id,
        label=phase.name,
        description=phase.goal,
        max_revision_rounds=cfg.max_revision_rounds,
    )
    STATE.add_task(root_task)

    # depth=0: work node handles execution, critique, and delegation
    await work_node(
        problem=problem, phase=phase, prev_phases=prev_phases,
        task_description=phase.goal, critics=critics,
        cwd=cwd, cfg=cfg, label=phase.name, depth=0,
        worktree_mgr=worktree_mgr,
        task_id=task_id,
    )

    git_commit(cwd, f"bureau: phase {phase.name} complete")


def _cleanup_bureau_temps(cwd: str) -> None:
    try:
        for f in Path(cwd).glob("_bureau_*"):
            f.unlink()
    except Exception:
        pass

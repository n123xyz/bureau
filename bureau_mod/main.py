"""CLI entry point, logging setup, and main execution loop."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

from bureau_mod.config import Config, Critic, Phase
from bureau_mod.git_utils import git_commit, git_init_if_needed
from bureau_mod.phases import run_phase
import bureau_mod.rate_limit as rate_limit_mod
from bureau_mod.state import STATE
import bureau_mod.state as state_mod
from bureau_mod.web import start_web_server
from bureau_mod.worktree import WorktreeManager

log = logging.getLogger("bureau")


def setup_logging(config_dir: Path) -> Path:
    log_dir = config_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"

    log.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    ))
    log.addHandler(console)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    log.addHandler(fh)

    log.info(f"Log file: {log_file}")
    return log_file


async def run(args: argparse.Namespace) -> None:
    config_dir = Path(args.config_dir).resolve()
    work_dir = Path(args.work_dir).resolve()

    setup_logging(config_dir)

    problem_file = config_dir / "problem.md"
    phases_file = config_dir / "phases.json"
    critics_file = config_dir / "critics.json"
    config_file = config_dir / "config.json"

    for f in [problem_file, phases_file, critics_file]:
        if not f.exists():
            log.error(f"Missing {f}")
            sys.exit(1)

    if not work_dir.exists():
        log.info(f"Creating work directory: {work_dir}")
        work_dir.mkdir(parents=True)

    problem = problem_file.read_text().strip()
    phases = [Phase(**p) for p in json.loads(phases_file.read_text())]
    critics = [Critic(role=c["role"], prompt=c["prompt"],
                       globs=c.get("globs"))
                for c in json.loads(critics_file.read_text())]
    cfg = Config.load(config_file)

    if args.model:
        cfg.model = args.model
    if args.timeout:
        cfg.timeout = args.timeout
    if args.port:
        cfg.web_port = args.port

    rate_limit_mod.WEB_PORT = cfg.web_port

    # Initialize global state
    STATE.config_dir = str(config_dir)
    STATE.work_dir = str(work_dir)
    state_mod.AGENT_SEMAPHORE = asyncio.Semaphore(cfg.max_parallel)

    log.info(f"Config:  {config_dir}")
    log.info(f"Work:    {work_dir}")
    log.info(f"Phases:  {[p.name for p in phases]}")
    log.info(f"Critics: {[c.role for c in critics]}")
    log.info(f"Model:   {cfg.model}  effort={cfg.effort}  "
             f"max_depth={cfg.max_depth}  timeout={cfg.timeout}s")
    log.info(f"Worktrees: {'enabled' if cfg.use_worktrees else 'disabled'}")

    # Log per-task-type config
    for tt, tc in cfg.task_type_defaults.items():
        parts = []
        if tc.model:
            parts.append(f"model={tc.model}")
        if tc.effort:
            parts.append(f"effort={tc.effort}")
        if tc.thinking:
            parts.append(f"thinking={tc.thinking}")
        if parts:
            log.info(f"  {tt}: {', '.join(parts)}")

    if cfg.usage_cap.max_cost_per_hour:
        log.info(f"  Cost cap: ${cfg.usage_cap.max_cost_per_hour:.2f}/hr")
    if cfg.usage_cap.max_utilization:
        log.info(f"  Utilization cap: {cfg.usage_cap.max_utilization:.0%}")

    cwd = str(work_dir)
    git_init_if_needed(cwd)

    worktree_mgr = None
    if cfg.use_worktrees:
        worktree_mgr = WorktreeManager(work_dir)

    web_runner = await start_web_server(cfg.web_port)

    # Determine phase range
    names = [p.name for p in phases]

    if args.only_phase:
        if args.only_phase not in names:
            log.error(f"Unknown phase '{args.only_phase}'. Have: {names}")
            sys.exit(1)
        idx = names.index(args.only_phase)
        start, end = idx, idx + 1
    elif args.start_phase:
        if args.start_phase not in names:
            log.error(f"Unknown phase '{args.start_phase}'. Have: {names}")
            sys.exit(1)
        start, end = names.index(args.start_phase), len(phases)
    elif args.resume_from:
        if args.resume_from not in names:
            log.error(f"Unknown phase '{args.resume_from}'. Have: {names}")
            sys.exit(1)
        start, end = names.index(args.resume_from), len(phases)
    else:
        start, end = 0, len(phases)

    prev_phases: list[str] = names[:start]

    try:
        for phase in phases[start:end]:
            if STATE.stopping:
                break
            await run_phase(
                problem=problem,
                phase=phase,
                prev_phases=prev_phases,
                critics=critics,
                cwd=cwd, cfg=cfg,
                worktree_mgr=worktree_mgr,
            )
            prev_phases.append(phase.name)
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("Stopping...")
    finally:
        STATE.stopping = True
        # Interrupt any still-running agents
        from bureau_mod.state import interrupt_all_clients
        await interrupt_all_clients()

        if worktree_mgr:
            await worktree_mgr.cleanup_all()
        git_commit(cwd, "bureau: final state")
        log.info(f"DONE. agents={STATE.total_agents}  "
                 f"commits={STATE.total_commits}  "
                 f"api_cost=${STATE.total_cost:.2f}  "
                 f"tokens={STATE.total_input_tokens}in/"
                 f"{STATE.total_output_tokens}out")

        checkpoint_path = config_dir / "checkpoint.json"
        STATE.save_checkpoint(checkpoint_path)

        await web_runner.cleanup()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bureau: Hierarchical multi-phase agent orchestrator (v3)",
    )
    p.add_argument("config_dir", type=str, help="Bureau config directory")
    p.add_argument("work_dir", type=str, help="Project work directory")
    p.add_argument("--start-phase", type=str, help="Start from this phase")
    p.add_argument("--only-phase", type=str, help="Run only this phase")
    p.add_argument("--resume-from", type=str, help="Resume from this phase")
    p.add_argument("--model", type=str, help="Override model")
    p.add_argument("--timeout", type=float, help="Per-agent timeout (seconds)")
    p.add_argument("--port", type=int, default=8765,
                   help="Web control server port")
    p.add_argument("--resume", type=str, help="Resume from checkpoint file")

    args = p.parse_args()

    cfg = Path(args.config_dir)
    if not cfg.exists():
        p.error(f"Config dir not found: {cfg}")
    for name in ["problem.md", "phases.json", "critics.json"]:
        if not (cfg / name).exists():
            p.error(f"Missing {cfg / name}")

    asyncio.run(_run_with_signals(args))


async def _run_with_signals(args: argparse.Namespace) -> None:
    """Wrapper that installs asyncio-native signal handlers."""
    from bureau_mod.state import request_stop
    import bureau_mod.state as _st

    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, request_stop)

    _st.MAIN_TASK = asyncio.current_task()
    try:
        await run(args)
    except asyncio.CancelledError:
        log.info("Cancelled by signal")

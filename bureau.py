#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["claude-agent-sdk", "aiohttp", "ollama", "ollama-mcp-bridge>=0.11.2"]
# ///
"""
Bureau: Hierarchical multi-phase agent orchestrator for software development (v3).

Features:
  - Multi-phase execution with configurable phases (eg. spec, interface, test, impl, debug)
  - Per-phase and per-task-type model/effort/thinking configuration
  - Hierarchical dynamic work-decomposition tree within each phase
  - Dynamic parallel execution of tasks with grouping by affected files
  - Critic/review/judge cycle at each leaf node with custom critics
  - LLM rate limit detection with exponential backoff
  - SDK-based utilization tracking for subscription plans
  - Cost/usage cap with automatic pause
  - SSE-powered real-time web dashboard with:
    - Full task tree with expand/collapse
    - Streaming agent output
    - Git log with click-to-expand diffs
    - Skip / stop-revising controls for running agents
    - Model/effort/thinking display per agent
    - Worktree vs main-tree indicator
    - File dependency and concurrency group display
  - Git worktrees for parallel isolation
  - Session checkpoint/restore

Control server endpoints:
  GET  /              — SPA dashboard (HTML)
  GET  /api/state     — Full state as JSON
  GET  /api/events    — SSE event stream
  GET  /api/gitlog    — Git log --oneline
  GET  /api/gitdiff   — Diff for a commit: ?hash=abc123
  GET  /api/task_output — Full output for a task: ?task_id=...
  POST /api/skip      — Skip a task (interrupts if running): {"task_id": "..."}
  POST /api/stop_revising — Stop revision, keep current: {"task_id": "..."}
  POST /api/pause     — Pause execution
  POST /api/resume    — Resume execution
  POST /api/reduce_rounds — Set revision rounds: {"rounds": 1}
  POST /api/checkpoint— Save state to disk
  POST /api/stop      — Graceful shutdown

Usage:
  uv run bureau.py ./config ./my_project
  uv run bureau.py ./config ./my_project --port 8765
  uv run bureau.py ./config ./my_project --resume checkpoint.json
"""

from bureau_mod.main import main

if __name__ == "__main__":
    main()


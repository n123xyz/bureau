"""Task state management, SSE broadcast, and global shared state."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger("bureau")


# ═══════════════════════════════════════════════════════════════════════════
# Task Status and Node
# ═══════════════════════════════════════════════════════════════════════════

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskNode:
    """Represents a node in the task tree."""
    id: str
    label: str
    description: str
    task_type: str = ""  # planner, executor, critic, judge, reviser
    status: TaskStatus = TaskStatus.PENDING
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    worktree_path: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    revision_round: int = 0
    max_revision_rounds: int = 3
    # Model info
    model: str = ""
    effort: str = ""
    thinking: str = ""
    # Dependency and concurrency info
    file_reads: list[str] = field(default_factory=list)
    file_writes: list[str] = field(default_factory=list)
    concurrency_group: int | None = None
    # Control flags
    stop_revising: bool = False
    # Accumulated agent output (tool actions, text)
    output_lines: list[str] = field(default_factory=list)
    # Full prompt (stored for UI inspection, not broadcast via SSE)
    prompt: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "task_type": self.task_type,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "children": self.children,
            "worktree_path": self.worktree_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "revision_round": self.revision_round,
            "max_revision_rounds": self.max_revision_rounds,
            "model": self.model,
            "effort": self.effort,
            "thinking": self.thinking,
            "file_reads": self.file_reads,
            "file_writes": self.file_writes,
            "concurrency_group": self.concurrency_group,
            "stop_revising": self.stop_revising,
            "output_lines": self.output_lines[-200:],  # cap for serialization
            "has_prompt": bool(self.prompt),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Bureau State
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BureauState:
    """Central state management for the bureau."""
    tasks: dict[str, TaskNode] = field(default_factory=dict)
    root_task_ids: list[str] = field(default_factory=list)
    paused: bool = False
    stopping: bool = False
    current_phase: str = ""
    total_cost: float = 0.0
    total_agents: int = 0
    total_commits: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    rate_limit_until: float | None = None
    rate_limit_backoff: float = 60.0
    rate_limit_utilization: float | None = None
    rate_limit_type: str | None = None
    config_dir: str = ""
    work_dir: str = ""
    checkpoint_file: str | None = None
    # Usage cap tracking
    cost_window_start: float = field(default_factory=time.time)
    cost_in_window: float = 0.0

    def add_task(self, task: TaskNode) -> None:
        self.tasks[task.id] = task
        if task.parent_id:
            parent = self.tasks.get(task.parent_id)
            if parent and task.id not in parent.children:
                parent.children.append(task.id)
                # Re-emit parent so clients see the updated children list
                emit_event("task", parent.to_dict())
        else:
            if task.id not in self.root_task_ids:
                self.root_task_ids.append(task.id)
        emit_event("task", task.to_dict())

    def get_task(self, task_id: str) -> TaskNode | None:
        return self.tasks.get(task_id)

    def update_task_status(self, task_id: str, status: TaskStatus,
                           error: str | None = None) -> None:
        task = self.tasks.get(task_id)
        if task:
            task.status = status
            if status == TaskStatus.RUNNING and task.started_at is None:
                task.started_at = time.time()
            if status in (TaskStatus.COMPLETED, TaskStatus.FAILED,
                          TaskStatus.SKIPPED):
                task.finished_at = time.time()
            if error:
                task.error = error
            emit_event("task", task.to_dict())

    def to_dict(self) -> dict:
        return {
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
            "root_task_ids": self.root_task_ids,
            "paused": self.paused,
            "stopping": self.stopping,
            "current_phase": self.current_phase,
            "total_cost": self.total_cost,
            "total_agents": self.total_agents,
            "total_commits": self.total_commits,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "rate_limit_until": self.rate_limit_until,
            "rate_limit_backoff": self.rate_limit_backoff,
            "rate_limit_utilization": self.rate_limit_utilization,
            "rate_limit_type": self.rate_limit_type,
            "config_dir": self.config_dir,
            "work_dir": self.work_dir,
        }

    def save_checkpoint(self, path: Path) -> None:
        data = self.to_dict()
        data["checkpoint_time"] = datetime.now().isoformat()
        path.write_text(json.dumps(data, indent=2))
        log.info(f"Checkpoint saved: {path}")

    @classmethod
    def load_checkpoint(cls, path: Path) -> BureauState:
        data = json.loads(path.read_text())
        state = cls()
        state.paused = data.get("paused", False)
        state.current_phase = data.get("current_phase", "")
        state.total_cost = data.get("total_cost", 0.0)
        state.total_agents = data.get("total_agents", 0)
        state.total_commits = data.get("total_commits", 0)
        state.total_input_tokens = data.get("total_input_tokens", 0)
        state.total_output_tokens = data.get("total_output_tokens", 0)
        state.config_dir = data.get("config_dir", "")
        state.work_dir = data.get("work_dir", "")

        for task_id, td in data.get("tasks", {}).items():
            task = TaskNode(
                id=td["id"], label=td["label"],
                description=td.get("description", ""),
                task_type=td.get("task_type", ""),
                status=TaskStatus(td["status"]),
                parent_id=td.get("parent_id"),
                children=td.get("children", []),
                worktree_path=td.get("worktree_path"),
                started_at=td.get("started_at"),
                finished_at=td.get("finished_at"),
                error=td.get("error"),
                revision_round=td.get("revision_round", 0),
                max_revision_rounds=td.get("max_revision_rounds", 3),
                model=td.get("model", ""),
                effort=td.get("effort", ""),
                thinking=td.get("thinking", ""),
                file_reads=td.get("file_reads", []),
                file_writes=td.get("file_writes", []),
                concurrency_group=td.get("concurrency_group"),
            )
            state.tasks[task_id] = task
        state.root_task_ids = data.get("root_task_ids", [])
        return state


# ═══════════════════════════════════════════════════════════════════════════
# Global state and SSE broadcast
# ═══════════════════════════════════════════════════════════════════════════

STATE: BureauState = BureauState()
PAUSE_EVENT: asyncio.Event = asyncio.Event()
PAUSE_EVENT.set()  # Start unpaused

AGENT_SEMAPHORE: asyncio.Semaphore | None = None

# Active SDK clients for interrupt support (task_id -> client)
ACTIVE_CLIENTS: dict[str, Any] = {}

# The main execution task, so we can cancel it on shutdown
MAIN_TASK: asyncio.Task | None = None

# SSE broadcast: set of asyncio.Queue, one per connected client
SSE_CLIENTS: set[asyncio.Queue] = set()


async def interrupt_all_clients() -> None:
    """Interrupt all active SDK clients for fast shutdown."""
    for task_id, client in list(ACTIVE_CLIENTS.items()):
        try:
            log.info(f"  Interrupting agent for {task_id}")
            await client.interrupt()
        except Exception:
            pass


def request_stop() -> None:
    """Set stopping flag, interrupt clients, cancel main task."""
    if STATE.stopping:
        # Second signal — force exit
        log.warning("Force exit (second signal)")
        import os
        os._exit(1)
    STATE.stopping = True
    PAUSE_EVENT.set()
    emit_event("stats", {"paused": False, "stopping": True})
    # Schedule client interrupts and task cancellation
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass
    if loop and loop.is_running():
        loop.create_task(_do_stop())


async def _do_stop() -> None:
    """Async shutdown: interrupt clients, cancel main task."""
    await interrupt_all_clients()
    if MAIN_TASK and not MAIN_TASK.done():
        MAIN_TASK.cancel()


def emit_event(event_type: str, data: Any) -> None:
    """Push an SSE event to all connected clients."""
    payload = json.dumps({"type": event_type, "data": data})
    dead: list[asyncio.Queue] = []
    for q in SSE_CLIENTS:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        SSE_CLIENTS.discard(q)


def emit_task_output(task_id: str, line: str) -> None:
    """Append an output line to a task and broadcast it."""
    task = STATE.get_task(task_id)
    if task:
        task.output_lines.append(line)
    emit_event("output", {"task_id": task_id, "line": line})


def emit_stats() -> None:
    """Broadcast current stats."""
    emit_event("stats", {
        "total_agents": STATE.total_agents,
        "total_cost": STATE.total_cost,
        "total_commits": STATE.total_commits,
        "total_input_tokens": STATE.total_input_tokens,
        "total_output_tokens": STATE.total_output_tokens,
        "rate_limit_utilization": STATE.rate_limit_utilization,
        "rate_limit_type": STATE.rate_limit_type,
        "rate_limit_until": STATE.rate_limit_until,
        "paused": STATE.paused,
        "current_phase": STATE.current_phase,
    })

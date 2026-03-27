"""Configuration with per-phase/task-type model overrides and usage caps."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentConfig:
    """Override model/effort/thinking for a specific context."""
    model: str | None = None
    effort: str | None = None
    thinking: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> AgentConfig:
        return cls(
            model=d.get("model"),
            effort=d.get("effort"),
            thinking=d.get("thinking"),
        )


@dataclass
class Phase:
    name: str
    goal: str


@dataclass
class Critic:
    role: str
    prompt: str


@dataclass
class UsageCap:
    """Caps on cost/usage that trigger automatic pause."""
    max_cost_per_hour: float | None = None
    max_utilization: float | None = None  # 0.0-1.0, from SDK RateLimitInfo
    pause_on_cap: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> UsageCap:
        return cls(
            max_cost_per_hour=d.get("max_cost_per_hour"),
            max_utilization=d.get("max_utilization"),
            pause_on_cap=d.get("pause_on_cap", True),
        )


@dataclass
class Config:
    # Global defaults
    model: str = "claude-opus-4-6"
    effort: str = "medium"
    thinking: str = "adaptive"
    max_depth: int = 3
    max_revision_rounds: int = 2
    max_split_pieces: int = 8
    timeout: float = 600.0
    stall_timeout: float = 120.0
    permission_mode: str = "bypassPermissions"
    parallel_critics: bool = True
    parallel_subtasks: bool = True
    max_parallel: int = 16
    use_worktrees: bool = True
    web_port: int = 8765
    setting_sources: list[str] = field(
        default_factory=lambda: ["user", "project", "local"]
    )

    # Per task-type defaults (across all phases)
    # Keys: planner, judge, executor, critic, reviser
    task_type_defaults: dict[str, AgentConfig] = field(default_factory=dict)

    # Per phase overrides
    # {phase_name: {"model": "...", "effort": "...", "thinking": "...",
    #               "task_types": {task_type: {"model": "...", ...}}}}
    phase_overrides: dict[str, dict] = field(default_factory=dict)

    # Usage caps
    usage_cap: UsageCap = field(default_factory=UsageCap)

    @classmethod
    def load(cls, path: Path) -> Config:
        cfg = cls()
        if path.exists():
            raw = json.loads(path.read_text())
            # Simple scalar fields
            for k in ("model", "effort", "thinking", "max_depth",
                       "max_revision_rounds", "max_split_pieces", "timeout",
                       "stall_timeout", "permission_mode", "parallel_critics",
                       "parallel_subtasks", "max_parallel", "use_worktrees",
                       "web_port", "setting_sources"):
                if k in raw:
                    setattr(cfg, k, raw[k])

            # Task-type defaults
            for tt, td in raw.get("task_type_defaults", {}).items():
                cfg.task_type_defaults[tt] = AgentConfig.from_dict(td)

            # Phase overrides
            cfg.phase_overrides = raw.get("phase_overrides", {})

            # Usage caps
            if "usage_cap" in raw:
                cfg.usage_cap = UsageCap.from_dict(raw["usage_cap"])

        return cfg

    def resolve(self, phase_name: str, task_type: str) -> tuple[str, str, str]:
        """Resolve (model, effort, thinking) for a specific phase + task_type.

        Priority (highest wins):
          1. phase_overrides[phase].task_types[task_type]
          2. phase_overrides[phase] top-level
          3. task_type_defaults[task_type]
          4. global defaults
        """
        model, effort, thinking = self.model, self.effort, self.thinking

        # Task-type defaults
        if task_type in self.task_type_defaults:
            tc = self.task_type_defaults[task_type]
            if tc.model:
                model = tc.model
            if tc.effort:
                effort = tc.effort
            if tc.thinking:
                thinking = tc.thinking

        # Phase-level override
        if phase_name in self.phase_overrides:
            po = self.phase_overrides[phase_name]
            if "model" in po:
                model = po["model"]
            if "effort" in po:
                effort = po["effort"]
            if "thinking" in po:
                thinking = po["thinking"]

            # Phase + task_type override
            task_types = po.get("task_types", {})
            if task_type in task_types:
                tt = task_types[task_type]
                if "model" in tt:
                    model = tt["model"]
                if "effort" in tt:
                    effort = tt["effort"]
                if "thinking" in tt:
                    thinking = tt["thinking"]

        return model, effort, thinking

    def thinking_config(self, thinking_str: str | None = None) -> dict[str, Any] | None:
        """Convert thinking string to SDK config dict."""
        t = thinking_str or self.thinking
        if t == "disabled":
            return {"type": "disabled"}
        if t == "adaptive":
            return {"type": "adaptive"}
        try:
            budget = int(t)
            return {"type": "enabled", "budget_tokens": budget}
        except (ValueError, TypeError):
            return {"type": "disabled"}

"""Context preamble and JSON extraction utilities."""

from __future__ import annotations

import json
import re
import textwrap
from typing import Any

from bureau_mod.config import Phase
from bureau_mod.git_utils import repo_file_listing


CONTEXT_TEMPLATE = textwrap.dedent("""\
    ## Context

    You are one agent in a multi-phase, hierarchically-decomposed software
    development pipeline. Here is your situational context.

    ### Problem description
    {problem}

    ### Current phase: {phase_name}
    Phase goal: {phase_goal}

    ### Completed phases
    {prev_phases}

    ### Your position in the decomposition tree
    {tree_path}

    ### Current repository contents
    {repo_listing}

    ### Working directory
    Your working directory is: {cwd}
    You MUST only read and write files within this directory.
    Do NOT create, modify, or reference files outside it.

    ---

""")


def make_context(
    problem: str,
    phase: Phase,
    prev_phases: list[str],
    tree_path: str = "(root)",
    cwd: str | None = None,
) -> str:
    listing = repo_file_listing(cwd) if cwd else "(unknown)"
    return CONTEXT_TEMPLATE.format(
        problem=problem,
        phase_name=phase.name,
        phase_goal=phase.goal,
        prev_phases=("\n".join(f"- {p}" for p in prev_phases) if prev_phases
                     else "(none yet)"),
        tree_path=tree_path,
        repo_listing=listing,
        cwd=cwd or "(unknown)",
    )


def extract_json(text: str) -> Any:
    """Try to extract JSON from agent text output."""
    for m in re.finditer(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL):
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    for m in re.finditer(r"\[[\s\S]*\]", text):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    for m in re.finditer(r"\{[\s\S]*\}", text):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    return None

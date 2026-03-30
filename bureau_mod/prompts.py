"""Shared prompt fragments used by worker, critic, and reviser agents.

Kept in a leaf module (no bureau_mod imports) to avoid circular imports
between decompose.py and revision.py.
"""

from __future__ import annotations

import textwrap

# Name of the file the work agent writes when it has subtasks to delegate.
SUBTASKS_FILE = "_bureau_subtasks.json"

HIERARCHY_CONTEXT = textwrap.dedent("""\
    ### How this system works
    This is a hierarchical task decomposition system. Work is organized as
    a tree of nodes. Each node has a budget (~N lines of file content it may
    write) and operates at one level of the hierarchy.

    A node's job is to do the work that is appropriate at its level —
    creating the structures, interfaces, and content that belong at that
    granularity — and then delegate sub-levels to child nodes when natural
    subdivisions exist (chapters→sections, packages→modules, components→
    subcomponents, etc.). Delegation is NOT just an overflow mechanism for
    when you run out of budget; it reflects the natural structure of the
    work.

    Each child node gets the same budget and can delegate further, so the
    total capacity of the tree grows with depth. A node should NOT try to
    do everything itself — it should do its level well and let children
    handle the details.
""")

SUBTASKS_SCHEMA_DOC = textwrap.dedent(f"""\
    ### Delegation via `{SUBTASKS_FILE}`
    To delegate, write a file called `{SUBTASKS_FILE}` containing a JSON
    array under a "subtasks" key. Each element is an object with:
      - "description": what the child node should do
      - "reads": list of files it needs as input (files that already exist)
      - "writes": list of files it will create or modify

    Example:
    ```json
    {{"subtasks": [
      {{"description": "Implement the parser module",
       "reads": ["src/ast.py"],
       "writes": ["src/parser.py", "src/tokenizer.py"]}},
      {{"description": "Write unit tests for the parser",
       "reads": ["src/parser.py", "src/tokenizer.py"],
       "writes": ["tests/test_parser.py"]}}
    ]}}
    ```

    The subtasks file is reviewed alongside your other output. A reviewer
    may revise it (merge, split, or adjust subtasks) before children run.

    Files listed in subtasks' "writes" DO NOT EXIST YET — child agents
    will create them later. Forward references to those files (imports,
    calls, type annotations, doc cross-references) are expected and fine.
""")

PARALLELISM_RULES = textwrap.dedent("""\
    ### Parallelism rules for subtasks
    - Each subtask must write to DIFFERENT files. No two subtasks may write
      the same file.
    - Multiple subtasks MAY read the same file — readers don't block each
      other.
    - A subtask that reads a file written by another subtask will wait for
      that writer to finish first. Use this to express dependencies.
    - All files must be within the project working directory.
""")

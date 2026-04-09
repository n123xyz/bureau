"""Git helpers: init, commit, log, diff, file listing."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from bureau_mod.state import STATE, emit_event

log = logging.getLogger("bureau")


def git_init_if_needed(cwd: str) -> None:
    if not (Path(cwd) / ".git").exists():
        subprocess.run(["git", "init"], cwd=cwd, capture_output=True, timeout=30)
        subprocess.run(
            ["git", "config", "--local", "commit.gpgsign", "false"],
            cwd=cwd, capture_output=True, timeout=10,
        )
        subprocess.run(
            ["git", "config", "--local", "core.hooksPath", "/dev/null"],
            cwd=cwd, capture_output=True, timeout=10,
        )
        # Write a broad .gitignore for common build artifacts
        gitignore = Path(cwd) / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(_GITIGNORE)
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "bureau: init",
             "--no-gpg-sign", "--no-verify"],
            cwd=cwd, capture_output=True, timeout=30,
        )

    # Even if the repo already existed, ensure _bureau_* files are ignored.
    # Without this, _bureau_subtasks.json gets committed and propagated to
    # worktrees, causing every child task to re-read the parent's plan and
    # spawn duplicate subtask trees (N^depth explosion).
    _ensure_bureau_gitignore(cwd)


# Lines that MUST be present in .gitignore for bureau to work correctly.
_BUREAU_IGNORE_LINES = ["_bureau_*", ".claude/"]


def _ensure_bureau_gitignore(cwd: str) -> None:
    """Ensure the .gitignore contains all required ignore patterns.

    Checks every non-blank, non-comment line from _GITIGNORE and appends
    any that are missing.  This covers build artifacts (which break
    merging) and bureau-internal files.
    """
    gitignore = Path(cwd) / ".gitignore"
    existing = ""
    if gitignore.exists():
        existing = gitignore.read_text()

    # No .gitignore at all (or blank) → write the full default template
    if not existing.strip():
        gitignore.write_text(_GITIGNORE)
        return

    # Find every pattern line from the template that's missing
    existing_lines = set(existing.splitlines())
    required = [
        line for line in _GITIGNORE.splitlines()
        if line and not line.startswith("#")
    ]
    missing = [line for line in required if line not in existing_lines]
    if not missing:
        return

    addition = "\n# Bureau defaults (auto-added)\n"
    addition += "\n".join(missing) + "\n"

    if not existing.endswith("\n"):
        addition = "\n" + addition

    gitignore.write_text(existing + addition)


# Common build/IDE artifacts to ignore across many languages
_GITIGNORE = """\
# Rust
target/
**/*.rs.bk

# C/C++
*.o
*.obj
*.so
*.dylib
*.dll
*.a
*.lib
*.exe
*.out

# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
*.whl
.venv/
venv/

# Node / JS / TS
node_modules/
npm-debug.log*
yarn-error.log*
.next/
dist/

# Java / JVM
*.class
*.jar
*.war
*.ear
hs_err_pid*

# Go
*.test

# General build dirs
build/
out/
cmake-build-*/

# IDE / editor
.idea/
.vscode/
*.swp
*.swo
*~
.DS_Store
Thumbs.db

# Claude / Bureau internals
.claude/
_bureau_*
"""


def git_commit(cwd: str, message: str) -> None:
    try:
        subprocess.run(["git", "add", "-A"], cwd=cwd, capture_output=True, timeout=30)
        r = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=cwd, capture_output=True, timeout=30,
        )
        if r.returncode != 0:
            subprocess.run(
                ["git", "commit", "-m", message, "--no-gpg-sign", "--no-verify"],
                cwd=cwd, capture_output=True, timeout=30,
            )
            STATE.total_commits += 1
            log.info(f"  git: {message[:72]}")
            # Get the new commit hash for SSE
            head = git_get_head(cwd)
            if head:
                emit_event("git_commit", {"hash": head[:8], "message": message})
    except Exception as e:
        log.warning(f"  git commit failed: {e}")


def git_get_head(cwd: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, capture_output=True, timeout=10, text=True,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def git_restore_to(cwd: str, commit: str) -> bool:
    try:
        r = subprocess.run(
            ["git", "reset", "--hard", commit],
            cwd=cwd, capture_output=True, timeout=30, text=True,
        )
        if r.returncode == 0:
            log.info(f"  git: restored to {commit[:10]}")
            return True
        log.warning(f"  git restore failed: {r.stderr}")
    except Exception as e:
        log.warning(f"  git restore failed: {e}")
    return False


def repo_file_listing(cwd: str) -> str:
    try:
        # Use git ls-files to respect .gitignore rules
        r = subprocess.run(
            ["git", "ls-files", "--cached", "--others",
             "--exclude-standard"],
            cwd=cwd, capture_output=True, timeout=10, text=True,
        )
        files = sorted(r.stdout.strip().splitlines()) if r.stdout.strip() else []
        if not files:
            return "(empty repository — no files yet)"
        if len(files) > 200:
            return "\n".join(files[:200]) + f"\n... and {len(files)-200} more files"
        return "\n".join(files)
    except Exception:
        return "(could not list files)"


def git_log_oneline(cwd: str, max_count: int = 100) -> list[dict[str, str]]:
    """Return git log as list of {hash, message}."""
    try:
        r = subprocess.run(
            ["git", "log", f"--max-count={max_count}", "--oneline",
             "--no-decorate"],
            cwd=cwd, capture_output=True, timeout=10, text=True,
        )
        if r.returncode != 0:
            return []
        entries = []
        for line in r.stdout.strip().splitlines():
            if " " in line:
                h, msg = line.split(" ", 1)
                entries.append({"hash": h, "message": msg})
        return entries
    except Exception:
        return []


def git_show_diff(cwd: str, commit_hash: str) -> str:
    """Return the diff for a specific commit."""
    # Validate hash is alphanumeric to prevent injection
    if not commit_hash.replace("-", "").isalnum():
        return "(invalid commit hash)"
    try:
        r = subprocess.run(
            ["git", "show", "--stat", "--patch", commit_hash],
            cwd=cwd, capture_output=True, timeout=15, text=True,
        )
        if r.returncode == 0:
            return r.stdout[:50000]  # cap output size
        return f"(error: {r.stderr[:200]})"
    except Exception as e:
        return f"(error: {e})"


def git_ls_tree(cwd: str) -> list[str]:
    """Return list of files tracked in HEAD."""
    try:
        r = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD"],
            cwd=cwd, capture_output=True, timeout=10, text=True,
        )
        if r.returncode == 0 and r.stdout.strip():
            return sorted(r.stdout.strip().splitlines())
    except Exception:
        pass
    return []


def read_repo_file(cwd: str, path: str,
                   extra_roots: list[str] | None = None) -> str | None:
    """Read a file, searching work_dir and worktree roots.

    Accepts relative paths (searched in each root) and absolute paths
    (verified to be under a known root).  Prevents path traversal.
    """
    roots = [Path(cwd).resolve()]
    for r in (extra_roots or []):
        rp = Path(r).resolve()
        if rp not in roots:
            roots.append(rp)

    p = Path(path)

    # Absolute path: check it's under one of the known roots
    if p.is_absolute():
        target = p.resolve()
        for root in roots:
            if str(target).startswith(str(root) + "/"):
                if target.is_file():
                    try:
                        return target.read_text(errors="replace")[:200000]
                    except Exception:
                        return None
        return None

    # Relative path: try each root
    for root in roots:
        target = (root / path).resolve()
        if not str(target).startswith(str(root) + "/"):
            continue  # traversal attempt
        if target.is_file():
            try:
                return target.read_text(errors="replace")[:200000]
            except Exception:
                pass
    return None

"""Git worktree management for parallel agent isolation."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Awaitable

log = logging.getLogger("bureau")

# Type for an async conflict resolver callback:
#   (repo_path: str, conflicted_files: list[str]) -> bool
ConflictResolver = Callable[[str, list[str]], Awaitable[bool]]

# Pattern matching git conflict markers
_CONFLICT_RE = re.compile(r"^<{7} ", re.MULTILINE)


class WorktreeManager:
    """Manages git worktrees for parallel agent isolation."""

    def __init__(self, main_repo: Path, worktree_base: Path | None = None):
        self.main_repo = main_repo
        self.worktree_base = worktree_base or Path(tempfile.mkdtemp(
            prefix="bureau-worktrees-"
        ))
        self.active_worktrees: dict[str, Path] = {}
        self._lock = asyncio.Lock()

    async def create_worktree(self, task_id: str, branch_name: str | None = None
                              ) -> Path:
        """Create an isolated worktree for a task."""
        async with self._lock:
            if task_id in self.active_worktrees:
                return self.active_worktrees[task_id]

            if branch_name is None:
                branch_name = f"task-{task_id}-{uuid.uuid4().hex[:8]}"

            worktree_path = self.worktree_base / task_id

            try:
                proc = await asyncio.create_subprocess_exec(
                    "git", "worktree", "add", "-b", branch_name,
                    str(worktree_path), "HEAD",
                    cwd=str(self.main_repo),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate()

                if proc.returncode != 0:
                    proc = await asyncio.create_subprocess_exec(
                        "git", "worktree", "add",
                        str(worktree_path), "HEAD",
                        cwd=str(self.main_repo),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    _, stderr = await proc.communicate()
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"Failed to create worktree: {stderr.decode()}"
                        )

                for config_cmd in [
                    ["git", "config", "--local", "commit.gpgsign", "false"],
                    ["git", "config", "--local", "core.hooksPath", "/dev/null"],
                ]:
                    await asyncio.create_subprocess_exec(
                        *config_cmd,
                        cwd=str(worktree_path),
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )

                self.active_worktrees[task_id] = worktree_path
                
                # Symlink the main repo's .venv to the worktree to avoid re-syncing over network
                main_venv = self.main_repo / ".venv"
                if main_venv.exists():
                    try:
                        worktree_venv = worktree_path / ".venv"
                        if not worktree_venv.exists():
                            worktree_venv.symlink_to(main_venv, target_is_directory=True)
                            log.debug(f"Symlinked .venv from {main_venv} to {worktree_path}")
                    except Exception as ve:
                        log.warning(f"Failed to symlink .venv to worktree: {ve}")

                log.debug(f"Created worktree for {task_id}: {worktree_path}")
                return worktree_path

            except Exception as e:
                log.error(f"Failed to create worktree for {task_id}: {e}")
                return self.main_repo

    async def merge_and_cleanup(self, task_id: str,
                                commit_message: str | None = None,
                                conflict_resolver: ConflictResolver | None = None,
                                ) -> bool:
        """Merge worktree changes back to main and clean up.

        If a merge conflict occurs:
          1. Try agent-based resolution via *conflict_resolver* callback
          2. Fall back to ``git merge -X theirs`` (prefer the task branch)
        """
        async with self._lock:
            if task_id not in self.active_worktrees:
                return True

            worktree_path = self.active_worktrees[task_id]

            try:
                await asyncio.create_subprocess_exec(
                    "git", "add", "-A",
                    cwd=str(worktree_path),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )

                proc = await asyncio.create_subprocess_exec(
                    "git", "diff", "--cached", "--quiet",
                    cwd=str(worktree_path),
                )
                await proc.communicate()

                if proc.returncode != 0:
                    msg = commit_message or f"bureau: task {task_id} complete"
                    await asyncio.create_subprocess_exec(
                        "git", "commit", "-m", msg,
                        "--no-gpg-sign", "--no-verify",
                        cwd=str(worktree_path),
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )

                proc = await asyncio.create_subprocess_exec(
                    "git", "rev-parse", "--abbrev-ref", "HEAD",
                    cwd=str(worktree_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await proc.communicate()
                branch_name = stdout.decode().strip()

                merge_msg = f"Merge {branch_name} (task {task_id})"
                proc = await asyncio.create_subprocess_exec(
                    "git", "merge", "--no-ff", "-m", merge_msg,
                    branch_name,
                    cwd=str(self.main_repo),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()

                if proc.returncode != 0:
                    log.warning(
                        f"Merge conflict for {task_id}: {stderr.decode()}"
                    )
                    merged = await self._handle_merge_conflict(
                        branch_name, task_id, conflict_resolver,
                    )
                    if not merged:
                        return False

                # ── Cleanup worktree and branch ────────────────────────
                await asyncio.create_subprocess_exec(
                    "git", "worktree", "remove", "--force", str(worktree_path),
                    cwd=str(self.main_repo),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )

                await asyncio.create_subprocess_exec(
                    "git", "branch", "-D", branch_name,
                    cwd=str(self.main_repo),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )

                del self.active_worktrees[task_id]
                log.debug(f"Merged and cleaned up worktree for {task_id}")
                return True

            except Exception as e:
                log.error(f"Failed to merge/cleanup worktree for {task_id}: {e}")
                return False

    # ── Conflict resolution helpers ────────────────────────────────────

    async def _handle_merge_conflict(
        self,
        branch_name: str,
        task_id: str,
        conflict_resolver: ConflictResolver | None,
    ) -> bool:
        """Attempt to resolve a merge conflict.

        Strategy:
          1. If a *conflict_resolver* callback is provided, let it edit the
             conflicted files in-place, then verify no markers remain.
          2. If that fails (or no resolver), abort and retry with
             ``-X theirs`` which prefers the task branch for every hunk.
          3. If *that* also fails, abort and give up.
        """
        repo = self.main_repo

        # ── Strategy 1: agent-based resolution ─────────────────────────
        if conflict_resolver:
            conflicted = await self._conflicted_files()
            if conflicted:
                log.info(
                    f"  merge: attempting agent resolution for {task_id} "
                    f"({len(conflicted)} files: "
                    f"{', '.join(conflicted[:5])})"
                )
                try:
                    ok = await conflict_resolver(str(repo), conflicted)
                except Exception as exc:
                    log.warning(f"  merge: resolver raised: {exc}")
                    ok = False

                if ok:
                    # Verify no conflict markers leaked through
                    still_dirty = self._files_with_markers(
                        repo, conflicted,
                    )
                    if still_dirty:
                        log.warning(
                            f"  merge: conflict markers remain in "
                            f"{still_dirty} — aborting agent resolution"
                        )
                    else:
                        # Stage everything and finish the merge commit
                        for f in conflicted:
                            await asyncio.create_subprocess_exec(
                                "git", "add", f,
                                cwd=str(repo),
                                stdout=asyncio.subprocess.DEVNULL,
                                stderr=asyncio.subprocess.DEVNULL,
                            )
                        proc = await asyncio.create_subprocess_exec(
                            "git", "commit", "--no-gpg-sign",
                            "--no-verify", "--no-edit",
                            cwd=str(repo),
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        _, stderr = await proc.communicate()
                        if proc.returncode == 0:
                            log.info(
                                f"  merge: agent resolved conflicts for "
                                f"{task_id}"
                            )
                            return True
                        log.warning(
                            f"  merge: commit after agent resolution "
                            f"failed: {stderr.decode()}"
                        )

        # ── Abort the current (failed) merge before retrying ───────────
        await asyncio.create_subprocess_exec(
            "git", "merge", "--abort",
            cwd=str(repo),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        # ── Strategy 2: re-merge preferring the task branch ───────────
        log.info(f"  merge: retrying with -X theirs for {task_id}")
        merge_msg = f"Merge {branch_name} (task {task_id}, auto-resolved)"
        proc = await asyncio.create_subprocess_exec(
            "git", "merge", "--no-ff", "-X", "theirs",
            "-m", merge_msg, branch_name,
            cwd=str(repo),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode == 0:
            log.info(f"  merge: resolved with -X theirs for {task_id}")
            return True

        log.error(
            f"  merge: -X theirs also failed for {task_id}: "
            f"{stderr.decode()}"
        )
        await asyncio.create_subprocess_exec(
            "git", "merge", "--abort",
            cwd=str(repo),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        return False

    async def _conflicted_files(self) -> list[str]:
        """Return relative paths of files with unresolved merge conflicts."""
        proc = await asyncio.create_subprocess_exec(
            "git", "diff", "--name-only", "--diff-filter=U",
            cwd=str(self.main_repo),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        if not stdout or not stdout.strip():
            return []
        return stdout.decode().strip().splitlines()

    @staticmethod
    def _files_with_markers(repo: Path, paths: list[str]) -> list[str]:
        """Return subset of *paths* that still contain conflict markers."""
        bad: list[str] = []
        for p in paths:
            try:
                content = (repo / p).read_text(errors="replace")
                if _CONFLICT_RE.search(content):
                    bad.append(p)
            except OSError:
                pass
        return bad

    async def cleanup_all(self) -> None:
        """Clean up all worktrees (for shutdown)."""
        for task_id in list(self.active_worktrees.keys()):
            try:
                await self.merge_and_cleanup(task_id)
            except Exception as e:
                log.warning(f"Failed to cleanup worktree {task_id}: {e}")

        try:
            if self.worktree_base.exists():
                shutil.rmtree(self.worktree_base, ignore_errors=True)
        except Exception:
            pass

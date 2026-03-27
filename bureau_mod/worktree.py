"""Git worktree management for parallel agent isolation."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

log = logging.getLogger("bureau")


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
                log.debug(f"Created worktree for {task_id}: {worktree_path}")
                return worktree_path

            except Exception as e:
                log.error(f"Failed to create worktree for {task_id}: {e}")
                return self.main_repo

    async def merge_and_cleanup(self, task_id: str,
                                commit_message: str | None = None) -> bool:
        """Merge worktree changes back to main and clean up."""
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

                proc = await asyncio.create_subprocess_exec(
                    "git", "merge", "--no-ff", "-m",
                    f"Merge {branch_name} (task {task_id})",
                    branch_name,
                    cwd=str(self.main_repo),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()

                if proc.returncode != 0:
                    log.warning(f"Merge conflict for {task_id}: {stderr.decode()}")
                    await asyncio.create_subprocess_exec(
                        "git", "merge", "--abort",
                        cwd=str(self.main_repo),
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    return False

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

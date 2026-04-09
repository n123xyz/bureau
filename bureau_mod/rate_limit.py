"""Rate limit detection, backoff, and usage cap enforcement."""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from typing import TYPE_CHECKING

from bureau_mod.state import STATE, PAUSE_EVENT, emit_event, emit_stats

if TYPE_CHECKING:
    from bureau_mod.config import UsageCap

log = logging.getLogger("bureau")

WEB_PORT: int = 8765  # updated by main


class RateLimitError(Exception):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def is_rate_limit_error(exc: Exception) -> tuple[bool, float | None]:
    """Check if an exception indicates a rate limit.
    Returns (is_rate_limit, retry_after_seconds)."""
    msg = str(exc).lower()
    rate_limit_patterns = [
        "rate limit", "rate_limit", "too many requests", "429",
        "quota exceeded", "credit", "usage limit", "capacity", "overloaded",
    ]
    if any(p in msg for p in rate_limit_patterns):
        retry_match = re.search(r"retry.{0,10}?(\d+)\s*(second|minute|hour)?", msg)
        if retry_match:
            value = int(retry_match.group(1))
            unit = retry_match.group(2) or "second"
            if "minute" in unit:
                value *= 60
            elif "hour" in unit:
                value *= 3600
            return True, float(value)
        return True, None
    return False, None


async def wait_for_rate_limit() -> None:
    """Wait until rate limit expires, with user feedback."""
    if STATE.rate_limit_until is None:
        return

    now = time.time()
    if now >= STATE.rate_limit_until:
        STATE.rate_limit_until = None
        STATE.rate_limit_backoff = 60.0
        return

    wait_time = STATE.rate_limit_until - now
    log.warning(f"Rate limited. Waiting {wait_time:.0f}s before retry...")
    emit_stats()

    while time.time() < STATE.rate_limit_until:
        if STATE.stopping:
            raise asyncio.CancelledError("Stopping")
        remaining = STATE.rate_limit_until - time.time()
        if remaining <= 0:
            break
        await asyncio.sleep(min(10, remaining))

    STATE.rate_limit_until = None
    STATE.rate_limit_backoff = 60.0
    log.info("Rate limit wait complete, resuming...")
    emit_stats()


def set_rate_limit(retry_after: float | None = None) -> None:
    """Set rate limit state with exponential backoff."""
    if retry_after:
        wait_time = retry_after
    else:
        wait_time = STATE.rate_limit_backoff * (1 + random.random() * 0.5)
        STATE.rate_limit_backoff = min(STATE.rate_limit_backoff * 2, 3600)

    STATE.rate_limit_until = time.time() + wait_time
    log.warning(f"Rate limit hit. Will retry in {wait_time:.0f}s")
    emit_stats()


def update_utilization(utilization: float | None, rate_limit_type: str | None,
                       status: str | None) -> None:
    """Update utilization info from SDK RateLimitEvent."""
    STATE.rate_limit_utilization = utilization
    STATE.rate_limit_type = rate_limit_type
    emit_event("rate_limit", {
        "utilization": utilization,
        "type": rate_limit_type,
        "status": status,
    })
    emit_stats()


def check_usage_cap(cfg_cap: UsageCap) -> bool:
    """Check if usage cap is exceeded. Returns True if should pause."""
    # Cost per hour cap
    if cfg_cap.max_cost_per_hour is not None:
        elapsed_h = (time.time() - STATE.cost_window_start) / 3600.0
        if elapsed_h > 0:
            rate = STATE.cost_in_window / elapsed_h
            if rate > cfg_cap.max_cost_per_hour:
                log.warning(f"Cost cap hit: ${rate:.2f}/hr > "
                            f"${cfg_cap.max_cost_per_hour:.2f}/hr")
                return True
        # Reset window every hour
        if elapsed_h >= 1.0:
            STATE.cost_window_start = time.time()
            STATE.cost_in_window = 0.0

    # Utilization cap
    if (cfg_cap.max_utilization is not None
            and STATE.rate_limit_utilization is not None):
        if STATE.rate_limit_utilization > cfg_cap.max_utilization:
            log.warning(f"Utilization cap hit: "
                        f"{STATE.rate_limit_utilization:.1%} > "
                        f"{cfg_cap.max_utilization:.1%}")
            return True

    return False


async def enforce_usage_cap(cfg_cap: UsageCap) -> None:
    """Check cap and pause if exceeded.

    Pauses work and polls every 30s.  Resumes automatically when the cap
    is no longer exceeded (e.g. utilization drops after a quota reset) or
    when manually un-paused via the web UI.
    """
    if not cfg_cap.pause_on_cap:
        return
    if not check_usage_cap(cfg_cap):
        return

    log.warning("Usage cap exceeded — pausing (will auto-resume when cap clears)")
    STATE.paused = True
    PAUSE_EVENT.clear()
    emit_stats()

    while True:
        # Unblock immediately if manually resumed via web UI
        try:
            await asyncio.wait_for(PAUSE_EVENT.wait(), timeout=30)
            break  # manually resumed
        except asyncio.TimeoutError:
            pass

        if STATE.stopping:
            raise asyncio.CancelledError("Stopping")

        # Re-check the cap — utilization may have dropped
        if not check_usage_cap(cfg_cap):
            log.info("Usage cap no longer exceeded — auto-resuming")
            STATE.paused = False
            PAUSE_EVENT.set()
            emit_stats()
            break

"""Sliding-window rate limiter for the Groq Free Tier.

Groq's free tier imposes strict requests-per-minute limits.  This
module provides a lightweight, sequential rate limiter that pauses
automatically when approaching the limit and resumes when safe.

Design:
- Sliding window of request timestamps (last 60 seconds).
- Blocks via time.sleep() when the window is full.
- No external dependencies.
- Thread-safe.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)

# Default: 28 RPM leaves 2-request headroom under Groq's 30 RPM limit
DEFAULT_MAX_RPM = 28


class RateLimiter:
    """Sliding-window rate limiter.

    Tracks request timestamps in a deque and blocks when the count
    within the last 60 seconds reaches ``max_rpm``.
    """

    def __init__(self, max_rpm: int = DEFAULT_MAX_RPM) -> None:
        self._max_rpm = max_rpm
        self._window: deque[float] = deque()
        self._lock = threading.Lock()
        self._total_waits = 0
        self._total_wait_seconds = 0.0
        logger.info("RateLimiter initialized — max_rpm=%d", max_rpm)

    def _prune_window(self) -> None:
        """Remove timestamps older than 60 seconds."""
        cutoff = time.monotonic() - 60.0
        while self._window and self._window[0] < cutoff:
            self._window.popleft()

    def wait_if_needed(self) -> float:
        """Block until it is safe to send another request.

        Returns:
            The number of seconds waited (0.0 if no wait was needed).
        """
        with self._lock:
            self._prune_window()

            if len(self._window) < self._max_rpm:
                return 0.0

            # Calculate how long to wait until the oldest request
            # falls outside the 60-second window
            oldest = self._window[0]
            wait_until = oldest + 60.0
            wait_seconds = max(0.0, wait_until - time.monotonic() + 0.5)  # +0.5s safety margin

        if wait_seconds > 0:
            logger.info(
                "Rate limit approached (%d/%d in window). Pausing %.1fs…",
                len(self._window),
                self._max_rpm,
                wait_seconds,
            )
            self._total_waits += 1
            self._total_wait_seconds += wait_seconds
            time.sleep(wait_seconds)

        return wait_seconds

    def record_request(self) -> None:
        """Record that a request was just sent."""
        with self._lock:
            self._window.append(time.monotonic())
            self._prune_window()

    def remaining(self) -> int:
        """Return how many requests can be sent in the current window."""
        with self._lock:
            self._prune_window()
            return max(0, self._max_rpm - len(self._window))

    def stats(self) -> dict[str, float | int]:
        """Return rate limiter statistics."""
        with self._lock:
            self._prune_window()
            return {
                "max_rpm": self._max_rpm,
                "current_window_count": len(self._window),
                "remaining": max(0, self._max_rpm - len(self._window)),
                "total_waits": self._total_waits,
                "total_wait_seconds": round(self._total_wait_seconds, 1),
            }

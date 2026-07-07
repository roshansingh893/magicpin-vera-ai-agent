"""File-based LLM response cache — eliminates redundant Groq API calls.

Every successful LLM response is cached to disk, keyed on the SHA-256
hash of (merchant_id, category_slug, trigger_kind, customer_id,
prompt_version).  Identical requests return the cached response
instantly without touching the API.

Design constraints:
- Zero external dependencies (stdlib only).
- Thread-safe via threading.Lock.
- Human-readable JSON files (one per cached response).
- Survives process restarts (file-based persistence).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default cache directory relative to project root
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "responses"


class ResponseCache:
    """File-based LLM response cache.

    Each cached response is stored as a JSON file named by its cache
    key hash.  The file contains both the response data and metadata
    (timestamp, original request parameters) for auditability.
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        logger.info("ResponseCache initialized — dir=%s", self._cache_dir)

    @staticmethod
    def _build_cache_key(
        merchant_id: str,
        category_slug: str,
        trigger_kind: str,
        customer_id: str | None = None,
        prompt_version: str = "default",
    ) -> str:
        """Build a deterministic SHA-256 cache key from request parameters."""
        parts = [
            f"merchant:{merchant_id}",
            f"category:{category_slug}",
            f"trigger:{trigger_kind}",
            f"customer:{customer_id or 'none'}",
            f"prompt:{prompt_version}",
        ]
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        """Return the file path for a given cache key."""
        return self._cache_dir / f"{key}.json"

    def has(
        self,
        merchant_id: str,
        category_slug: str,
        trigger_kind: str,
        customer_id: str | None = None,
        prompt_version: str = "default",
    ) -> bool:
        """Check if a cached response exists for the given parameters."""
        key = self._build_cache_key(
            merchant_id, category_slug, trigger_kind, customer_id, prompt_version
        )
        return self._cache_path(key).exists()

    def get(
        self,
        merchant_id: str,
        category_slug: str,
        trigger_kind: str,
        customer_id: str | None = None,
        prompt_version: str = "default",
    ) -> Optional[dict[str, Any]]:
        """Retrieve a cached response, or None if not cached.

        Returns:
            The cached response dict (with 'body', 'cta', etc.) or None.
        """
        key = self._build_cache_key(
            merchant_id, category_slug, trigger_kind, customer_id, prompt_version
        )
        path = self._cache_path(key)

        with self._lock:
            if not path.exists():
                self._misses += 1
                return None

            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._hits += 1
                logger.debug("Cache HIT — key=%s…", key[:12])
                return data.get("response")
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Cache read failed for %s: %s", key[:12], exc)
                self._misses += 1
                return None

    def put(
        self,
        merchant_id: str,
        category_slug: str,
        trigger_kind: str,
        response: dict[str, Any],
        customer_id: str | None = None,
        prompt_version: str = "default",
        test_id: str = "",
    ) -> None:
        """Store a successful LLM response in the cache.

        Args:
            merchant_id: Merchant identifier.
            category_slug: Category slug.
            trigger_kind: Trigger kind string.
            response: The composed message dict to cache.
            customer_id: Optional customer identifier.
            prompt_version: Prompt version label.
            test_id: Optional test identifier for traceability.
        """
        key = self._build_cache_key(
            merchant_id, category_slug, trigger_kind, customer_id, prompt_version
        )
        path = self._cache_path(key)

        entry = {
            "cache_key": key,
            "test_id": test_id,
            "merchant_id": merchant_id,
            "category_slug": category_slug,
            "trigger_kind": trigger_kind,
            "customer_id": customer_id,
            "prompt_version": prompt_version,
            "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response": response,
        }

        with self._lock:
            path.write_text(
                json.dumps(entry, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        logger.debug("Cache PUT — key=%s… test_id=%s", key[:12], test_id)

    def warm_from_submission(self, submission_path: Path | str) -> int:
        """Pre-populate the cache from an existing submission.jsonl file.

        Reads each line of the JSONL, extracts the test_id and response
        fields, and stores them.  Since submission.jsonl doesn't contain
        full request parameters, we use the test_id as a simplified key.

        Args:
            submission_path: Path to submission.jsonl.

        Returns:
            Number of entries loaded into the cache.
        """
        path = Path(submission_path)
        if not path.exists():
            logger.warning("Submission file not found for cache warming: %s", path)
            return 0

        loaded = 0
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            try:
                entry = json.loads(line)
                test_id = entry.get("test_id", "")
                if not test_id:
                    continue

                # Store with test_id-based key for lookup by test_id
                tid_path = self._cache_dir / f"tid_{test_id}.json"
                cache_entry = {
                    "cache_key": f"tid_{test_id}",
                    "test_id": test_id,
                    "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "source": "submission_warmup",
                    "response": entry,
                }
                tid_path.write_text(
                    json.dumps(cache_entry, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                loaded += 1
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Failed to warm cache from line: %s", exc)

        logger.info("Cache warmed from submission: %d entries loaded", loaded)
        return loaded

    def get_by_test_id(self, test_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a cached response by test_id (from submission warmup).

        Args:
            test_id: The test identifier (e.g., "T01").

        Returns:
            The cached response dict or None.
        """
        path = self._cache_dir / f"tid_{test_id}.json"
        with self._lock:
            if not path.exists():
                return None
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._hits += 1
                return data.get("response")
            except (json.JSONDecodeError, KeyError):
                return None

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total_files = len(list(self._cache_dir.glob("*.json")))
            return {
                "cache_dir": str(self._cache_dir),
                "total_entries": total_files,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(
                    self._hits / max(self._hits + self._misses, 1), 4
                ),
            }

    def clear(self) -> int:
        """Remove all cached responses. Returns count of deleted entries."""
        with self._lock:
            count = 0
            for f in self._cache_dir.glob("*.json"):
                f.unlink()
                count += 1
            self._hits = 0
            self._misses = 0
        logger.info("Cache cleared — %d entries removed", count)
        return count

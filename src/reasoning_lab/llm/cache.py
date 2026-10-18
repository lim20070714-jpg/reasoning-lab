"""Content-addressed disk cache for completions.

Reasoning sweeps re-run the same prompts (ablations, strategy comparisons,
replaying a saved tree). Caching keyed on the prompt+params fingerprint makes
experiments reproducible and keeps API spend flat while iterating.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .base import Completion

log = logging.getLogger(__name__)


class CompletionCache:
    def __init__(self, root: Path | str, *, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def path_for(self, fingerprint: str) -> Path:
        return self.root / fingerprint[:2] / f"{fingerprint}.json"

    def get(self, fingerprint: str) -> Completion | None:
        if not self.enabled or not fingerprint:
            return None
        file = self.path_for(fingerprint)
        if not file.is_file():
            self.misses += 1
            return None
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:  # a corrupt entry must not kill a sweep
            log.warning("cache entry unreadable (%s): %s", file, exc)
            self.misses += 1
            return None
        self.hits += 1
        cached = Completion.model_validate(data)
        cached.cached = True
        return cached

    def put(self, completion: Completion) -> None:
        if not self.enabled or not completion.fingerprint or completion.cached:
            return
        file = self.path_for(completion.fingerprint)
        try:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(
                json.dumps(completion.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            log.warning("could not write cache entry %s: %s", file, exc)

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}

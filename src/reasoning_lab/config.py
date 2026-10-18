"""Environment-driven configuration with a dependency-free ``.env`` loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, Field

MOCK_PROVIDER: Final = "mock"


def load_dotenv(path: str | os.PathLike[str] | None = None, *, override: bool = False) -> dict[str, str]:
    """Parse a ``KEY=value`` file into ``os.environ``. Returns the loaded pairs."""
    file = Path(path) if path else Path.cwd() / ".env"
    loaded: dict[str, str] = {}
    if not file.is_file():
        return loaded
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


class LLMSettings(BaseModel):
    provider: Literal["openai", "mock"] = "openai"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    top_p: float | None = None
    max_tokens: int = 768
    timeout_s: float = 120.0
    max_retries: int = 3
    max_concurrency: int = 4
    #: per-request sleep for the offline mock, so a human can steer a search
    mock_latency_ms: float = 0.0
    cache_dir: Path | None = None
    user_agent: str = "reasoning-lab"

    @property
    def is_mock(self) -> bool:
        return self.provider == MOCK_PROVIDER or self.base_url.startswith("mock://")

    def require_credentials(self) -> None:
        if self.is_mock:
            return
        if not self.api_key:
            raise RuntimeError(
                "No API key configured. Set REASONING_LAB_API_KEY (see .env.example) "
                "or use REASONING_LAB_API_BASE_URL=mock:// for offline runs."
            )


class Settings(BaseModel):
    run_dir: Path = Field(default_factory=lambda: Path(_env("REASONING_LAB_RUN_DIR") or "runs"))
    data_dir: Path = Field(default_factory=lambda: Path(_env("REASONING_LAB_DATA_DIR") or "data"))
    llm: LLMSettings = Field(default_factory=LLMSettings)
    verbose: bool = False

    @classmethod
    def from_env(cls, dotenv_path: str | os.PathLike[str] | None = None) -> Settings:
        load_dotenv(dotenv_path)
        cache_raw = _env("REASONING_LAB_CACHE_DIR", "cache/llm")
        llm = LLMSettings(
            provider=MOCK_PROVIDER if _env("REASONING_LAB_PROVIDER") == MOCK_PROVIDER else "openai",
            base_url=_env("REASONING_LAB_API_BASE_URL", "https://api.openai.com/v1"),
            api_key=_env("REASONING_LAB_API_KEY"),
            model=_env("REASONING_LAB_MODEL", "gpt-4o-mini"),
            max_tokens=int(_env("REASONING_LAB_MAX_TOKENS", "768") or 768),
            timeout_s=float(_env("REASONING_LAB_TIMEOUT_SECONDS", "120") or 120),
            max_retries=int(_env("REASONING_LAB_MAX_RETRIES", "3") or 3),
            max_concurrency=int(_env("REASONING_LAB_MAX_CONCURRENCY", "4") or 4),
            mock_latency_ms=float(_env("REASONING_LAB_MOCK_LATENCY_MS", "0") or 0),
            cache_dir=Path(cache_raw) if cache_raw else None,
        )
        return cls(
            run_dir=Path(_env("REASONING_LAB_RUN_DIR") or "runs"),
            data_dir=Path(_env("REASONING_LAB_DATA_DIR") or "data"),
            llm=llm,
            verbose=_env("REASONING_LAB_VERBOSE").lower() in {"1", "true", "yes"},
        )

    def ensure_dirs(self) -> Settings:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.llm.cache_dir:
            self.llm.cache_dir.mkdir(parents=True, exist_ok=True)
        return self

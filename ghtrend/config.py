import os
from dataclasses import dataclass, field

import yaml


@dataclass
class Config:
    daily_top_n: int = 10
    languages: list[str] | None = None
    semantic_top_k: int = 10
    embedding_model: str = "BAAI/bge-m3"
    trending_since: str = "daily"
    data_dir: str = "data"


def load_config(path: str = "config.yaml") -> Config:
    data = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    known = {f for f in Config().__dataclass_fields__}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in data.items() if k in known}
    return Config(**filtered)


def get_webhook_url() -> str:
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        raise RuntimeError("环境变量 DISCORD_WEBHOOK_URL 未设置")
    return url


def get_github_token() -> str | None:
    return os.environ.get("GITHUB_TOKEN")

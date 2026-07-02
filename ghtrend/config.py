import os
from dataclasses import dataclass, field

import yaml


@dataclass
class Config:
    daily_top_n: int = 10
    languages: list[str] | None = None
    semantic_top_k: int = 10
    embedding_model: str = "BAAI/bge-m3"
    # 本机语义搜索可改用在线 OpenAI 兼容 embedding 服务(留空则用本地模型)
    embedding_api_base: str = ""
    embedding_api_model: str = "bge-m3"
    # 实时 GitHub 语义搜索:LLM 提关键词 + reranker 重排(留空 llm_api_base 则禁用该模式)
    llm_api_base: str = ""
    llm_model: str = "deepseek-v4-flash"
    # 摘要可单独指定 LLM(留空则沿用 llm_api_base / llm_model)
    summary_api_base: str = ""
    summary_model: str = ""
    # 每日推送的描述翻译可单独指定 LLM(留空则沿用 llm_api_base / llm_model)
    translate_api_base: str = ""
    translate_model: str = ""
    reranker_api_base: str = ""
    reranker_model: str = "bge-reranker-v2-m3"
    realtime_per_keyword: int = 10
    web_port: int = 8005
    # 监听地址:127.0.0.1=仅本机;0.0.0.0=所有接口(局域网可访问)
    web_host: str = "127.0.0.1"
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


def get_embedding_api_key() -> str | None:
    return os.environ.get("EMBEDDING_API_KEY")


def get_llm_api_key() -> str | None:
    return os.environ.get("LLM_API_KEY")


def get_reranker_api_key() -> str | None:
    return os.environ.get("RERANKER_API_KEY")


def get_summary_api_key() -> str | None:
    return os.environ.get("SUMMARY_API_KEY")


def get_translate_api_key() -> str | None:
    return os.environ.get("TRANSLATE_API_KEY")

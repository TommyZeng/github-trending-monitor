import os
import textwrap
from ghtrend import config


def test_load_config_reads_yaml(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent("""
        daily_top_n: 5
        languages: [Python, Rust]
        semantic_top_k: 3
        embedding_model: "test-model"
        embedding_api_base: "http://svc:7000/v1"
        embedding_api_model: "bge-m3"
        trending_since: "weekly"
        data_dir: "mydata"
    """))
    cfg = config.load_config(str(p))
    assert cfg.daily_top_n == 5
    assert cfg.languages == ["Python", "Rust"]
    assert cfg.semantic_top_k == 3
    assert cfg.embedding_model == "test-model"
    assert cfg.embedding_api_base == "http://svc:7000/v1"
    assert cfg.embedding_api_model == "bge-m3"
    assert cfg.trending_since == "weekly"
    assert cfg.data_dir == "mydata"


def test_embedding_api_base_defaults_empty(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("daily_top_n: 7\n")
    cfg = config.load_config(str(p))
    assert cfg.embedding_api_base == ""


def test_get_embedding_api_key_reads_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test")
    assert config.get_embedding_api_key() == "sk-test"
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    assert config.get_embedding_api_key() is None


def test_load_config_uses_defaults_for_missing(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("daily_top_n: 7\n")
    cfg = config.load_config(str(p))
    assert cfg.daily_top_n == 7
    assert cfg.languages is None
    assert cfg.semantic_top_k == 10


def test_secret_helpers_read_env(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://hook")
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    assert config.get_webhook_url() == "https://hook"
    assert config.get_github_token() == "tok"


def test_get_github_token_optional(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert config.get_github_token() is None

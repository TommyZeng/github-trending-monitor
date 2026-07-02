import json
import os
import numpy as np
from ghtrend import collect, store
from ghtrend.config import Config


class _DummyEmbedder:
    def encode(self, texts):
        return np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)


def test_run_collects_dedups_and_notifies(tmp_path):
    cfg = Config(daily_top_n=2, data_dir=str(tmp_path))
    sent = {}

    def fake_fetcher(since, language=None, session=None):
        # a/x 今日新增更多(200>10),尽管总 star 更少
        return [{"full_name": "a/x", "url": "u", "stars_today": 200},
                {"full_name": "b/y", "url": "u", "stars_today": 10}]

    def fake_enricher(full_name, token=None, session=None):
        stars = {"a/x": 50, "b/y": 99}[full_name]
        return {"full_name": full_name, "url": f"https://github.com/{full_name}",
                "description": "d", "stars": stars, "language": "Python",
                "topics": ["ai"], "readme_excerpt": "r"}

    def fake_notifier(webhook_url, projects, title, session=None):
        sent["title"] = title
        sent["names"] = [p["full_name"] for p in projects]

    collect.run(cfg, "https://hook", "tok", _DummyEmbedder(), today="2026-06-02",
                fetcher=fake_fetcher, enricher=fake_enricher, notifier=fake_notifier,
                translate=lambda s: f"[zh]{s}")

    projects, embs = store.load(str(tmp_path))
    assert len(projects) == 2
    assert embs.shape[0] == 2
    # 推送按「今日新增 star」降序:a/x(200) 在 b/y(10) 之前,即便 a/x 总 star 更少
    assert sent["names"] == ["a/x", "b/y"]
    assert all(p["stars_today"] in (200, 10) for p in projects)  # 持久化了今日增量
    assert "2026-06-02" in sent["title"]
    # 每个项目都写入了中文描述
    assert all(p["description_zh"] == "[zh]d" for p in projects)


def test_run_skips_failed_enrichment(tmp_path):
    cfg = Config(data_dir=str(tmp_path))

    def fake_fetcher(since, language=None, session=None):
        return [{"full_name": "a/x", "url": "u"}, {"full_name": "bad/repo", "url": "u"}]

    def fake_enricher(full_name, token=None, session=None):
        return None if full_name == "bad/repo" else {
            "full_name": full_name, "url": "u", "description": "d", "stars": 1,
            "language": "Go", "topics": [], "readme_excerpt": ""}

    collect.run(cfg, "https://hook", None, _DummyEmbedder(), today="2026-06-02",
                fetcher=fake_fetcher, enricher=fake_enricher,
                notifier=lambda *a, **k: None, translate=lambda s: s)

    projects, _ = store.load(str(tmp_path))
    assert [p["full_name"] for p in projects] == ["a/x"]


def _mini_fetch_enrich():
    def fake_fetcher(since, language=None, session=None):
        return [{"full_name": "a/x", "url": "u", "stars_today": 5}]
    def fake_enricher(full_name, token=None, session=None):
        return {"full_name": full_name, "url": "u", "description": "A fast tool",
                "stars": 1, "language": "Go", "topics": [], "readme_excerpt": ""}
    return fake_fetcher, fake_enricher


def test_run_prefers_batch_translator(tmp_path):
    cfg = Config(data_dir=str(tmp_path))
    fetcher, enricher = _mini_fetch_enrich()
    collect.run(cfg, "https://hook", None, _DummyEmbedder(), today="2026-07-02",
                fetcher=fetcher, enricher=enricher, notifier=lambda *a, **k: None,
                translate=lambda s: "[google]" + s,
                translate_batch=lambda texts: ["[llm]" + t for t in texts])
    projects, _ = store.load(str(tmp_path))
    assert projects[0]["description_zh"] == "[llm]A fast tool"


def test_run_falls_back_to_per_item_when_batch_fails(tmp_path):
    cfg = Config(data_dir=str(tmp_path))
    fetcher, enricher = _mini_fetch_enrich()
    collect.run(cfg, "https://hook", None, _DummyEmbedder(), today="2026-07-02",
                fetcher=fetcher, enricher=enricher, notifier=lambda *a, **k: None,
                translate=lambda s: "[google]" + s,
                translate_batch=lambda texts: None)   # LLM 挂了
    projects, _ = store.load(str(tmp_path))
    assert projects[0]["description_zh"] == "[google]A fast tool"


def test_build_translate_batch_prefers_translate_config():
    cfg = Config(translate_api_base="https://t.example/v1", translate_model="mt",
                 llm_api_base="https://llm.example/v1", llm_model="ml")
    calls = {}
    def fake_llm_batch(texts, base, model, api_key=None):
        calls.update(base=base, model=model, api_key=api_key)
        return ["译"]
    fn = collect.build_translate_batch(cfg, "key-t", llm_batch=fake_llm_batch)
    assert fn(["x"]) == ["译"]
    assert calls == {"base": "https://t.example/v1", "model": "mt", "api_key": "key-t"}


def test_build_translate_batch_falls_back_to_llm_config():
    cfg = Config(llm_api_base="https://llm.example/v1", llm_model="ml")
    calls = {}
    def fake_llm_batch(texts, base, model, api_key=None):
        calls.update(base=base, model=model)
        return ["译"]
    fn = collect.build_translate_batch(cfg, "k", llm_batch=fake_llm_batch)
    fn(["x"])
    assert calls == {"base": "https://llm.example/v1", "model": "ml"}


def test_build_translate_batch_none_without_base_or_key():
    assert collect.build_translate_batch(Config(), "k") is None          # 没配 base
    assert collect.build_translate_batch(
        Config(translate_api_base="https://t.example/v1"), None) is None  # 没 key

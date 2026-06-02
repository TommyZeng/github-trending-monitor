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
        return [{"full_name": "a/x", "url": "u"}, {"full_name": "b/y", "url": "u"}]

    def fake_enricher(full_name, token=None, session=None):
        stars = {"a/x": 50, "b/y": 99}[full_name]
        return {"full_name": full_name, "url": f"https://github.com/{full_name}",
                "description": "d", "stars": stars, "language": "Python",
                "topics": ["ai"], "readme_excerpt": "r"}

    def fake_notifier(webhook_url, projects, title, session=None):
        sent["title"] = title
        sent["names"] = [p["full_name"] for p in projects]

    collect.run(cfg, "https://hook", "tok", _DummyEmbedder(), today="2026-06-02",
                fetcher=fake_fetcher, enricher=fake_enricher, notifier=fake_notifier)

    projects, embs = store.load(str(tmp_path))
    assert len(projects) == 2
    assert embs.shape[0] == 2
    # 推送按 star 降序取 top 2
    assert sent["names"] == ["b/y", "a/x"]
    assert "2026-06-02" in sent["title"]


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
                notifier=lambda *a, **k: None)

    projects, _ = store.load(str(tmp_path))
    assert [p["full_name"] for p in projects] == ["a/x"]

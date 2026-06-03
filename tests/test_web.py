import numpy as np
from fastapi.testclient import TestClient
from ghtrend import web, store
from ghtrend.config import Config


class _DummyEmbedder:
    def encode(self, texts):
        # 查询 "x" -> 偏向第一个项目
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


def _seed(data_dir):
    projects, embs = store.load(data_dir)
    items = [
        {"full_name": "a/x", "url": "https://github.com/a/x", "description": "d1",
         "stars": 10, "language": "Python", "topics": ["ai"], "readme_excerpt": ""},
        {"full_name": "b/y", "url": "https://github.com/b/y", "description": "d2",
         "stars": 20, "language": "Go", "topics": ["cli"], "readme_excerpt": ""},
    ]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    projects, embs = store.upsert(projects, embs, items, vecs, "2026-06-02")
    store.save(data_dir, projects, embs)


def test_index_served(tmp_path):
    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), None)
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "GitHub" in r.text


def test_semantic_search_ranks(tmp_path):
    _seed(str(tmp_path))
    app = web.create_app(Config(data_dir=str(tmp_path), semantic_top_k=2),
                         _DummyEmbedder(), None)
    client = TestClient(app)
    r = client.get("/api/semantic", params={"q": "anything"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert results[0]["full_name"] == "a/x"   # 查询向量[1,0] 最接近 a/x
    assert "score" in results[0]


def test_keyword_search_uses_github(tmp_path, monkeypatch):
    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), "tok")
    monkeypatch.setattr(web.github_enricher, "search_repos",
                        lambda q, token=None, limit=10: [
                            {"full_name": "z/found", "url": "u", "description": "d",
                             "stars": 1, "language": "C", "topics": []}])
    client = TestClient(app)
    r = client.get("/api/keyword", params={"q": "rust"})
    assert r.json()["results"][0]["full_name"] == "z/found"


def test_realtime_uses_injected_fn(tmp_path):
    calls = {}
    def fake_realtime(q):
        calls["q"] = q
        return [{"full_name": "rt/repo", "url": "u", "description": "d", "score": 0.9}]
    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), None,
                         realtime_fn=fake_realtime)
    client = TestClient(app)
    r = client.get("/api/realtime", params={"q": "用Rust写的CLI"})
    assert calls["q"] == "用Rust写的CLI"
    assert r.json()["results"][0]["full_name"] == "rt/repo"


def test_realtime_disabled_when_no_fn(tmp_path):
    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), None)
    client = TestClient(app)
    r = client.get("/api/realtime", params={"q": "x"})
    body = r.json()
    assert body["results"] == []
    assert "error" in body


def _summarize(items):
    for it in items:
        it["summary_zh"] = "中文摘要:" + it["full_name"]
    return items


def test_keyword_applies_summary(tmp_path, monkeypatch):
    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), "tok",
                         summarize_fn=_summarize)
    monkeypatch.setattr(web.github_enricher, "search_repos",
                        lambda q, token=None, limit=10: [
                            {"full_name": "z/found", "url": "u", "description": "raw",
                             "stars": 1, "language": "C", "topics": []}])
    client = TestClient(app)
    r = client.get("/api/keyword", params={"q": "rust"})
    assert r.json()["results"][0]["summary_zh"] == "中文摘要:z/found"


def test_realtime_applies_summary(tmp_path):
    app = web.create_app(
        Config(data_dir=str(tmp_path)), _DummyEmbedder(), None,
        realtime_fn=lambda q: [{"full_name": "rt/repo", "url": "u", "description": "raw"}],
        summarize_fn=_summarize)
    client = TestClient(app)
    r = client.get("/api/realtime", params={"q": "x"})
    assert r.json()["results"][0]["summary_zh"] == "中文摘要:rt/repo"

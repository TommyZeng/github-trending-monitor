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


def test_favorite_then_semantic_finds_it(tmp_path):
    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), None)
    client = TestClient(app)
    proj = {"full_name": "fav/repo", "url": "u", "description": "d",
            "stars": 1, "language": "Go", "topics": ["x"]}
    r = client.post("/api/favorite", json=proj)
    assert r.json()["ok"] is True
    assert r.json()["count"] == 1

    r2 = client.get("/api/semantic", params={"q": "anything"})
    results = r2.json()["results"]
    assert any(p["full_name"] == "fav/repo" and p.get("favorited") for p in results)


def test_favorites_list_returns_saved(tmp_path):
    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), None)
    client = TestClient(app)
    client.post("/api/favorite", json={"full_name": "a/one", "url": "u", "description": "d", "topics": []})
    client.post("/api/favorite", json={"full_name": "b/two", "url": "u", "description": "d", "topics": []})
    r = client.get("/api/favorites")
    results = r.json()["results"]
    assert {p["full_name"] for p in results} == {"a/one", "b/two"}
    assert all(p["favorited"] for p in results)


def test_favorites_list_empty(tmp_path):
    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), None)
    client = TestClient(app)
    assert client.get("/api/favorites").json()["results"] == []


def test_leaderboard_sorts_by_period_stars(tmp_path, monkeypatch):
    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), None)
    captured = {}
    def fake_fetch(since="daily", language=None, session=None):
        captured["since"] = since; captured["language"] = language
        return [
            {"full_name": "a/x", "url": "u", "description": "d", "language": "Go",
             "stars": 100, "stars_today": 50},
            {"full_name": "b/y", "url": "u", "description": "d", "language": "Rust",
             "stars": 999, "stars_today": 300},
        ]
    monkeypatch.setattr(web.trending_fetcher, "fetch_trending", fake_fetch)
    client = TestClient(app)
    r = client.get("/api/leaderboard", params={"since": "weekly", "language": "rust"})
    body = r.json()
    assert captured["since"] == "weekly" and captured["language"] == "rust"
    # 按 period star(stars_today)降序:b/y(300) 在 a/x(50) 前
    assert [p["full_name"] for p in body["results"]] == ["b/y", "a/x"]


def test_leaderboard_bad_since_defaults_daily(tmp_path, monkeypatch):
    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), None)
    monkeypatch.setattr(web.trending_fetcher, "fetch_trending",
                        lambda since="daily", language=None, session=None: [])
    client = TestClient(app)
    assert client.get("/api/leaderboard", params={"since": "bogus"}).json()["since"] == "daily"


def test_library_lists_trending_and_marks_favorited(tmp_path):
    # 往主库(trending)放两个项目
    projects, embs = store.load(str(tmp_path))
    items = [
        {"full_name": "a/x", "url": "u", "description": "d", "stars": 5,
         "language": "Go", "topics": [], "readme_excerpt": ""},
        {"full_name": "b/y", "url": "u", "description": "d", "stars": 9,
         "language": "Rust", "topics": [], "readme_excerpt": ""},
    ]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    projects, embs = store.upsert(projects, embs, items, vecs, "2026-06-04")
    store.save(str(tmp_path), projects, embs)

    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), None)
    client = TestClient(app)
    # 收藏其中一个
    client.post("/api/favorite", json=items[0])

    results = client.get("/api/library").json()["results"]
    assert {p["full_name"] for p in results} == {"a/x", "b/y"}
    by_name = {p["full_name"]: p for p in results}
    assert by_name["a/x"]["favorited"] is True      # 已收藏的被标记
    assert by_name["b/y"]["favorited"] is False


def test_favorite_requires_full_name(tmp_path):
    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), None)
    client = TestClient(app)
    r = client.post("/api/favorite", json={"description": "no name"})
    assert r.status_code == 400
    assert r.json()["ok"] is False


def test_favorite_dedups(tmp_path):
    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), None)
    client = TestClient(app)
    proj = {"full_name": "fav/repo", "url": "u", "description": "d", "topics": []}
    client.post("/api/favorite", json=proj)
    r = client.post("/api/favorite", json=proj)
    assert r.json()["count"] == 1   # 重复收藏不增加


def _seed_kw(data_dir):
    projects, embs = store.load(data_dir)
    items = [
        {"full_name": "rustify/tui-app", "url": "u", "description": "A terminal UI written in Rust",
         "description_zh": "用 Rust 写的终端界面", "stars": 300, "language": "Rust",
         "topics": ["tui", "cli"], "readme_excerpt": ""},
        {"full_name": "pyweb/flask-admin", "url": "u", "description": "Admin panel for Flask",
         "description_zh": "Flask 管理后台", "stars": 800, "language": "Python",
         "topics": ["web"], "readme_excerpt": ""},
        {"full_name": "gogo/rust-bindgen", "url": "u", "description": "Bindings generator",
         "description_zh": "绑定生成器", "stars": 500, "language": "Go",
         "topics": [], "readme_excerpt": ""},
    ]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    projects, embs = store.upsert(projects, embs, items, vecs, "2026-07-08")
    store.save(data_dir, projects, embs)


def test_local_keyword_matches_multiple_fields_sorted_by_stars(tmp_path):
    _seed_kw(str(tmp_path))
    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), None)
    client = TestClient(app)
    # "rust" 命中 full_name(rustify、rust-bindgen)、description、language,按 star 降序
    r = client.get("/api/local_keyword", params={"q": "rust"})
    names = [p["full_name"] for p in r.json()["results"]]
    assert names == ["gogo/rust-bindgen", "rustify/tui-app"]


def test_local_keyword_multi_term_and_semantics(tmp_path):
    _seed_kw(str(tmp_path))
    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), None)
    client = TestClient(app)
    # 多词 AND:两个词都要命中(任意字段)
    r = client.get("/api/local_keyword", params={"q": "rust terminal"})
    assert [p["full_name"] for p in r.json()["results"]] == ["rustify/tui-app"]


def test_local_keyword_matches_chinese_description(tmp_path):
    _seed_kw(str(tmp_path))
    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), None)
    client = TestClient(app)
    r = client.get("/api/local_keyword", params={"q": "管理后台"})
    assert [p["full_name"] for p in r.json()["results"]] == ["pyweb/flask-admin"]


def test_local_keyword_includes_favorites_and_marks(tmp_path):
    _seed_kw(str(tmp_path))
    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), None)
    client = TestClient(app)
    client.post("/api/favorite", json={
        "full_name": "fav/rusty-fav", "url": "u", "description": "rust favorite",
        "stars": 9999, "topics": []})
    r = client.get("/api/local_keyword", params={"q": "rust"})
    results = r.json()["results"]
    assert results[0]["full_name"] == "fav/rusty-fav"   # star 最高排最前
    assert results[0]["favorited"] is True
    assert results[1]["favorited"] is False


def test_local_keyword_blank_query_empty(tmp_path):
    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), None)
    client = TestClient(app)
    assert client.get("/api/local_keyword", params={"q": "  "}).json()["results"] == []

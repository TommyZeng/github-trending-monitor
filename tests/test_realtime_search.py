from ghtrend import realtime_search
from ghtrend.config import Config


def _cfg():
    return Config(llm_api_base="http://llm/v1", reranker_api_base="http://rr/v1",
                  realtime_per_keyword=10, semantic_top_k=2)


def _repo(fn, desc="d"):
    return {"full_name": fn, "url": f"https://github.com/{fn}", "description": desc,
            "stars": 1, "language": "Rust", "topics": ["cli"]}


def test_search_extracts_searches_dedups_and_reranks():
    calls = {"search": []}

    def fake_extract(query, base, model, api_key=None):
        return ["kw-a", "kw-b"]

    def fake_searcher(kw, token=None, limit=10):
        calls["search"].append(kw)
        # 两个关键词返回有重叠的仓库,需去重
        if kw == "kw-a":
            return [_repo("o/alpha"), _repo("o/beta")]
        return [_repo("o/beta"), _repo("o/gamma")]

    def fake_rerank(query, docs, base, model, api_key=None):
        # docs 顺序 = 去重后候选顺序 [alpha, beta, gamma];让 gamma 最高、alpha 次之
        # 返回 (index, score) 乱序,函数内部应已排序好;这里直接给排序结果
        return [(2, 0.9), (0, 0.8), (1, 0.1)]

    results = realtime_search.search(
        "用Rust写的CLI", _cfg(), github_token="t",
        llm_api_key="lk", reranker_api_key="rk",
        extract=fake_extract, searcher=fake_searcher, rerank_fn=fake_rerank)

    assert calls["search"] == ["kw-a", "kw-b"]          # 每个关键词都搜了
    # semantic_top_k=2,取重排后前二:gamma(0.9), alpha(0.8)
    assert [r["full_name"] for r in results] == ["o/gamma", "o/alpha"]
    assert results[0]["score"] == 0.9
    # 去重生效:beta 只出现一次(被排在第三,未进 top2)
    assert all(r["full_name"] != "o/beta" for r in results)


def test_search_truncates_long_documents_before_rerank():
    captured = {}

    def fake_rerank(query, docs, base, model, api_key=None):
        captured["docs"] = docs
        return [(0, 0.9)]

    huge = _repo("o/huge", desc="x" * 50000)  # 异常长描述
    results = realtime_search.search(
        "q", _cfg(), extract=lambda *a, **k: ["kw"],
        searcher=lambda *a, **k: [huge], rerank_fn=fake_rerank)

    assert len(captured["docs"][0]) <= realtime_search.MAX_DOC_CHARS
    assert results[0]["full_name"] == "o/huge"


def test_search_returns_empty_when_no_candidates():
    results = realtime_search.search(
        "q", _cfg(), extract=lambda *a, **k: ["kw"],
        searcher=lambda *a, **k: [], rerank_fn=lambda *a, **k: [])
    assert results == []


def test_search_falls_back_unranked_when_reranker_down():
    # reranker 服务挂掉时不应 500,回退按候选顺序返回 top_k(无 score)
    def boom(*a, **k):
        raise RuntimeError("connection refused")

    repos = [_repo(f"o/r{i}") for i in range(5)]
    results = realtime_search.search(
        "q", _cfg(), extract=lambda *a, **k: ["kw"],
        searcher=lambda *a, **k: repos, rerank_fn=boom)

    assert [r["full_name"] for r in results] == ["o/r0", "o/r1"]   # semantic_top_k=2
    assert all("score" not in r for r in results)

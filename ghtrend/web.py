import os
from datetime import datetime, timezone

from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, JSONResponse

from . import store, github_enricher
from .config import Config
from .embedder import build_text

_TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "index.html")


def _fav_dir(config: Config) -> str:
    return os.path.join(config.data_dir, "favorites")


def create_app(config: Config, embedder, github_token,
               realtime_fn=None, summarize_fn=None) -> FastAPI:
    app = FastAPI(title="GitHub Trending Search")

    def _summarize(results):
        return summarize_fn(results) if summarize_fn else results

    @app.get("/", response_class=HTMLResponse)
    def index():
        with open(_TEMPLATE, encoding="utf-8") as f:
            return f.read()

    @app.get("/api/semantic")
    def semantic(q: str):
        # 同时检索 trending 库 + 收藏库(收藏优先去重)
        fav_p, fav_e = store.load(_fav_dir(config))
        for p in fav_p:
            p["favorited"] = True
        tr_p, tr_e = store.load(config.data_dir)
        projects, embeddings = store.merge([(fav_p, fav_e), (tr_p, tr_e)])
        if not projects:
            return JSONResponse({"results": []})
        qvec = embedder.encode([q])[0]
        ranked = store.cosine_topk(qvec, embeddings, config.semantic_top_k)
        results = []
        for i, score in ranked:
            p = dict(projects[i])
            p["score"] = score
            results.append(p)
        return JSONResponse({"results": results})

    @app.post("/api/favorite")
    def favorite(project: dict = Body(...)):
        fn = project.get("full_name")
        if not fn:
            return JSONResponse({"ok": False, "error": "缺少 full_name"}, status_code=400)
        today = datetime.now(timezone.utc).date().isoformat()
        vec = embedder.encode([build_text(project)])
        fav_dir = _fav_dir(config)
        projects, embeddings = store.load(fav_dir)
        projects, embeddings = store.upsert(projects, embeddings, [project], vec, today)
        store.save(fav_dir, projects, embeddings)
        return JSONResponse({"ok": True, "favorited": fn, "count": len(projects)})

    @app.get("/api/keyword")
    def keyword(q: str):
        results = github_enricher.search_repos(
            q, token=github_token, limit=config.semantic_top_k)
        return JSONResponse({"results": _summarize(results)})

    @app.get("/api/realtime")
    def realtime(q: str):
        if realtime_fn is None:
            return JSONResponse(
                {"results": [], "error": "实时语义搜索未启用(需配置 llm_api_base / reranker_api_base)"})
        return JSONResponse({"results": _summarize(realtime_fn(q))})

    return app


def main() -> None:
    import uvicorn
    from .config import (load_config, get_github_token, get_embedding_api_key,
                         get_llm_api_key, get_reranker_api_key, get_summary_api_key)
    from .embedder import Embedder, RemoteEmbedder
    from . import realtime_search, summarizer
    cfg = load_config()
    github_token = get_github_token()
    llm_key = get_llm_api_key()
    if cfg.embedding_api_base:
        embedder = RemoteEmbedder(cfg.embedding_api_base, cfg.embedding_api_model,
                                  api_key=get_embedding_api_key())
        print(f"语义搜索使用在线 embedding 服务: {cfg.embedding_api_base} ({cfg.embedding_api_model})")
    else:
        embedder = Embedder(cfg.embedding_model)
        print(f"语义搜索使用本地模型: {cfg.embedding_model}")

    realtime_fn = None
    if cfg.llm_api_base and cfg.reranker_api_base:
        rr_key = get_reranker_api_key()
        realtime_fn = lambda q: realtime_search.search(
            q, cfg, github_token=github_token, llm_api_key=llm_key, reranker_api_key=rr_key)
        print(f"实时语义搜索已启用: 关键词={cfg.llm_model} 重排={cfg.reranker_model}")
    else:
        print("实时语义搜索未启用(未配置 llm_api_base / reranker_api_base)")

    # 摘要可用独立 LLM(summary_api_base),否则沿用关键词那套(llm_api_base)
    summary_base = cfg.summary_api_base or cfg.llm_api_base
    summary_model = cfg.summary_model or cfg.llm_model
    summary_key = get_summary_api_key() if cfg.summary_api_base else llm_key
    summarize_fn = None
    if summary_base:
        summarize_fn = lambda items: summarizer.add_summaries(
            items, summary_base, summary_model, api_key=summary_key)
        print(f"GitHub 搜索结果中文摘要已启用: {summary_model} @ {summary_base}")

    app = create_app(cfg, embedder, github_token,
                     realtime_fn=realtime_fn, summarize_fn=summarize_fn)
    print(f"启动 web 服务: http://127.0.0.1:{cfg.web_port}")
    uvicorn.run(app, host="127.0.0.1", port=cfg.web_port)


if __name__ == "__main__":
    main()

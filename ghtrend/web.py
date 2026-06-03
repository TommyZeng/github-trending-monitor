import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from . import store, github_enricher
from .config import Config
from .embedder import build_text  # noqa: F401  (保持依赖显式)

_TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "index.html")


def create_app(config: Config, embedder, github_token, realtime_fn=None) -> FastAPI:
    app = FastAPI(title="GitHub Trending Search")

    @app.get("/", response_class=HTMLResponse)
    def index():
        with open(_TEMPLATE, encoding="utf-8") as f:
            return f.read()

    @app.get("/api/semantic")
    def semantic(q: str):
        projects, embeddings = store.load(config.data_dir)
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

    @app.get("/api/keyword")
    def keyword(q: str):
        results = github_enricher.search_repos(
            q, token=github_token, limit=config.semantic_top_k)
        return JSONResponse({"results": results})

    @app.get("/api/realtime")
    def realtime(q: str):
        if realtime_fn is None:
            return JSONResponse(
                {"results": [], "error": "实时语义搜索未启用(需配置 llm_api_base / reranker_api_base)"})
        return JSONResponse({"results": realtime_fn(q)})

    return app


def main() -> None:
    import uvicorn
    from .config import (load_config, get_github_token, get_embedding_api_key,
                         get_llm_api_key, get_reranker_api_key)
    from .embedder import Embedder, RemoteEmbedder
    from . import realtime_search
    cfg = load_config()
    github_token = get_github_token()
    if cfg.embedding_api_base:
        embedder = RemoteEmbedder(cfg.embedding_api_base, cfg.embedding_api_model,
                                  api_key=get_embedding_api_key())
        print(f"语义搜索使用在线 embedding 服务: {cfg.embedding_api_base} ({cfg.embedding_api_model})")
    else:
        embedder = Embedder(cfg.embedding_model)
        print(f"语义搜索使用本地模型: {cfg.embedding_model}")

    realtime_fn = None
    if cfg.llm_api_base and cfg.reranker_api_base:
        llm_key, rr_key = get_llm_api_key(), get_reranker_api_key()
        realtime_fn = lambda q: realtime_search.search(
            q, cfg, github_token=github_token, llm_api_key=llm_key, reranker_api_key=rr_key)
        print(f"实时语义搜索已启用: 关键词={cfg.llm_model} 重排={cfg.reranker_model}")
    else:
        print("实时语义搜索未启用(未配置 llm_api_base / reranker_api_base)")

    app = create_app(cfg, embedder, github_token, realtime_fn=realtime_fn)
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()

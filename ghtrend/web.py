import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from . import store, github_enricher
from .config import Config
from .embedder import build_text  # noqa: F401  (保持依赖显式)

_TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "index.html")


def create_app(config: Config, embedder, github_token) -> FastAPI:
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

    return app


def main() -> None:
    import uvicorn
    from .config import load_config, get_github_token
    from .embedder import Embedder
    cfg = load_config()
    app = create_app(cfg, Embedder(cfg.embedding_model), get_github_token())
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()

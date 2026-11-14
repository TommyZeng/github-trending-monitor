import os
import re
from datetime import datetime, timezone

import requests

from fastapi import FastAPI, Body, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from . import store, github_enricher, trending_fetcher
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

    @app.get("/api/local_keyword")
    def local_keyword(q: str):
        # 本地库(trending + 收藏)关键词检索:多词 AND,命中任意字段即可,按 star 降序
        terms = [t.lower() for t in q.split() if t.strip()]
        if not terms:
            return JSONResponse({"results": []})
        fav_p, fav_e = store.load(_fav_dir(config))
        for p in fav_p:
            p["favorited"] = True
        tr_p, tr_e = store.load(config.data_dir)
        projects, _ = store.merge([(fav_p, fav_e), (tr_p, tr_e)])
        def hay(p):
            parts = [p.get("full_name"), p.get("description"), p.get("description_zh"),
                     p.get("language"), " ".join(p.get("topics") or [])]
            return " ".join(s for s in parts if s).lower()
        hits = [p for p in projects if all(t in hay(p) for t in terms)]
        for p in hits:
            p.setdefault("favorited", False)
        hits.sort(key=lambda p: p.get("stars", 0) or 0, reverse=True)
        return JSONResponse({"results": hits[:50]})

    _SINCE = {"daily", "weekly", "monthly"}

    @app.get("/api/leaderboard")
    def leaderboard(since: str = "daily", language: str = ""):
        if since not in _SINCE:
            since = "daily"
        try:
            items = trending_fetcher.fetch_trending(since=since, language=language or None)
        except Exception:
            return JSONResponse({"results": [], "since": since, "error": "拉取 GitHub trending 失败,请重试"})
        items.sort(key=lambda p: p.get("stars_today") or 0, reverse=True)
        return JSONResponse({"results": _summarize(items), "since": since})

    @app.get("/api/library")
    def library():
        # 本地语义库 = 每日 trending 采集积累的全部项目
        projects, _ = store.load(config.data_dir)
        fav_names = {p["full_name"] for p in store.load(_fav_dir(config))[0]}
        for p in projects:
            p["favorited"] = p["full_name"] in fav_names
        # 按收录时间(first_seen)倒序,其次 star 倒序
        projects.sort(key=lambda p: (p.get("first_seen", ""), p.get("stars", 0)), reverse=True)
        return JSONResponse({"results": projects})

    @app.get("/api/favorites")
    def favorites():
        projects, _ = store.load(_fav_dir(config))
        for p in projects:
            p["favorited"] = True
        # 按收藏时间(first_seen)倒序,其次 star 倒序
        projects.sort(key=lambda p: (p.get("first_seen", ""), p.get("stars", 0)), reverse=True)
        return JSONResponse({"results": projects})

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

    # ---------- 问答:转发到 Vibe-Trading API(它只监听本机,不直接对外) ----------
    _SID = re.compile(r"[A-Za-z0-9_-]{1,64}")

    def _qa_call(method, path, **kw):
        """调上游;未启用返回 503、上游不可用返回 502(以 JSONResponse 形式返回)。"""
        if not config.qa_api_base:
            return None, JSONResponse({"error": "问答功能未启用(未配置 qa_api_base)"}, status_code=503)
        try:
            resp = method(f"{config.qa_api_base.rstrip('/')}{path}", **kw)
            resp.raise_for_status()
            return resp, None
        except Exception:
            return None, JSONResponse({"error": "问答服务暂不可用,请确认 vibe-trading-api 服务在运行"},
                                      status_code=502)

    def _bad_sid(sid):
        return None if _SID.fullmatch(sid) else JSONResponse({"error": "非法会话 ID"}, status_code=400)

    @app.post("/api/qa/sessions")
    def qa_create():
        resp, err = _qa_call(requests.post, "/sessions", json={"title": "网页问答"}, timeout=30)
        return err or JSONResponse({"session_id": resp.json()["session_id"]})

    @app.post("/api/qa/sessions/{sid}/messages")
    def qa_send(sid: str, body: dict = Body(...)):
        if bad := _bad_sid(sid):
            return bad
        content = (body.get("content") or "").strip()
        if not content:
            return JSONResponse({"error": "问题不能为空"}, status_code=400)
        _, err = _qa_call(requests.post, f"/sessions/{sid}/messages",
                          json={"content": content}, timeout=30)
        return err or JSONResponse({"ok": True})

    @app.post("/api/qa/sessions/{sid}/cancel")
    def qa_cancel(sid: str):
        if bad := _bad_sid(sid):
            return bad
        _, err = _qa_call(requests.post, f"/sessions/{sid}/cancel", timeout=30)
        return err or JSONResponse({"ok": True})

    @app.get("/api/qa/sessions/{sid}/events")
    def qa_events(sid: str, request: Request):
        if bad := _bad_sid(sid):
            return bad
        # 会话级长连接:一轮回答结束后不断开,追问复用;读超时设为不限。
        # 透传 Last-Event-ID:浏览器断线重连时上游据此续传,否则会重放整轮事件
        last_id = request.headers.get("last-event-id")
        upstream, err = _qa_call(requests.get, f"/sessions/{sid}/events", stream=True,
                                 timeout=(10, None),
                                 headers={"Last-Event-ID": last_id} if last_id else {})
        if err:
            return err

        def relay():
            try:
                yield from upstream.iter_content(chunk_size=None)
            finally:
                upstream.close()
        return StreamingResponse(relay(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    # 模型画图有时输出图片,有时输出内嵌 SVG 的 HTML,两种都放行
    _FIG_TYPES = {".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg",
                  ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif",
                  ".html": "text/html", ".htm": "text/html"}

    @app.get("/api/qa/files")
    def qa_file(path: str):
        # 只放行 runs 目录内的图表文件;按真实路径校验,挡住 ../ 和软链接越界
        root = os.path.realpath(os.path.expanduser(config.qa_runs_dir))
        real = os.path.realpath(os.path.expanduser(path))
        ext = os.path.splitext(real)[1].lower()
        if not real.startswith(root + os.sep) or ext not in _FIG_TYPES:
            return JSONResponse({"error": "不允许访问该文件"}, status_code=403)
        if not os.path.isfile(real):
            return JSONResponse({"error": "文件不存在"}, status_code=404)
        # 内容由模型生成,CSP 兜底(直接在新标签页打开时同样生效):default-src 'none' 禁止一切
        # 网络请求与外部资源。HTML 图表常用脚本绘图,允许内联脚本,但 sandbox 不给 allow-same-origin,
        # 脚本处于无主来源,碰不到本站 DOM/cookie;图片类一律不许脚本
        csp = "default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:; "
        csp += ("script-src 'unsafe-inline'; sandbox allow-scripts" if ext in (".html", ".htm") else "sandbox")
        return FileResponse(real, media_type=_FIG_TYPES[ext], headers={
            "Content-Security-Policy": csp, "X-Content-Type-Options": "nosniff"})

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
    print(f"启动 web 服务: http://{cfg.web_host}:{cfg.web_port}")
    # 问答页的事件流是长连接,默认会一直等它关闭才退出(systemd 重启要卡 90 秒超时);最多等 3 秒
    uvicorn.run(app, host=cfg.web_host, port=cfg.web_port, timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()

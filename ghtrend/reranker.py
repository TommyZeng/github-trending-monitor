from urllib.parse import urlsplit

_DASHSCOPE_RERANK_PATH = "/api/v1/services/rerank/text-rerank/text-rerank"


def _is_dashscope(api_base: str) -> bool:
    return "aliyuncs.com" in urlsplit(api_base).netloc


def _dashscope_url(api_base: str) -> str:
    parts = urlsplit(api_base)
    return f"{parts.scheme}://{parts.netloc}{_DASHSCOPE_RERANK_PATH}"


def rerank(query: str, documents: list[str], api_base: str, model: str,
           api_key=None, session=None) -> list[tuple[int, float]]:
    """调 OpenAI/Jina 风格的 rerank 服务(如 vLLM bge-reranker)。
    返回 [(原始文档下标, 相关性分数)],按分数降序。documents 为空直接返回 []。"""
    if not documents:
        return []
    sess = session
    if sess is None:
        import requests
        sess = requests
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if _is_dashscope(api_base):
        # 阿里云百炼/DashScope 的 rerank 不在 OpenAI 兼容路径下,
        # 请求体(query/documents 嵌在 input)与响应(results 嵌在 output)也自成一套
        url = _dashscope_url(api_base)
        payload = {"model": model,
                   "input": {"query": query, "documents": list(documents)},
                   "parameters": {"return_documents": False}}
    else:
        url = f"{api_base.rstrip('/')}/rerank"
        payload = {"model": model, "query": query, "documents": list(documents)}
    resp = sess.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    results = body.get("output", {}).get("results") or body.get("results") or []
    ranked = [(r["index"], float(r["relevance_score"])) for r in results]
    ranked.sort(key=lambda t: t[1], reverse=True)
    return ranked

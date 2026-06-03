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
    resp = sess.post(
        f"{api_base.rstrip('/')}/rerank",
        json={"model": model, "query": query, "documents": list(documents)},
        headers=headers, timeout=60)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    ranked = [(r["index"], float(r["relevance_score"])) for r in results]
    ranked.sort(key=lambda t: t[1], reverse=True)
    return ranked

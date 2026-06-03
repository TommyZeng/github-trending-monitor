from .config import Config
from .embedder import build_text
from .keyword_extractor import extract_keywords
from .github_enricher import search_repos
from .reranker import rerank

# 单个文档送入 reranker 前的最大字符数(reranker 上下文上限 8192 token,
# 描述类文本远用不到;截断可防个别仓库超长描述触发 400)。
MAX_DOC_CHARS = 2000


def search(query: str, config: Config, github_token=None,
           llm_api_key=None, reranker_api_key=None,
           extract=extract_keywords, searcher=search_repos, rerank_fn=rerank) -> list[dict]:
    """实时 GitHub 语义搜索:
    ① LLM 把 query 拆成多组关键词 → ② 每组调 GitHub 搜索取候选并按 full_name 去重
    → ③ reranker 对候选重排 → ④ 返回 Top semantic_top_k(附 score)。"""
    keywords = extract(query, config.llm_api_base, config.llm_model, api_key=llm_api_key)

    seen: dict[str, dict] = {}
    for kw in keywords:
        for repo in searcher(kw, token=github_token, limit=config.realtime_per_keyword):
            fn = repo.get("full_name")
            if fn and fn not in seen:
                seen[fn] = repo
    candidates = list(seen.values())
    if not candidates:
        return []

    docs = [build_text(c)[:MAX_DOC_CHARS] for c in candidates]
    ranked = rerank_fn(query, docs, config.reranker_api_base, config.reranker_model,
                       api_key=reranker_api_key)

    results = []
    for idx, score in ranked[:config.semantic_top_k]:
        item = dict(candidates[idx])
        item["score"] = score
        results.append(item)
    return results

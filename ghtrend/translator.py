import re

# 单条描述送入 LLM 前的最大字符数(推送描述用不到更多;
# 防个别仓库的超长描述撑爆 LLM 上下文导致整批翻译失败)
MAX_TEXT_CHARS = 400

# 翻译失败重试次数(Google 兜底在 Actions 环境已不可用,只能靠 LLM 自身重试)
MAX_ATTEMPTS = 3

_LLM_PROMPT = (
    "把下面这些 GitHub 项目的英文描述翻译成简体中文:通顺自然,"
    "保留技术专有名词英文原样(如 CLI、RAG、LLM、API 及框架/产品名),去掉无关符号。"
    "严格按输入顺序输出一个 JSON 字符串数组,长度与条数相同,不要任何解释或代码块标记。\n\n{block}"
)


def _translate_once(texts, idx, api_base, model, api_key, sess) -> list[str] | None:
    """翻译 idx 指定的那些文本;失败(网络/解析/条数不齐)返回 None。"""
    from .summarizer import _parse_array, _clean

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    block = "\n".join(f"{n}. {str(texts[i])[:MAX_TEXT_CHARS]}"
                      for n, i in enumerate(idx, 1))
    parsed = None
    for _ in range(MAX_ATTEMPTS):     # 偶发网络/限流重试
        try:
            resp = sess.post(
                f"{api_base.rstrip('/')}/chat/completions",
                json={"model": model, "temperature": 0,
                      "messages": [{"role": "user",
                                    "content": _LLM_PROMPT.format(block=block)}]},
                headers=headers, timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception:
            continue
        parsed = _parse_array(content)
        if parsed is not None and len(parsed) == len(idx):
            break
        parsed = None
    if parsed is None:
        return None
    return [_clean(v) or str(texts[i]) for i, v in zip(idx, parsed)]


def llm_translate_batch(texts, api_base: str, model: str,
                        api_key=None, session=None) -> list[str] | None:
    """用 LLM 批量翻译成中文,返回与输入等长的译文列表(空文本占位 "")。
    整批失败时二分拆批重试——个别描述会被服务端拒绝并连累整批,拆开后
    只有出问题的那条保留原文。全部失败返回 None。"""
    texts = list(texts or [])
    if not texts:
        return []
    idx = [i for i, t in enumerate(texts) if t]
    if not idx:
        return ["" for _ in texts]
    sess = session
    if sess is None:
        import requests
        sess = requests

    out = ["" for _ in texts]
    ok = False

    def fill(sub_idx):
        nonlocal ok
        if not sub_idx:
            return
        got = _translate_once(texts, sub_idx, api_base, model, api_key, sess)
        if got is not None:
            for i, v in zip(sub_idx, got):
                out[i] = v
            ok = True
        elif len(sub_idx) > 1:
            mid = len(sub_idx) // 2
            fill(sub_idx[:mid])
            fill(sub_idx[mid:])
        else:
            out[sub_idx[0]] = str(texts[sub_idx[0]])   # 单条仍失败:保留原文

    fill(idx)
    return out if ok else None

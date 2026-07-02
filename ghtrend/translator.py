import re

# 单条描述送入 LLM 前的最大字符数(推送描述用不到更多;
# 防个别仓库的超长描述撑爆 LLM 上下文导致整批翻译失败)
MAX_TEXT_CHARS = 400

_LLM_PROMPT = (
    "把下面这些 GitHub 项目的英文描述翻译成简体中文:通顺自然,"
    "保留技术专有名词英文原样(如 CLI、RAG、LLM、API 及框架/产品名),去掉无关符号。"
    "严格按输入顺序输出一个 JSON 字符串数组,长度与条数相同,不要任何解释或代码块标记。\n\n{block}"
)


def llm_translate_batch(texts, api_base: str, model: str,
                        api_key=None, session=None) -> list[str] | None:
    """用 LLM 一次调用批量翻译成中文。成功返回与输入等长的译文列表(空文本占位 "");
    任何失败(网络/解析/条数不齐)返回 None,由调用方回退其他翻译方式。"""
    from .summarizer import _parse_array, _clean

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
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    block = "\n".join(f"{n}. {str(texts[i])[:MAX_TEXT_CHARS]}"
                      for n, i in enumerate(idx, 1))
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
        return None
    parsed = _parse_array(content)
    if parsed is None or len(parsed) != len(idx):
        return None
    out = ["" for _ in texts]
    for i, s in zip(idx, parsed):
        out[i] = _clean(s) or str(texts[i])
    return out


def _default_translator():
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source="auto", target="zh-CN")


def translate_to_zh(text, translator=None) -> str:
    """把文本翻成简体中文。空文本返回 "";翻译出错时回退返回原文,绝不抛异常。"""
    if not text:
        return ""
    try:
        t = translator if translator is not None else _default_translator()
        return t.translate(text)
    except Exception:
        return text

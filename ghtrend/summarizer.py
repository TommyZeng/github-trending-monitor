import json
import re

# 每条描述送入摘要 prompt 前的最大字符数(生成一句话摘要用不到更多;
# 防个别仓库的超长描述/满屏全角空格撑爆 LLM 上下文导致整批摘要失败)。
MAX_DESC_CHARS = 300

_PROMPT = (
    "下面是若干 GitHub 项目(描述里可能混有 emoji、徽章、中英文)。"
    "请为每个项目生成一句简洁通顺的中文摘要,说明它是做什么的,不超过 40 字,去掉无关符号。"
    "严格按输入顺序输出一个 JSON 字符串数组,长度与项目数相同,不要任何解释或代码块标记。\n\n{block}"
)


def _build_block(items: list[dict]) -> str:
    lines = []
    for i, p in enumerate(items, 1):
        topics = ", ".join(p.get("topics") or [])
        desc = (p.get("description") or "")[:MAX_DESC_CHARS]
        lines.append(f"{i}. {p.get('full_name', '')} | {desc} | topics: {topics}")
    return "\n".join(lines)


def _clean(s: str) -> str:
    # 小模型有时把输入序号写进摘要(如 "1. xxx"),去掉开头的列表编号
    return re.sub(r"^\s*\d+\s*[.、)]\s*", "", s).strip()


def _parse_array(content: str) -> list[str] | None:
    text = content.strip()
    # 推理模型(MiniMax-M3、qwen 系等)会先输出 <think>...</think>,先剥掉再找 JSON
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        data = json.loads(text)
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    return [str(x).strip() for x in data]


def add_summaries(items: list[dict], api_base: str, model: str,
                  api_key=None, session=None) -> list[dict]:
    """给每个项目加 summary_zh(LLM 批量生成的一句话中文摘要)。
    一次 LLM 调用;任何失败都原样返回(不加 summary_zh),不影响搜索结果。"""
    if not items:
        return items
    sess = session
    if sess is None:
        import requests
        sess = requests
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = sess.post(
            f"{api_base.rstrip('/')}/chat/completions",
            json={"model": model, "temperature": 0,
                  "messages": [{"role": "user",
                                "content": _PROMPT.format(block=_build_block(items))}]},
            headers=headers, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return items
    summaries = _parse_array(content)
    if summaries:
        for item, s in zip(items, summaries):
            s = _clean(s)
            if s:
                item["summary_zh"] = s
    return items

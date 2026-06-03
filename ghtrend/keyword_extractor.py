import json
import re

_PROMPT = (
    "你是 GitHub 搜索助手。根据用户的自然语言需求,提取 {n} 组用于在 GitHub 上搜索仓库的"
    "英文关键词。只输出一个 JSON 数组,每个元素是一组关键词字符串,不要任何解释或代码块标记。\n"
    "用户需求:{query}"
)


def _parse_keywords(content: str) -> list[str] | None:
    text = content.strip()
    # 去掉可能的 ```json ... ``` 代码块包裹
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    # 容错:截取第一个 [ 到最后一个 ]
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        data = json.loads(text)
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    kws = [str(x).strip() for x in data if str(x).strip()]
    return kws or None


def extract_keywords(query: str, api_base: str, model: str,
                     api_key=None, n: int = 5, session=None) -> list[str]:
    """调 OpenAI 兼容 LLM 把自然语言拆成多组英文搜索关键词。
    任何失败(网络/解析)都回退为 [query],保证后续搜索不中断。"""
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
                                "content": _PROMPT.format(n=n, query=query)}]},
            headers=headers, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return [query]
    return _parse_keywords(content) or [query]

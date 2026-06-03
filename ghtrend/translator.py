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

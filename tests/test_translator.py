from ghtrend import translator


class _FakeTranslator:
    def __init__(self, out): self._out = out
    def translate(self, text): return self._out


class _RaisingTranslator:
    def translate(self, text): raise RuntimeError("rate limited")


def test_empty_returns_empty():
    assert translator.translate_to_zh("") == ""
    assert translator.translate_to_zh(None) == ""


def test_uses_injected_translator():
    out = translator.translate_to_zh("A fast tool", translator=_FakeTranslator("一个快速工具"))
    assert out == "一个快速工具"


def test_falls_back_to_original_on_error():
    # 翻译服务报错时不应崩溃,回退原文
    out = translator.translate_to_zh("A fast tool", translator=_RaisingTranslator())
    assert out == "A fast tool"

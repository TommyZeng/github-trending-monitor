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


# ---------- LLM 批量翻译 ----------

class _FakeSession:
    """伪造 OpenAI 兼容 chat/completions 响应。"""
    def __init__(self, content):
        self._content = content
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        content = self._content
        class _Resp:
            def raise_for_status(self): pass
            def json(self):
                return {"choices": [{"message": {"content": content}}]}
        return _Resp()


class _ErrorSession:
    def post(self, *a, **k): raise RuntimeError("connection refused")


def test_llm_batch_translates_and_keeps_order():
    sess = _FakeSession('["快速的CLI工具", "自托管的RAG框架"]')
    out = translator.llm_translate_batch(
        ["A fast CLI tool", "Self-hosted RAG framework"],
        api_base="https://api.example.com/v1", model="m", api_key="k", session=sess)
    assert out == ["快速的CLI工具", "自托管的RAG框架"]
    call = sess.calls[0]
    assert call["url"] == "https://api.example.com/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer k"
    assert call["json"]["model"] == "m"


def test_llm_batch_empty_list_returns_empty_without_calling():
    sess = _FakeSession('[]')
    assert translator.llm_translate_batch([], api_base="b", model="m", session=sess) == []
    assert sess.calls == []


def test_llm_batch_preserves_empty_texts():
    # 空描述不送 LLM,占位 "" 原样保留;只有非空文本参与翻译
    sess = _FakeSession('["工具"]')
    out = translator.llm_translate_batch(
        ["", "Tool", None], api_base="b", model="m", session=sess)
    assert out == ["", "工具", ""]
    # prompt 里只包含 1 条待翻译文本
    assert "Tool" in sess.calls[0]["json"]["messages"][0]["content"]


def test_llm_batch_returns_none_on_error():
    out = translator.llm_translate_batch(
        ["A tool"], api_base="b", model="m", session=_ErrorSession())
    assert out is None


def test_llm_batch_returns_none_on_length_mismatch():
    sess = _FakeSession('["只有一条"]')
    out = translator.llm_translate_batch(
        ["one", "two"], api_base="b", model="m", session=sess)
    assert out is None


def test_llm_batch_no_auth_header_without_key():
    sess = _FakeSession('["工具"]')
    translator.llm_translate_batch(["Tool"], api_base="b", model="m", session=sess)
    assert "Authorization" not in sess.calls[0]["headers"]


def test_llm_batch_truncates_overlong_text():
    # 个别仓库描述有几万字符,送入前必须截断,防止撑爆 LLM 上下文
    sess = _FakeSession('["翻译"]')
    translator.llm_translate_batch(["x" * 50000], api_base="b", model="m", session=sess)
    prompt = sess.calls[0]["json"]["messages"][0]["content"]
    assert len(prompt) < 2000

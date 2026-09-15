from ghtrend import translator


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


def test_llm_batch_bisects_on_length_mismatch():
    # 整批条数不齐会触发二分拆批,拆成单条后各自对齐即成功
    sess = _FakeSession('["只有一条"]')
    out = translator.llm_translate_batch(
        ["one", "two"], api_base="b", model="m", session=sess)
    assert out == ["只有一条", "只有一条"]


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


def test_llm_batch_strips_reasoning_think_block():
    # 推理模型(MiniMax-M3 等)会先输出 <think>...</think>,里面可能夹带 [ ] 干扰解析
    sess = _FakeSession(
        '<think>Let me check requirements [1] and [2]... '
        'the answer should be ["x"]</think>\n\n["轻量级工具"]')
    out = translator.llm_translate_batch(["A tool"], api_base="b", model="m", session=sess)
    assert out == ["轻量级工具"]


# ---------- LLM 重试(替代已失效的 Google 兜底) ----------

class _FlakySession:
    """前 n 次失败,之后成功。"""
    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls += 1
        failing = self.calls <= self.fail_times
        class _Resp:
            def raise_for_status(self):
                if failing:
                    raise RuntimeError("503 temporarily unavailable")
            def json(self):
                return {"choices": [{"message": {"content": '["工具"]'}}]}
        return _Resp()


def test_llm_batch_retries_transient_failure():
    sess = _FlakySession(fail_times=2)
    out = translator.llm_translate_batch(["Tool"], api_base="b", model="m", session=sess)
    assert out == ["工具"]
    assert sess.calls == 3          # 失败 2 次后第 3 次成功


def test_llm_batch_gives_up_after_max_attempts():
    sess = _FlakySession(fail_times=99)
    assert translator.llm_translate_batch(["Tool"], api_base="b", model="m", session=sess) is None
    assert sess.calls == 3          # 最多尝试 3 次


class _PoisonSession:
    """含 badtext 的批次整批失败(模拟个别描述触发服务端拒绝)。"""
    def __init__(self): self.calls = 0

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls += 1
        content_in = json["messages"][0]["content"]
        bad = "badtext" in content_in
        import re as _re
        rows = _re.findall(r"^\d+\. ", content_in, flags=_re.M)
        class _Resp:
            def raise_for_status(self):
                if bad:
                    raise RuntimeError("400 rejected")
            def json(self):
                import json as _json
                return {"choices": [{"message": {"content":
                    _json.dumps([f"译{i}" for i in range(len(rows))], ensure_ascii=False)}}]}
        return _Resp()


def test_llm_batch_bisects_around_poison_item():
    # 4 条里 1 条会让整批失败:其余 3 条仍应拿到译文,坏的那条保留原文
    texts = ["t0", "t1", "badtext", "t3"]
    out = translator.llm_translate_batch(texts, api_base="b", model="m",
                                         session=_PoisonSession())
    assert out is not None
    assert out[0].startswith("译") and out[1].startswith("译") and out[3].startswith("译")
    assert out[2] == "badtext"      # 无法翻译的保留原文

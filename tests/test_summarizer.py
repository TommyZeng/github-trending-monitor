from ghtrend import summarizer


class _Resp:
    def __init__(self, content): self._c = content
    def raise_for_status(self): pass
    def json(self): return {"choices": [{"message": {"content": self._c}}]}


class _Session:
    def __init__(self, content): self._c = content; self.calls = []
    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return _Resp(self._c)


def _items():
    return [
        {"full_name": "a/x", "description": "🚀 A fast tool", "topics": ["cli"]},
        {"full_name": "b/y", "description": "blah", "topics": []},
    ]


def test_add_summaries_sets_summary_zh_in_order():
    sess = _Session('["快速命令行工具", "另一个项目"]')
    out = summarizer.add_summaries(_items(), "http://llm/v1", "deepseek",
                                   api_key="k", session=sess)
    assert out[0]["summary_zh"] == "快速命令行工具"
    assert out[1]["summary_zh"] == "另一个项目"
    call = sess.calls[0]
    assert call["url"] == "http://llm/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer k"


def test_add_summaries_strips_code_fence():
    sess = _Session('```json\n["摘要1", "摘要2"]\n```')
    out = summarizer.add_summaries(_items(), "http://llm/v1", "m", session=sess)
    assert out[0]["summary_zh"] == "摘要1"


def test_add_summaries_returns_items_unchanged_on_error():
    class _Boom:
        def post(self, *a, **k): raise RuntimeError("down")
    out = summarizer.add_summaries(_items(), "http://llm/v1", "m", session=_Boom())
    assert "summary_zh" not in out[0]
    assert out[0]["full_name"] == "a/x"


def test_add_summaries_empty_list():
    assert summarizer.add_summaries([], "http://llm/v1", "m", session=_Session("[]")) == []


def test_build_block_truncates_long_description():
    items = [{"full_name": "a/x", "description": "y" * 50000, "topics": []}]
    block = summarizer._build_block(items)
    # 单条描述不应把整段 5 万字符塞进 prompt
    assert len(block) < summarizer.MAX_DESC_CHARS + 200
    assert "y" * (summarizer.MAX_DESC_CHARS + 1) not in block

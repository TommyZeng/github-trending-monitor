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


def test_add_summaries_strips_leading_list_number():
    # 小模型可能把序号写进摘要
    sess = _Session('["1. 快速命令行工具", "2、另一个项目"]')
    out = summarizer.add_summaries(_items(), "http://llm/v1", "m", session=sess)
    assert out[0]["summary_zh"] == "快速命令行工具"
    assert out[1]["summary_zh"] == "另一个项目"


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


class _ModerationSession:
    """伪造内容审核:prompt 里含 badrepo 的批次返回 422,其余正常。"""
    def __init__(self):
        self.calls = 0

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls += 1
        content_in = json["messages"][0]["content"]
        bad = "badrepo" in content_in
        n = content_in.count("\n") - content_in.split("\n\n")[0].count("\n")  # 粗略条数
        import re as _re
        rows = _re.findall(r"^\d+\. ", content_in, flags=_re.M)
        class _Resp:
            def raise_for_status(self):
                if bad:
                    raise RuntimeError("422 input new_sensitive")
            def json(self):
                import json as _json
                return {"choices": [{"message": {"content":
                    _json.dumps([f"摘要{i}" for i in range(len(rows))], ensure_ascii=False)}}]}
        return _Resp()


def test_add_summaries_bisects_around_moderated_item():
    # 4 条里 1 条触发审核:其余 3 条仍应拿到摘要,坏的那条没有
    items = [{"full_name": f"o/repo{i}", "description": "fine", "topics": []} for i in range(4)]
    items[2]["description"] = "badrepo content"
    sess = _ModerationSession()
    out = summarizer.add_summaries(items, "http://llm/v1", "m", session=sess)
    assert "summary_zh" in out[0] and "summary_zh" in out[1] and "summary_zh" in out[3]
    assert "summary_zh" not in out[2]

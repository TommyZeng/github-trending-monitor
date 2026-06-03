from ghtrend import keyword_extractor as ke


class _Resp:
    def __init__(self, content): self._c = content
    def raise_for_status(self): pass
    def json(self):
        return {"choices": [{"message": {"content": self._c}}]}


class _Session:
    def __init__(self, content): self._c = content; self.calls = []
    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return _Resp(self._c)


def test_extract_parses_json_array():
    sess = _Session('["rust terminal AI", "rust CLI assistant"]')
    kws = ke.extract_keywords("用Rust写的终端AI", "http://llm/v1", "deepseek",
                              api_key="k", session=sess)
    assert kws == ["rust terminal AI", "rust CLI assistant"]
    call = sess.calls[0]
    assert call["url"] == "http://llm/v1/chat/completions"
    assert call["json"]["model"] == "deepseek"
    assert call["headers"]["Authorization"] == "Bearer k"


def test_extract_strips_code_fence():
    sess = _Session('```json\n["a", "b"]\n```')
    assert ke.extract_keywords("q", "http://llm/v1", "m", session=sess) == ["a", "b"]


def test_extract_falls_back_to_query_on_bad_output():
    sess = _Session("抱歉我无法处理")
    assert ke.extract_keywords("my query", "http://llm/v1", "m", session=sess) == ["my query"]


def test_extract_falls_back_on_error():
    class _Boom:
        def post(self, *a, **k): raise RuntimeError("down")
    assert ke.extract_keywords("my query", "http://llm/v1", "m", session=_Boom()) == ["my query"]

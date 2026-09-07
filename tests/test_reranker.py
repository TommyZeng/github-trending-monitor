from ghtrend import reranker


class _Resp:
    def __init__(self, payload): self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p


class _Session:
    def __init__(self, payload): self._p = payload; self.calls = []
    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return _Resp(self._p)


def test_rerank_returns_index_score_sorted_desc():
    payload = {"results": [
        {"index": 0, "relevance_score": 0.2},
        {"index": 1, "relevance_score": 0.9},
        {"index": 2, "relevance_score": 0.5},
    ]}
    sess = _Session(payload)
    out = reranker.rerank("q", ["d0", "d1", "d2"], "http://rr/v1", "bge-reranker-v2-m3",
                          api_key="k", session=sess)
    assert out == [(1, 0.9), (2, 0.5), (0, 0.2)]
    call = sess.calls[0]
    assert call["url"] == "http://rr/v1/rerank"
    assert call["json"]["query"] == "q"
    assert call["json"]["documents"] == ["d0", "d1", "d2"]
    assert call["json"]["model"] == "bge-reranker-v2-m3"
    assert call["headers"]["Authorization"] == "Bearer k"


def test_rerank_empty_documents_returns_empty():
    out = reranker.rerank("q", [], "http://rr/v1", "m", session=_Session({"results": []}))
    assert out == []


class _DashScopeSession:
    """阿里云 DashScope 原生 rerank:路径与响应格式都与 Jina 风格不同。"""
    def __init__(self): self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json})
        class _Resp:
            def raise_for_status(self): pass
            def json(self):
                return {"output": {"results": [
                    {"index": 1, "relevance_score": 0.70},
                    {"index": 0, "relevance_score": 0.36}]}}
        return _Resp()


def test_rerank_uses_dashscope_native_api_for_aliyun():
    sess = _DashScopeSession()
    out = reranker.rerank(
        "密码管理器", ["文件管理器", "密码管理工具"],
        "https://llm-x.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        "qwen3-rerank", api_key="k", session=sess)
    assert out == [(1, 0.70), (0, 0.36)]
    call = sess.calls[0]
    # 走原生路径(而非 compatible-mode 下的 /rerank)
    assert call["url"] == ("https://llm-x.cn-beijing.maas.aliyuncs.com"
                           "/api/v1/services/rerank/text-rerank/text-rerank")
    # 原生请求体:query/documents 嵌在 input 下
    assert call["json"]["input"]["query"] == "密码管理器"
    assert call["json"]["input"]["documents"] == ["文件管理器", "密码管理工具"]


def test_rerank_keeps_jina_style_for_self_hosted():
    # 自建 vLLM(localhost)仍走原来的 Jina 风格,不受影响
    sess = _Session({"results": [{"index": 0, "relevance_score": 0.9}]})
    reranker.rerank("q", ["a"], "http://localhost:8787/v1", "m", session=sess)
    assert sess.calls[0]["url"] == "http://localhost:8787/v1/rerank"
    assert sess.calls[0]["json"]["query"] == "q"

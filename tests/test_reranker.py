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

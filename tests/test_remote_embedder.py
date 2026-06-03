import numpy as np
from ghtrend.embedder import RemoteEmbedder


class _FakeResp:
    def __init__(self, payload): self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p


class _FakeSession:
    def __init__(self, payload): self._p = payload; self.calls = []
    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return _FakeResp(self._p)


def test_encode_posts_and_returns_normalized_float32():
    # 返回两个未归一化的向量
    payload = {"data": [
        {"index": 0, "embedding": [3.0, 4.0]},
        {"index": 1, "embedding": [0.0, 2.0]},
    ]}
    sess = _FakeSession(payload)
    emb = RemoteEmbedder("http://svc:7000/v1", "bge-m3", api_key="k", session=sess)
    out = emb.encode(["a", "b"])

    assert out.dtype == np.float32
    assert out.shape == (2, 2)
    # 归一化:每行模长为 1
    assert np.allclose(np.linalg.norm(out, axis=1), [1.0, 1.0])
    assert np.allclose(out[0], [0.6, 0.8])

    call = sess.calls[0]
    assert call["url"] == "http://svc:7000/v1/embeddings"
    assert call["json"]["model"] == "bge-m3"
    assert call["json"]["input"] == ["a", "b"]
    assert call["headers"]["Authorization"] == "Bearer k"


def test_encode_orders_by_index():
    # 服务返回乱序,需按 index 重排
    payload = {"data": [
        {"index": 1, "embedding": [0.0, 1.0]},
        {"index": 0, "embedding": [1.0, 0.0]},
    ]}
    emb = RemoteEmbedder("http://svc/v1", "bge-m3", api_key=None, session=_FakeSession(payload))
    out = emb.encode(["first", "second"])
    assert np.allclose(out[0], [1.0, 0.0])
    assert np.allclose(out[1], [0.0, 1.0])

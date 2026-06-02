import numpy as np
from ghtrend import embedder


def test_build_text_combines_fields():
    p = {"full_name": "a/x", "description": "fast tool",
         "topics": ["cli", "rust"], "readme_excerpt": "hello"}
    text = embedder.build_text(p)
    assert "a/x" in text
    assert "fast tool" in text
    assert "cli rust" in text
    assert "hello" in text


def test_build_text_skips_empty():
    p = {"full_name": "a/x", "description": None, "topics": [], "readme_excerpt": ""}
    text = embedder.build_text(p)
    assert text == "a/x"


class _FakeModel:
    def encode(self, texts, normalize_embeddings=True):
        return np.array([[float(len(t)), 1.0] for t in texts])


def test_encode_returns_float32_array_via_injected_model():
    emb = embedder.Embedder("unused", model=_FakeModel())
    out = emb.encode(["ab", "abc"])
    assert out.dtype == np.float32
    assert out.shape == (2, 2)
    assert out[0, 0] == 2.0 and out[1, 0] == 3.0

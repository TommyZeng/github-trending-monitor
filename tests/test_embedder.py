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


def test_build_text_includes_chinese_description():
    # 中文译文必须进索引:中文查询才能同语言匹配,而非依赖跨语言对齐
    p = {"full_name": "a/x", "description": "A password manager",
         "description_zh": "一个密码管理器", "topics": [], "readme_excerpt": ""}
    text = embedder.build_text(p)
    assert "一个密码管理器" in text
    assert "A password manager" in text   # 英文原文仍保留


def test_build_text_without_chinese_still_works():
    p = {"full_name": "a/x", "description": "A tool", "topics": [], "readme_excerpt": ""}
    assert embedder.build_text(p) == "a/x\nA tool"

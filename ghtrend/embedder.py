import numpy as np


def build_text(project: dict) -> str:
    parts = [
        project.get("full_name") or "",
        project.get("description") or "",
        " ".join(project.get("topics") or []),
        project.get("readme_excerpt") or "",
    ]
    return "\n".join(p for p in parts if p)


class Embedder:
    def __init__(self, model_name: str, model=None):
        self.model_name = model_name
        self._model = model

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._get_model().encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


class RemoteEmbedder:
    """调用 OpenAI 兼容的在线 embedding 服务(如自建 vLLM bge-m3)。
    返回 L2 归一化的 float32 向量,与本地 Embedder 输出一致,可与已存向量直接比对。"""

    def __init__(self, api_base: str, model: str, api_key=None, session=None):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._session = session

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests
        return self._session

    def encode(self, texts: list[str]) -> np.ndarray:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = self._get_session().post(
            f"{self.api_base}/embeddings",
            json={"model": self.model, "input": list(texts)},
            headers=headers, timeout=60)
        resp.raise_for_status()
        items = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
        vecs = np.asarray([it["embedding"] for it in items], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).astype(np.float32)

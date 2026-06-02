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

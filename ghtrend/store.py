import json
import os

import numpy as np

PROJECTS_FILE = "projects.jsonl"
EMBEDDINGS_FILE = "embeddings.npy"


def _paths(data_dir: str) -> tuple[str, str]:
    return (os.path.join(data_dir, PROJECTS_FILE),
            os.path.join(data_dir, EMBEDDINGS_FILE))


def load(data_dir: str) -> tuple[list[dict], np.ndarray]:
    pjson, pemb = _paths(data_dir)
    projects: list[dict] = []
    if os.path.exists(pjson):
        with open(pjson, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    projects.append(json.loads(line))
    if os.path.exists(pemb):
        embs = np.load(pemb).astype(np.float32)
    else:
        embs = np.zeros((0, 0), dtype=np.float32)
    return projects, embs


def save(data_dir: str, projects: list[dict], embeddings: np.ndarray) -> None:
    if len(projects) != embeddings.shape[0]:
        raise ValueError(
            f"行不对齐: projects={len(projects)} embeddings={embeddings.shape[0]}")
    os.makedirs(data_dir, exist_ok=True)
    pjson, pemb = _paths(data_dir)
    tmp = pjson + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for p in projects:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    os.replace(tmp, pjson)
    np.save(pemb, embeddings.astype(np.float32))


_UPDATE_FIELDS = ("url", "description", "description_zh", "stars", "language", "topics", "readme_excerpt")


def upsert(projects: list[dict], embeddings: np.ndarray,
           new_items: list[dict], new_embeddings: np.ndarray,
           today: str) -> tuple[list[dict], np.ndarray]:
    projects = [dict(p) for p in projects]
    index = {p["full_name"]: i for i, p in enumerate(projects)}
    rows = [embeddings[i] for i in range(embeddings.shape[0])] if embeddings.shape[0] else []

    for item, vec in zip(new_items, new_embeddings):
        name = item["full_name"]
        if name in index:
            i = index[name]
            for fld in _UPDATE_FIELDS:
                if fld in item:
                    projects[i][fld] = item[fld]
            if today not in projects[i]["trending_history"]:
                projects[i]["trending_history"].append(today)
            rows[i] = np.asarray(vec, dtype=np.float32)
        else:
            rec = dict(item)
            rec["first_seen"] = today
            rec["trending_history"] = [today]
            projects.append(rec)
            index[name] = len(projects) - 1
            rows.append(np.asarray(vec, dtype=np.float32))

    out = np.vstack(rows).astype(np.float32) if rows else np.zeros((0, 0), dtype=np.float32)
    return projects, out


def cosine_topk(query_vec: np.ndarray, embeddings: np.ndarray, k: int) -> list[tuple[int, float]]:
    if embeddings.shape[0] == 0:
        return []
    q = np.asarray(query_vec, dtype=np.float32)
    sims = embeddings @ q
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]

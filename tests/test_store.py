import numpy as np
from ghtrend import store


def _item(full_name, stars=1, desc="d"):
    return {
        "full_name": full_name, "url": f"https://github.com/{full_name}",
        "description": desc, "stars": stars, "language": "Python",
        "topics": ["ai"], "readme_excerpt": "readme",
    }


def test_load_empty_returns_empty(tmp_path):
    projects, embs = store.load(str(tmp_path))
    assert projects == []
    assert embs.shape[0] == 0


def test_upsert_adds_new_then_save_load_roundtrip(tmp_path):
    projects, embs = store.load(str(tmp_path))
    new = [_item("a/x"), _item("b/y")]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    projects, embs = store.upsert(projects, embs, new, vecs, "2026-06-02")
    assert len(projects) == 2
    assert embs.shape == (2, 2)
    assert projects[0]["first_seen"] == "2026-06-02"
    assert projects[0]["trending_history"] == ["2026-06-02"]
    store.save(str(tmp_path), projects, embs)

    p2, e2 = store.load(str(tmp_path))
    assert len(p2) == 2
    assert e2.shape == (2, 2)
    assert {p["full_name"] for p in p2} == {"a/x", "b/y"}


def test_upsert_dedups_and_updates_existing(tmp_path):
    projects, embs = store.load(str(tmp_path))
    projects, embs = store.upsert(
        projects, embs, [_item("a/x", stars=10)],
        np.array([[1.0, 0.0]], dtype=np.float32), "2026-06-01")
    # 第二天同一个项目再次上榜,star 变化、向量更新
    projects, embs = store.upsert(
        projects, embs, [_item("a/x", stars=99)],
        np.array([[0.0, 1.0]], dtype=np.float32), "2026-06-02")
    assert len(projects) == 1
    assert embs.shape == (1, 2)
    assert projects[0]["stars"] == 99
    assert projects[0]["first_seen"] == "2026-06-01"
    assert projects[0]["trending_history"] == ["2026-06-01", "2026-06-02"]
    assert np.allclose(embs[0], [0.0, 1.0])


def test_save_rejects_misaligned(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        store.save(str(tmp_path), [_item("a/x")], np.zeros((2, 2), dtype=np.float32))


def test_merge_dedups_by_full_name_keeping_priority_and_alignment():
    # 第一个 store 优先(收藏),与第二个(trending)有重叠 b/y
    fav_p = [_item("b/y", desc="fav"), _item("c/z")]
    fav_e = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    tr_p = [_item("a/x"), _item("b/y", desc="trending")]
    tr_e = np.array([[0.5, 0.5], [0.9, 0.1]], dtype=np.float32)

    projects, embs = store.merge([(fav_p, fav_e), (tr_p, tr_e)])
    names = [p["full_name"] for p in projects]
    assert names == ["b/y", "c/z", "a/x"]          # 收藏优先,b/y 去重只留一次
    assert projects[0]["description"] == "fav"      # 保留的是收藏那条
    assert embs.shape == (3, 2)
    assert np.allclose(embs[0], [1.0, 0.0])         # b/y 用的是收藏的向量(对齐正确)
    assert np.allclose(embs[2], [0.5, 0.5])         # a/x 用 trending 的向量


def test_merge_handles_empty_stores():
    projects, embs = store.merge([([], np.zeros((0, 0), dtype=np.float32))])
    assert projects == [] and embs.shape[0] == 0


def test_cosine_topk_orders_by_similarity():
    embs = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
    q = np.array([1.0, 0.0], dtype=np.float32)
    res = store.cosine_topk(q, embs, k=2)
    assert [i for i, _ in res] == [0, 2]
    assert res[0][1] > res[1][1]

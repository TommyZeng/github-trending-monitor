# GitHub 热门项目监控 + 语义搜索 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个个人用的 GitHub 热门项目监控应用:GitHub Actions 每日抓取 trending、补元数据、算向量、推送 Discord 并把数据写回仓库;本机 FastAPI 网页对累积库做语义搜索 + GitHub 全站关键词搜索。

**Architecture:** 采集端(Actions 定时)与搜索端(本机网页)通过 git 仓库的数据文件解耦。数据存为 `data/projects.jsonl`(元数据)+ `data/embeddings.npy`(行对齐向量),无数据库服务。各组件单一职责、可独立测试。

**Tech Stack:** Python 3.13、requests、beautifulsoup4、numpy、sentence-transformers(BGE 多语言)、FastAPI + uvicorn、PyYAML、pytest。

---

## File Structure

项目根:`/home/afei/workspace/github-trending-monitor`

```
github-trending-monitor/
├── requirements.txt                 # 依赖
├── config.yaml                      # 可改配置(非密钥)
├── .gitignore
├── ghtrend/
│   ├── __init__.py
│   ├── config.py                    # 配置 + 环境密钥读取
│   ├── store.py                     # jsonl+npy 读写、去重、行对齐、余弦检索
│   ├── embedder.py                  # 文本→向量(采集端/搜索端共用)
│   ├── trending_fetcher.py          # 抓 github.com/trending 并解析
│   ├── github_enricher.py           # GitHub API 补元数据 + 关键词搜索
│   ├── discord_notifier.py          # Discord Webhook 推送
│   ├── collect.py                   # 采集主流程 + __main__ 入口
│   └── web.py                       # FastAPI 应用 + create_app()
├── templates/
│   └── index.html                   # 搜索单页
├── data/
│   └── .gitkeep                     # 运行后生成 projects.jsonl / embeddings.npy
├── tests/
│   ├── __init__.py
│   ├── fixtures/
│   │   └── trending_sample.html
│   ├── test_store.py
│   ├── test_embedder.py
│   ├── test_trending_fetcher.py
│   ├── test_github_enricher.py
│   ├── test_discord_notifier.py
│   ├── test_collect.py
│   └── test_web.py
└── .github/workflows/daily.yml      # 每日定时采集
```

**模块接口约定(后续任务必须保持一致):**

- `config.Config` 字段:`daily_top_n:int=10`、`languages:list[str]|None=None`、`semantic_top_k:int=10`、`embedding_model:str="BAAI/bge-m3"`、`trending_since:str="daily"`、`data_dir:str="data"`。`load_config(path)->Config`;`get_webhook_url()->str`;`get_github_token()->str|None`。
- `store`:`load(data_dir)->tuple[list[dict],np.ndarray]`;`save(data_dir,projects,embeddings)->None`;`upsert(projects,embeddings,new_items,new_embeddings,today)->tuple[list[dict],np.ndarray]`;`cosine_topk(query_vec,embeddings,k)->list[tuple[int,float]]`。
- `embedder`:`build_text(project:dict)->str`;`Embedder(model_name,model=None).encode(texts:list[str])->np.ndarray`(L2 归一化、float32)。
- `trending_fetcher`:`parse_trending(html:str)->list[dict]`;`fetch_trending(since="daily",language=None,session=None)->list[dict]`。每个 dict 至少含 `full_name`、`url`。
- `github_enricher`:`enrich(full_name,token=None,session=None)->dict|None`;`search_repos(query,token=None,limit=10,session=None)->list[dict]`。
- `discord_notifier`:`build_payload(projects,title)->dict`;`send(webhook_url,projects,title,session=None)->None`。
- `collect`:`run(config,webhook_url,github_token,embedder,today=None,fetcher=...,enricher=...,notifier=...)->None`。
- `web`:`create_app(config,embedder,github_token)->FastAPI`,路由 `GET /`、`GET /api/semantic?q=`、`GET /api/keyword?q=`。

**项目字典(project dict)统一字段:** `full_name`、`url`、`description`、`stars:int`、`language`、`topics:list[str]`、`readme_excerpt`、`first_seen`、`trending_history:list[str]`。

---

## Task 1: 项目脚手架与依赖

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `config.yaml`
- Create: `ghtrend/__init__.py`
- Create: `tests/__init__.py`
- Create: `data/.gitkeep`

- [ ] **Step 1: 写依赖文件**

`requirements.txt`:
```
requests>=2.32
beautifulsoup4>=4.12
numpy>=1.26
sentence-transformers>=3.0
fastapi>=0.110
uvicorn>=0.29
PyYAML>=6.0
pytest>=8.0
httpx>=0.27
```

- [ ] **Step 2: 写 .gitignore**

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
.pytest_cache/
.env
```
> 注意:`data/` **不** 忽略——采集端要把数据提交回仓库。

- [ ] **Step 3: 写默认配置**

`config.yaml`:
```yaml
daily_top_n: 10
languages: null        # null=全部;或 [Python, Rust, TypeScript]
semantic_top_k: 10
embedding_model: "BAAI/bge-m3"
trending_since: "daily"
data_dir: "data"
```

- [ ] **Step 4: 创建空包文件**

`ghtrend/__init__.py`:(空文件)
`tests/__init__.py`:(空文件)
`data/.gitkeep`:(空文件)

- [ ] **Step 5: 创建虚拟环境并安装依赖**

Run:
```bash
cd /home/afei/workspace/github-trending-monitor
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
Expected: 安装成功(sentence-transformers 体积较大,首次较慢)。

- [ ] **Step 6: Commit**

```bash
cd /home/afei/workspace/github-trending-monitor
git add -A
git commit -m "chore: 项目脚手架与依赖"
```

---

## Task 2: config 模块

**Files:**
- Create: `ghtrend/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: 写失败测试**

`tests/test_config.py`:
```python
import os
import textwrap
from ghtrend import config


def test_load_config_reads_yaml(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent("""
        daily_top_n: 5
        languages: [Python, Rust]
        semantic_top_k: 3
        embedding_model: "test-model"
        trending_since: "weekly"
        data_dir: "mydata"
    """))
    cfg = config.load_config(str(p))
    assert cfg.daily_top_n == 5
    assert cfg.languages == ["Python", "Rust"]
    assert cfg.semantic_top_k == 3
    assert cfg.embedding_model == "test-model"
    assert cfg.trending_since == "weekly"
    assert cfg.data_dir == "mydata"


def test_load_config_uses_defaults_for_missing(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("daily_top_n: 7\n")
    cfg = config.load_config(str(p))
    assert cfg.daily_top_n == 7
    assert cfg.languages is None
    assert cfg.semantic_top_k == 10


def test_secret_helpers_read_env(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://hook")
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    assert config.get_webhook_url() == "https://hook"
    assert config.get_github_token() == "tok"


def test_get_github_token_optional(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert config.get_github_token() is None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL(`ModuleNotFoundError` 或 `AttributeError: load_config`)

- [ ] **Step 3: 写实现**

`ghtrend/config.py`:
```python
import os
from dataclasses import dataclass, field

import yaml


@dataclass
class Config:
    daily_top_n: int = 10
    languages: list[str] | None = None
    semantic_top_k: int = 10
    embedding_model: str = "BAAI/bge-m3"
    trending_since: str = "daily"
    data_dir: str = "data"


def load_config(path: str = "config.yaml") -> Config:
    data = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    known = {f for f in Config().__dataclass_fields__}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in data.items() if k in known}
    return Config(**filtered)


def get_webhook_url() -> str:
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        raise RuntimeError("环境变量 DISCORD_WEBHOOK_URL 未设置")
    return url


def get_github_token() -> str | None:
    return os.environ.get("GITHUB_TOKEN")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS(4 passed)

- [ ] **Step 5: Commit**

```bash
git add ghtrend/config.py tests/test_config.py
git commit -m "feat: config 模块(yaml 配置 + 环境密钥)"
```

---

## Task 3: store 模块(存储/去重/行对齐/检索)

**Files:**
- Create: `ghtrend/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: 写失败测试**

`tests/test_store.py`:
```python
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


def test_cosine_topk_orders_by_similarity():
    embs = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
    q = np.array([1.0, 0.0], dtype=np.float32)
    res = store.cosine_topk(q, embs, k=2)
    assert [i for i, _ in res] == [0, 2]
    assert res[0][1] > res[1][1]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest tests/test_store.py -v`
Expected: FAIL(`ModuleNotFoundError: ghtrend.store`)

- [ ] **Step 3: 写实现**

`ghtrend/store.py`:
```python
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


_UPDATE_FIELDS = ("url", "description", "stars", "language", "topics", "readme_excerpt")


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
```
> 说明:向量已在 embedder 里 L2 归一化,所以点积即余弦相似度。

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest tests/test_store.py -v`
Expected: PASS(5 passed)

- [ ] **Step 5: Commit**

```bash
git add ghtrend/store.py tests/test_store.py
git commit -m "feat: store(jsonl+npy 存储、去重、行对齐、余弦检索)"
```

---

## Task 4: embedder 模块

**Files:**
- Create: `ghtrend/embedder.py`
- Test: `tests/test_embedder.py`

- [ ] **Step 1: 写失败测试**

`tests/test_embedder.py`:
```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest tests/test_embedder.py -v`
Expected: FAIL(`ModuleNotFoundError: ghtrend.embedder`)

- [ ] **Step 3: 写实现**

`ghtrend/embedder.py`:
```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest tests/test_embedder.py -v`
Expected: PASS(4 passed)

- [ ] **Step 5: Commit**

```bash
git add ghtrend/embedder.py tests/test_embedder.py
git commit -m "feat: embedder(build_text + 可注入模型的向量编码)"
```

---

## Task 5: trending_fetcher 模块

**Files:**
- Create: `tests/fixtures/trending_sample.html`
- Create: `ghtrend/trending_fetcher.py`
- Test: `tests/test_trending_fetcher.py`

- [ ] **Step 1: 写 HTML fixture**

`tests/fixtures/trending_sample.html`:
```html
<!DOCTYPE html><html><body>
<article class="Box-row">
  <h2 class="h3 lh-condensed">
    <a href="/owner-one/repo-one">owner-one / repo-one</a>
  </h2>
  <p class="col-9 color-fg-muted my-1 pr-4">A cool project.</p>
</article>
<article class="Box-row">
  <h2 class="h3 lh-condensed">
    <a href="/owner-two/repo-two">owner-two / repo-two</a>
  </h2>
  <p class="col-9 color-fg-muted my-1 pr-4">Another one.</p>
</article>
</body></html>
```

- [ ] **Step 2: 写失败测试**

`tests/test_trending_fetcher.py`:
```python
import os
from ghtrend import trending_fetcher


def _fixture():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "trending_sample.html")
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_parse_trending_extracts_full_names():
    items = trending_fetcher.parse_trending(_fixture())
    assert [i["full_name"] for i in items] == ["owner-one/repo-one", "owner-two/repo-two"]
    assert items[0]["url"] == "https://github.com/owner-one/repo-one"


def test_fetch_trending_uses_session(monkeypatch):
    class _Resp:
        text = _fixture()
        def raise_for_status(self): pass
    class _Session:
        def __init__(self): self.called_with = None
        def get(self, url, params=None, timeout=None, headers=None):
            self.called_with = (url, params)
            return _Resp()
    sess = _Session()
    items = trending_fetcher.fetch_trending(since="weekly", language="rust", session=sess)
    assert len(items) == 2
    assert sess.called_with[0] == "https://github.com/trending"
    assert sess.called_with[1]["since"] == "weekly"
    assert sess.called_with[1]["language"] == "rust"
```

- [ ] **Step 3: 运行测试确认失败**

Run: `.venv/bin/pytest tests/test_trending_fetcher.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 4: 写实现**

`ghtrend/trending_fetcher.py`:
```python
import requests
from bs4 import BeautifulSoup

TRENDING_URL = "https://github.com/trending"


def parse_trending(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for article in soup.select("article.Box-row"):
        a = article.select_one("h2 a")
        if not a or not a.get("href"):
            continue
        full_name = a["href"].strip("/")
        if full_name.count("/") != 1:
            continue
        items.append({
            "full_name": full_name,
            "url": f"https://github.com/{full_name}",
        })
    return items


def fetch_trending(since: str = "daily", language=None, session=None) -> list[dict]:
    sess = session or requests
    params = {"since": since}
    if language:
        params["language"] = language
    resp = sess.get(
        TRENDING_URL, params=params, timeout=30,
        headers={"User-Agent": "ghtrend/1.0"})
    resp.raise_for_status()
    return parse_trending(resp.text)
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv/bin/pytest tests/test_trending_fetcher.py -v`
Expected: PASS(2 passed)

- [ ] **Step 6: Commit**

```bash
git add ghtrend/trending_fetcher.py tests/test_trending_fetcher.py tests/fixtures/trending_sample.html
git commit -m "feat: trending_fetcher(抓取并解析 github trending)"
```

---

## Task 6: github_enricher 模块

**Files:**
- Create: `ghtrend/github_enricher.py`
- Test: `tests/test_github_enricher.py`

- [ ] **Step 1: 写失败测试**

`tests/test_github_enricher.py`:
```python
import base64
from ghtrend import github_enricher


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
    def json(self): return self._payload
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")


class _Session:
    def __init__(self, routes): self.routes = routes; self.calls = []
    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append((url, params))
        return self.routes[url]


def test_enrich_returns_metadata():
    readme_b64 = base64.b64encode(b"# Title\nLong readme body here.").decode()
    sess = _Session({
        "https://api.github.com/repos/a/x": _Resp({
            "stargazers_count": 123, "description": "desc",
            "language": "Python", "topics": ["ai", "cli"],
            "html_url": "https://github.com/a/x",
        }),
        "https://api.github.com/repos/a/x/readme": _Resp({"content": readme_b64}),
    })
    out = github_enricher.enrich("a/x", token="t", session=sess)
    assert out["full_name"] == "a/x"
    assert out["stars"] == 123
    assert out["description"] == "desc"
    assert out["language"] == "Python"
    assert out["topics"] == ["ai", "cli"]
    assert "readme" in out["readme_excerpt"].lower()


def test_enrich_returns_none_on_repo_error():
    sess = _Session({"https://api.github.com/repos/a/x": _Resp({}, status=404)})
    assert github_enricher.enrich("a/x", session=sess) is None


def test_search_repos_maps_results():
    sess = _Session({
        "https://api.github.com/search/repositories": _Resp({"items": [
            {"full_name": "a/x", "stargazers_count": 5, "description": "d",
             "language": "Go", "topics": ["t"], "html_url": "https://github.com/a/x"},
        ]})
    })
    res = github_enricher.search_repos("rust cli", token="t", limit=5, session=sess)
    assert len(res) == 1
    assert res[0]["full_name"] == "a/x"
    assert res[0]["stars"] == 5
    assert sess.calls[0][1]["q"] == "rust cli"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest tests/test_github_enricher.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 写实现**

`ghtrend/github_enricher.py`:
```python
import base64

import requests

API = "https://api.github.com"


def _headers(token):
    h = {"Accept": "application/vnd.github+json", "User-Agent": "ghtrend/1.0"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _excerpt(readme_payload: dict, limit: int = 500) -> str:
    content = readme_payload.get("content")
    if not content:
        return ""
    try:
        text = base64.b64decode(content).decode("utf-8", errors="ignore")
    except Exception:
        return ""
    return text.strip()[:limit]


def enrich(full_name: str, token=None, session=None) -> dict | None:
    sess = session or requests
    try:
        repo_resp = sess.get(f"{API}/repos/{full_name}", timeout=30, headers=_headers(token))
        repo_resp.raise_for_status()
        repo = repo_resp.json()
    except Exception:
        return None

    readme_excerpt = ""
    try:
        rd = sess.get(f"{API}/repos/{full_name}/readme", timeout=30, headers=_headers(token))
        rd.raise_for_status()
        readme_excerpt = _excerpt(rd.json())
    except Exception:
        readme_excerpt = ""

    return {
        "full_name": full_name,
        "url": repo.get("html_url", f"https://github.com/{full_name}"),
        "description": repo.get("description"),
        "stars": repo.get("stargazers_count", 0),
        "language": repo.get("language"),
        "topics": repo.get("topics", []) or [],
        "readme_excerpt": readme_excerpt,
    }


def search_repos(query: str, token=None, limit: int = 10, session=None) -> list[dict]:
    sess = session or requests
    resp = sess.get(
        f"{API}/search/repositories",
        params={"q": query, "sort": "stars", "order": "desc", "per_page": limit},
        timeout=30, headers=_headers(token))
    resp.raise_for_status()
    out = []
    for it in resp.json().get("items", []):
        out.append({
            "full_name": it.get("full_name"),
            "url": it.get("html_url"),
            "description": it.get("description"),
            "stars": it.get("stargazers_count", 0),
            "language": it.get("language"),
            "topics": it.get("topics", []) or [],
        })
    return out
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest tests/test_github_enricher.py -v`
Expected: PASS(3 passed)

- [ ] **Step 5: Commit**

```bash
git add ghtrend/github_enricher.py tests/test_github_enricher.py
git commit -m "feat: github_enricher(元数据补充 + 关键词搜索)"
```

---

## Task 7: discord_notifier 模块

**Files:**
- Create: `ghtrend/discord_notifier.py`
- Test: `tests/test_discord_notifier.py`

- [ ] **Step 1: 写失败测试**

`tests/test_discord_notifier.py`:
```python
from ghtrend import discord_notifier


def _p(name, stars):
    return {"full_name": name, "url": f"https://github.com/{name}",
            "description": "desc", "stars": stars, "language": "Python",
            "topics": ["ai"]}


def test_build_payload_makes_embeds():
    payload = discord_notifier.build_payload([_p("a/x", 10), _p("b/y", 5)], "Title")
    assert payload["content"] == "Title"
    assert len(payload["embeds"]) == 2
    assert payload["embeds"][0]["title"] == "a/x"
    assert payload["embeds"][0]["url"] == "https://github.com/a/x"
    assert "10" in payload["embeds"][0]["description"]


def test_build_payload_caps_at_10():
    payload = discord_notifier.build_payload([_p(f"o/r{i}", i) for i in range(15)], "T")
    assert len(payload["embeds"]) == 10


def test_send_posts_payload(monkeypatch):
    captured = {}
    class _Session:
        def post(self, url, json=None, timeout=None):
            captured["url"] = url; captured["json"] = json
            class R:
                def raise_for_status(self): pass
            return R()
    discord_notifier.send("https://hook", [_p("a/x", 1)], "T", session=_Session())
    assert captured["url"] == "https://hook"
    assert captured["json"]["embeds"][0]["title"] == "a/x"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest tests/test_discord_notifier.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 写实现**

`ghtrend/discord_notifier.py`:
```python
import requests

MAX_EMBEDS = 10


def build_payload(projects: list[dict], title: str) -> dict:
    embeds = []
    for p in projects[:MAX_EMBEDS]:
        topics = ", ".join(p.get("topics") or [])
        desc = p.get("description") or "(no description)"
        lang = p.get("language") or "?"
        embeds.append({
            "title": p["full_name"],
            "url": p["url"],
            "description": f"⭐ {p.get('stars', 0)} | {lang}\n{desc}"
                           + (f"\n`{topics}`" if topics else ""),
        })
    return {"content": title, "embeds": embeds}


def send(webhook_url: str, projects: list[dict], title: str, session=None) -> None:
    sess = session or requests
    resp = sess.post(webhook_url, json=build_payload(projects, title), timeout=30)
    resp.raise_for_status()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest tests/test_discord_notifier.py -v`
Expected: PASS(3 passed)

- [ ] **Step 5: Commit**

```bash
git add ghtrend/discord_notifier.py tests/test_discord_notifier.py
git commit -m "feat: discord_notifier(Webhook 卡片推送)"
```

---

## Task 8: collect 采集主流程

**Files:**
- Create: `ghtrend/collect.py`
- Test: `tests/test_collect.py`

- [ ] **Step 1: 写失败测试**

`tests/test_collect.py`:
```python
import json
import os
import numpy as np
from ghtrend import collect, store
from ghtrend.config import Config


class _DummyEmbedder:
    def encode(self, texts):
        return np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)


def test_run_collects_dedups_and_notifies(tmp_path):
    cfg = Config(daily_top_n=2, data_dir=str(tmp_path))
    sent = {}

    def fake_fetcher(since, language=None, session=None):
        return [{"full_name": "a/x", "url": "u"}, {"full_name": "b/y", "url": "u"}]

    def fake_enricher(full_name, token=None, session=None):
        stars = {"a/x": 50, "b/y": 99}[full_name]
        return {"full_name": full_name, "url": f"https://github.com/{full_name}",
                "description": "d", "stars": stars, "language": "Python",
                "topics": ["ai"], "readme_excerpt": "r"}

    def fake_notifier(webhook_url, projects, title, session=None):
        sent["title"] = title
        sent["names"] = [p["full_name"] for p in projects]

    collect.run(cfg, "https://hook", "tok", _DummyEmbedder(), today="2026-06-02",
                fetcher=fake_fetcher, enricher=fake_enricher, notifier=fake_notifier)

    projects, embs = store.load(str(tmp_path))
    assert len(projects) == 2
    assert embs.shape[0] == 2
    # 推送按 star 降序取 top 2
    assert sent["names"] == ["b/y", "a/x"]
    assert "2026-06-02" in sent["title"]


def test_run_skips_failed_enrichment(tmp_path):
    cfg = Config(data_dir=str(tmp_path))

    def fake_fetcher(since, language=None, session=None):
        return [{"full_name": "a/x", "url": "u"}, {"full_name": "bad/repo", "url": "u"}]

    def fake_enricher(full_name, token=None, session=None):
        return None if full_name == "bad/repo" else {
            "full_name": full_name, "url": "u", "description": "d", "stars": 1,
            "language": "Go", "topics": [], "readme_excerpt": ""}

    collect.run(cfg, "https://hook", None, _DummyEmbedder(), today="2026-06-02",
                fetcher=fake_fetcher, enricher=fake_enricher,
                notifier=lambda *a, **k: None)

    projects, _ = store.load(str(tmp_path))
    assert [p["full_name"] for p in projects] == ["a/x"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest tests/test_collect.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 写实现**

`ghtrend/collect.py`:
```python
from datetime import date, timezone, datetime

from . import store
from .config import Config, load_config, get_webhook_url, get_github_token
from .embedder import Embedder, build_text
from .trending_fetcher import fetch_trending
from .github_enricher import enrich
from .discord_notifier import send


def run(config: Config, webhook_url: str, github_token, embedder, today=None,
        fetcher=fetch_trending, enricher=enrich, notifier=send) -> None:
    today = today or datetime.now(timezone.utc).date().isoformat()

    raw = fetcher(since=config.trending_since, language=config.languages[0]
                  if config.languages else None)
    enriched = []
    for item in raw:
        meta = enricher(item["full_name"], token=github_token)
        if meta:
            enriched.append(meta)

    if not enriched:
        print("没有可用项目,跳过本次。")
        return

    texts = [build_text(p) for p in enriched]
    vecs = embedder.encode(texts)

    projects, embeddings = store.load(config.data_dir)
    projects, embeddings = store.upsert(projects, embeddings, enriched, vecs, today)
    store.save(config.data_dir, projects, embeddings)
    print(f"已写入 {len(enriched)} 个项目,库总量 {len(projects)}。")

    top = sorted(enriched, key=lambda p: p.get("stars", 0), reverse=True)[:config.daily_top_n]
    notifier(webhook_url, top, f"🔥 GitHub Trending {today}")
    print("Discord 推送完成。")


def main() -> None:
    cfg = load_config()
    embedder = Embedder(cfg.embedding_model)
    run(cfg, get_webhook_url(), get_github_token(), embedder)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest tests/test_collect.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: Commit**

```bash
git add ghtrend/collect.py tests/test_collect.py
git commit -m "feat: collect 采集主流程(抓取→补元数据→向量→存储→推送)"
```

---

## Task 9: web 搜索服务(FastAPI)

**Files:**
- Create: `templates/index.html`
- Create: `ghtrend/web.py`
- Test: `tests/test_web.py`

- [ ] **Step 1: 写搜索页**

`templates/index.html`:
```html
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GitHub 热门项目搜索</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 820px; margin: 2rem auto; padding: 0 1rem; }
    .bar { display: flex; gap: .5rem; margin-bottom: 1rem; }
    input { flex: 1; padding: .6rem; font-size: 1rem; }
    button { padding: .6rem 1rem; cursor: pointer; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: .8rem 1rem; margin: .6rem 0; }
    .card a { font-weight: 600; font-size: 1.05rem; text-decoration: none; }
    .meta { color: #666; font-size: .85rem; margin: .3rem 0; }
    .topics { color: #0969da; font-size: .8rem; }
  </style>
</head>
<body>
  <h2>GitHub 热门项目搜索</h2>
  <div class="bar">
    <input id="q" placeholder="输入一句话,如:Rust 写的终端 AI 助手" />
    <button onclick="run('semantic')">语义搜索</button>
    <button onclick="run('keyword')">GitHub 关键词</button>
  </div>
  <div id="results"></div>
  <script>
    async function run(mode) {
      const q = document.getElementById('q').value.trim();
      if (!q) return;
      const box = document.getElementById('results');
      box.innerHTML = '搜索中…';
      const r = await fetch(`/api/${mode}?q=` + encodeURIComponent(q));
      const data = await r.json();
      if (!data.results.length) { box.innerHTML = '无结果'; return; }
      box.innerHTML = data.results.map(p => `
        <div class="card">
          <a href="${p.url}" target="_blank">${p.full_name}</a>
          <div class="meta">⭐ ${p.stars ?? 0} · ${p.language ?? '?'}${p.score != null ? ' · 相似度 ' + p.score.toFixed(3) : ''}</div>
          <div>${p.description ?? ''}</div>
          <div class="topics">${(p.topics || []).join(', ')}</div>
        </div>`).join('');
    }
  </script>
</body>
</html>
```

- [ ] **Step 2: 写失败测试**

`tests/test_web.py`:
```python
import numpy as np
from fastapi.testclient import TestClient
from ghtrend import web, store
from ghtrend.config import Config


class _DummyEmbedder:
    def encode(self, texts):
        # 查询 "x" -> 偏向第一个项目
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


def _seed(data_dir):
    projects, embs = store.load(data_dir)
    items = [
        {"full_name": "a/x", "url": "https://github.com/a/x", "description": "d1",
         "stars": 10, "language": "Python", "topics": ["ai"], "readme_excerpt": ""},
        {"full_name": "b/y", "url": "https://github.com/b/y", "description": "d2",
         "stars": 20, "language": "Go", "topics": ["cli"], "readme_excerpt": ""},
    ]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    projects, embs = store.upsert(projects, embs, items, vecs, "2026-06-02")
    store.save(data_dir, projects, embs)


def test_index_served(tmp_path):
    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), None)
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "GitHub" in r.text


def test_semantic_search_ranks(tmp_path):
    _seed(str(tmp_path))
    app = web.create_app(Config(data_dir=str(tmp_path), semantic_top_k=2),
                         _DummyEmbedder(), None)
    client = TestClient(app)
    r = client.get("/api/semantic", params={"q": "anything"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert results[0]["full_name"] == "a/x"   # 查询向量[1,0] 最接近 a/x
    assert "score" in results[0]


def test_keyword_search_uses_github(tmp_path, monkeypatch):
    app = web.create_app(Config(data_dir=str(tmp_path)), _DummyEmbedder(), "tok")
    monkeypatch.setattr(web.github_enricher, "search_repos",
                        lambda q, token=None, limit=10: [
                            {"full_name": "z/found", "url": "u", "description": "d",
                             "stars": 1, "language": "C", "topics": []}])
    client = TestClient(app)
    r = client.get("/api/keyword", params={"q": "rust"})
    assert r.json()["results"][0]["full_name"] == "z/found"
```

- [ ] **Step 3: 运行测试确认失败**

Run: `.venv/bin/pytest tests/test_web.py -v`
Expected: FAIL(`ModuleNotFoundError: ghtrend.web`)

- [ ] **Step 4: 写实现**

`ghtrend/web.py`:
```python
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from . import store, github_enricher
from .config import Config
from .embedder import build_text  # noqa: F401  (保持依赖显式)

_TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "index.html")


def create_app(config: Config, embedder, github_token) -> FastAPI:
    app = FastAPI(title="GitHub Trending Search")

    @app.get("/", response_class=HTMLResponse)
    def index():
        with open(_TEMPLATE, encoding="utf-8") as f:
            return f.read()

    @app.get("/api/semantic")
    def semantic(q: str):
        projects, embeddings = store.load(config.data_dir)
        if not projects:
            return JSONResponse({"results": []})
        qvec = embedder.encode([q])[0]
        ranked = store.cosine_topk(qvec, embeddings, config.semantic_top_k)
        results = []
        for i, score in ranked:
            p = dict(projects[i])
            p["score"] = score
            results.append(p)
        return JSONResponse({"results": results})

    @app.get("/api/keyword")
    def keyword(q: str):
        results = github_enricher.search_repos(
            q, token=github_token, limit=config.semantic_top_k)
        return JSONResponse({"results": results})

    return app


def main() -> None:
    import uvicorn
    from .config import load_config, get_github_token
    from .embedder import Embedder
    cfg = load_config()
    app = create_app(cfg, Embedder(cfg.embedding_model), get_github_token())
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv/bin/pytest tests/test_web.py -v`
Expected: PASS(3 passed)

- [ ] **Step 6: Commit**

```bash
git add ghtrend/web.py templates/index.html tests/test_web.py
git commit -m "feat: web(语义搜索 + GitHub 关键词搜索 + 单页界面)"
```

---

## Task 10: GitHub Actions 每日工作流

**Files:**
- Create: `.github/workflows/daily.yml`

- [ ] **Step 1: 写工作流**

`.github/workflows/daily.yml`:
```yaml
name: daily-collect

on:
  schedule:
    - cron: "0 1 * * *"   # UTC 01:00 = 北京时间 09:00
  workflow_dispatch: {}

permissions:
  contents: write

jobs:
  collect:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Install deps
        run: pip install -r requirements.txt

      - name: Cache HF models
        uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          key: hf-${{ hashFiles('config.yaml') }}

      - name: Run collector
        env:
          DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: python -m ghtrend.collect

      - name: Commit data
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data/
          git diff --cached --quiet || git commit -m "data: trending $(date -u +%F)"
          git push
```

- [ ] **Step 2: 校验 YAML 语法**

Run:
```bash
.venv/bin/python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/daily.yml')); print('yaml ok')"
```
Expected: `yaml ok`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/daily.yml
git commit -m "ci: 每日 trending 采集工作流"
```

---

## Task 11: README、本机运行脚本与全量验证

**Files:**
- Create: `README.md`
- Create: `run_web.sh`

- [ ] **Step 1: 写 README**

`README.md`:
```markdown
# GitHub 热门项目监控 + 语义搜索

每日通过 GitHub Actions 抓取 GitHub Trending,补元数据、算向量、推送 Discord,并把数据写回仓库;
本机网页对累积库做语义搜索 + GitHub 全站关键词搜索。

## 一次性准备
1. 把本项目推到一个 GitHub 仓库。
2. Discord 频道 → 编辑频道 → 整合 → Webhook → 新建 → 复制 URL。
3. 仓库 Settings → Secrets and variables → Actions 新增:
   - `DISCORD_WEBHOOK_URL`(上一步的 URL)
   - (内置 `GITHUB_TOKEN` 已自动提供,无需手动加)
4. 本机:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

## 每日推送
由 `.github/workflows/daily.yml` 定时(默认北京时间 09:00)自动运行。
也可在 Actions 页手动 `Run workflow` 触发。

## 本机搜索
```bash
export GITHUB_TOKEN=<你的 PAT>   # 关键词搜索用,提高速率
git pull                         # 拉取最新数据
./run_web.sh
```
浏览器打开 http://127.0.0.1:8000 。

## 配置
见 `config.yaml`(推送数量、语言过滤、embedding 模型等)。
```

- [ ] **Step 2: 写本机启动脚本**

`run_web.sh`:
```bash
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
git pull --rebase 2>/dev/null || true
.venv/bin/python -m ghtrend.web
```

- [ ] **Step 3: 赋可执行权限**

Run: `chmod +x run_web.sh`
Expected: 无输出

- [ ] **Step 4: 运行全部测试**

Run: `.venv/bin/pytest -v`
Expected: 全部 PASS(config/store/embedder/trending_fetcher/github_enricher/discord_notifier/collect/web)

- [ ] **Step 5: 本机真实冒烟测试(语义库为空也应正常起服务)**

Run:
```bash
.venv/bin/python -c "from ghtrend.web import create_app; from ghtrend.config import load_config; from ghtrend.embedder import Embedder; print('app import ok')"
```
Expected: `app import ok`
> 注:首次 `Embedder` 真正编码会下载 BGE 模型;此步只验证导入与装配,不触发下载。

- [ ] **Step 6: Commit**

```bash
git add README.md run_web.sh
git commit -m "docs: README + 本机启动脚本"
```

---

## Self-Review(已执行)

**Spec 覆盖检查:**
- 每日推送(Discord Webhook)→ Task 7 + Task 8 + Task 10 ✅
- Trending 抓取 + Search API 补元数据 → Task 5 + Task 6 + Task 8 ✅
- 语义搜索(本地累积库 + BGE)→ Task 3(store.cosine_topk)+ Task 4 + Task 9 ✅
- GitHub 关键词搜索 → Task 6(search_repos)+ Task 9 ✅
- 数据存储 jsonl+npy、去重、行对齐 → Task 3 ✅
- 配置项(数量/语言/topk/模型)→ Task 2 + 各处读取 ✅
- 一次性准备(仓库/Webhook/token)→ Task 11 README ✅
- 错误处理(抓取失败跳过、enrich 失败跳过、行对齐校验)→ Task 3 save 校验、Task 6 enrich 返回 None、Task 8 跳过 None ✅

**占位符扫描:** 无 TBD/TODO;每个代码步骤含完整代码。

**类型/签名一致性:** `Config` 字段、`store.upsert/cosine_topk/save/load`、`Embedder.encode`、`enrich/search_repos`、`build_payload/send`、`collect.run`、`create_app` 在各任务中签名一致。project dict 字段统一(full_name/url/description/stars/language/topics/readme_excerpt/first_seen/trending_history)。

**已知取舍:** BGE 真实模型的端到端编码不在单测覆盖(用注入的假模型测接口),避免测试下载大模型;真实模型在 Task 11 Step 5 仅验证装配。

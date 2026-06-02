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

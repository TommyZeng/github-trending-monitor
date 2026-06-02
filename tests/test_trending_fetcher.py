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

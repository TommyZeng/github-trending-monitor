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


def test_build_payload_prefers_chinese_description():
    p = _p("a/x", 10)
    p["description_zh"] = "一个很棒的项目"
    payload = discord_notifier.build_payload([p], "T")
    assert "一个很棒的项目" in payload["embeds"][0]["description"]
    assert "desc" not in payload["embeds"][0]["description"]


def test_build_payload_falls_back_to_english_without_zh():
    payload = discord_notifier.build_payload([_p("a/x", 10)], "T")
    assert "desc" in payload["embeds"][0]["description"]


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

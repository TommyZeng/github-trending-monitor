from datetime import date
from ghtrend import weekly_report as wr


def _p(name, history, first_seen=None, stars=100, zh="中文描述", topics=None):
    return {"full_name": name, "url": f"https://github.com/{name}",
            "description": "en desc", "description_zh": zh, "stars": stars,
            "language": "Python", "topics": topics or [],
            "first_seen": first_seen or history[0], "trending_history": history}


def test_week_window_covers_previous_monday_to_sunday():
    # 周一出报告,覆盖上周一到周日
    assert wr.week_window(date(2026, 10, 26)) == (date(2026, 10, 19), date(2026, 10, 25))


def test_weekly_projects_counts_days_and_marks_new():
    start, end = date(2026, 10, 19), date(2026, 10, 25)
    projects = [
        _p("a/old", ["2026-09-01", "2026-10-20", "2026-10-21", "2026-10-22"], first_seen="2026-09-01"),
        _p("b/new", ["2026-10-24"]),
        _p("c/gone", ["2026-10-01"]),          # 本周没上榜
    ]
    week = {p["full_name"]: p for p in wr.weekly_projects(projects, start, end)}
    assert set(week) == {"a/old", "b/new"}
    assert week["a/old"]["days_on"] == 3 and week["a/old"]["is_new"] is False
    assert week["b/new"]["days_on"] == 1 and week["b/new"]["is_new"] is True


def test_parse_focus_strips_think_and_returns_themes():
    content = ('<think>maybe [x]</think>\n'
               '[{"theme": "AI 编码 Agent", "summary": "技能生态爆发", "repos": ["a/old"]}]')
    themes = wr.parse_focus(content)
    assert themes == [{"theme": "AI 编码 Agent", "summary": "技能生态爆发", "repos": ["a/old"]}]


def test_parse_focus_returns_empty_on_garbage():
    assert wr.parse_focus("不是 JSON") == []


def _sample():
    start, end = date(2026, 10, 19), date(2026, 10, 25)
    week = wr.weekly_projects([
        _p("a/old", ["2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23"], first_seen="2026-09-01"),
        _p("b/new", ["2026-10-24"]),
    ], start, end)
    top = [dict(week[0], week_stars=5000), dict(week[1], week_stars=800)]
    themes = [{"theme": "AI 编码 Agent", "summary": "技能生态爆发", "repos": ["a/old"]}]
    return start, end, week, top, themes


def test_render_markdown_has_all_sections():
    start, end, week, top, themes = _sample()
    md = wr.render_markdown(start, end, week, top, themes)
    for section in ("本周概览", "本周焦点", "本周 Top", "持续霸榜"):
        assert section in md
    assert "AI 编码 Agent" in md and "+5,000" in md
    assert "a/old" in md.split("持续霸榜")[1]      # 上榜 4 天进入持续霸榜


def test_discord_payload_respects_limits():
    start, end, week, top, themes = _sample()
    top = top * 10                                    # 撑长内容
    payload = wr.build_discord_payload(start, end, week, top, themes, report_url="https://x/r.md")
    embeds = payload["embeds"]
    assert len(embeds) <= 10
    assert all(len(e.get("description", "")) <= 4096 for e in embeds)
    assert sum(len(e.get("title", "")) + len(e.get("description", "")) for e in embeds) <= 6000
    assert "https://x/r.md" in str(payload)


def test_build_top_ranks_by_weekly_stars_and_translates_missing():
    start, end, week, _, _ = _sample()
    trending = [
        {"full_name": "x/outsider", "url": "u", "description": "Not in library", "stars": 9, "stars_today": 9000},
        {"full_name": "a/old", "url": "u", "description": "en", "stars": 100, "stars_today": 500},
    ]
    top = wr.build_top(week, trending, translate=lambda texts: ["译:" + t for t in texts])
    assert [p["full_name"] for p in top] == ["x/outsider", "a/old"]
    assert top[0]["week_stars"] == 9000
    assert top[0]["description_zh"] == "译:Not in library"   # 不在库里的补翻
    assert top[1]["days_on"] == 4                              # 库里的带上榜天数


def test_build_top_falls_back_to_days_on_without_trending():
    start, end, week, _, _ = _sample()
    top = wr.build_top(week, [])
    assert [p["full_name"] for p in top] == ["a/old", "b/new"]


def test_render_markdown_uses_latest_stars_and_clips_long_desc():
    start, end, week, top, themes = _sample()
    top[0]["stars"] = 99999                         # 周榜最新 star
    top[0]["description_zh"] = "长" * 300
    md = wr.render_markdown(start, end, week, top, themes)
    assert "99,999" in md.split("持续霸榜")[1]       # 持续霸榜也用最新值
    assert "长" * 100 not in md.split("持续霸榜")[0]  # 表格里已截断

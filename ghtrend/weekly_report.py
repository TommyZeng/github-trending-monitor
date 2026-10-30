"""每周热门周报:汇总上周(周一~周日)上榜项目,LLM 归纳焦点,推 Discord 并存档 Markdown。"""
import json
import os
import re
from datetime import date, datetime, timedelta, timezone

import requests

from .config import load_config, get_webhook_url, get_translate_api_key
from .store import load
from .trending_fetcher import fetch_trending
from .translator import llm_translate_batch

TOP_N = 10
STEADY_DAYS = 4          # 一周上榜 ≥4 天算「持续霸榜」
MAX_ATTEMPTS = 3

_FOCUS_PROMPT = (
    "下面是 GitHub 上周的热门项目(名称 | 简介 | topics | 上榜天数)。"
    "请归纳出 3~4 个本周技术焦点/趋势,每个给一句中文点评(不超过 50 字),并列出最能代表它的 1~3 个项目名。"
    "严格只输出 JSON 数组,形如 "
    '[{{"theme": "主题名", "summary": "一句话点评", "repos": ["owner/repo"]}}],'
    "不要任何解释或代码块标记。\n\n{block}"
)


def week_window(report_day: date) -> tuple[date, date]:
    """报告日所在周的上一个完整周:上周一 ~ 上周日。"""
    end = report_day - timedelta(days=report_day.weekday() + 1)
    return end - timedelta(days=6), end


def weekly_projects(projects: list[dict], start: date, end: date) -> list[dict]:
    """本周上过榜的项目,附 days_on(本周上榜天数)与 is_new(本周首次入库)。"""
    lo, hi = start.isoformat(), end.isoformat()
    out = []
    for p in projects:
        days = [d for d in p.get("trending_history") or [] if lo <= d <= hi]
        if days:
            out.append(dict(p, days_on=len(days),
                            is_new=lo <= (p.get("first_seen") or "") <= hi))
    return out


def parse_focus(content: str) -> list[dict]:
    text = re.sub(r"<think>.*?</think>", "", content or "", flags=re.S).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except Exception:
        return []
    return [d for d in data if isinstance(d, dict) and d.get("theme")]


def summarize_focus(week: list[dict], api_base: str, model: str, api_key=None) -> list[dict]:
    """LLM 归纳本周焦点;失败返回 [](报告照常生成,只是没有焦点板块)。"""
    if not week:
        return []
    block = "\n".join(
        f"{p['full_name']} | {(p.get('description_zh') or p.get('description') or '')[:120]}"
        f" | {', '.join((p.get('topics') or [])[:6])} | {p['days_on']}天"
        for p in sorted(week, key=lambda p: -p["days_on"]))
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for _ in range(MAX_ATTEMPTS):
        try:
            resp = requests.post(
                f"{api_base.rstrip('/')}/chat/completions",
                json={"model": model, "temperature": 0,
                      "messages": [{"role": "user", "content": _FOCUS_PROMPT.format(block=block)}]},
                headers=headers, timeout=120)
            resp.raise_for_status()
            themes = parse_focus(resp.json()["choices"][0]["message"]["content"])
        except Exception:
            continue
        if themes:
            return themes
    return []


def _desc(p: dict) -> str:
    return p.get("description_zh") or p.get("description") or ""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 1] + "…"


def render_markdown(start, end, week, top, themes) -> str:
    new = [p for p in week if p["is_new"]]
    # 周榜里的 star 是最新值,库里的可能是早几天的;同一份报告统一用最新的
    latest = {p["full_name"]: p.get("stars", 0) for p in top}
    week = [dict(p, stars=latest.get(p["full_name"], p.get("stars", 0))) for p in week]
    steady = sorted([p for p in week if p["days_on"] >= STEADY_DAYS],
                    key=lambda p: (-p["days_on"], -p.get("stars", 0)))
    lines = [f"# GitHub 热门周报 {start:%Y-%m-%d} ~ {end:%Y-%m-%d}", "",
             "## 📊 本周概览", "",
             f"本周共 **{len(week)}** 个项目登上 trending,其中新面孔 **{len(new)}** 个,"
             f"持续霸榜(上榜 ≥{STEADY_DAYS} 天)**{len(steady)}** 个。", ""]
    lines += ["## 🎯 本周焦点", ""]
    if themes:
        for t in themes:
            repos = "、".join(f"`{r}`" for r in t.get("repos") or [])
            lines.append(f"- **{t['theme']}**:{t.get('summary', '')}" + (f" 代表项目:{repos}" if repos else ""))
    else:
        lines.append("(本周焦点生成失败)")
    lines += ["", f"## 🔥 本周 Top {len(top)}(按本周新增 star)", "",
              "| # | 项目 | 本周新增 | 总 star | 上榜天数 | 简介 |",
              "|---|---|---|---|---|---|"]
    for i, p in enumerate(top, 1):
        days = f"{p['days_on']} 天" if p.get("days_on") else "-"
        lines.append(f"| {i} | [{p['full_name']}]({p['url']}) | +{p.get('week_stars') or 0:,} "
                     f"| {p.get('stars', 0):,} | {days} | {_clip(_desc(p), 80).replace('|', '/')} |")
    lines += ["", "## 🏃 持续霸榜", ""]
    if steady:
        for p in steady:
            lines.append(f"- [{p['full_name']}]({p['url']}) · 上榜 {p['days_on']} 天 · ⭐ {p.get('stars', 0):,} — {_desc(p)}")
    else:
        lines.append(f"(本周没有上榜 ≥{STEADY_DAYS} 天的项目)")
    return "\n".join(lines) + "\n"


def build_discord_payload(start, end, week, top, themes, report_url=None) -> dict:
    """三段 embed:焦点 / Top / 持续霸榜。总字数控制在 Discord 单条消息 6000 字以内。"""
    new = sum(1 for p in week if p["is_new"])
    focus = "\n".join(f"**{t['theme']}** — {t.get('summary', '')}" for t in themes) or "(本周焦点生成失败)"
    top_lines = "\n".join(
        f"{i}. [{p['full_name']}]({p['url']}) +{p.get('week_stars') or 0:,}⭐"
        + (f" · 上榜{p['days_on']}天" if p.get("days_on") else "")
        + f"\n　{_clip(_desc(p), 60)}"
        for i, p in enumerate(top, 1))
    steady = sorted([p for p in week if p["days_on"] >= STEADY_DAYS], key=lambda p: -p["days_on"])
    steady_lines = "\n".join(f"[{p['full_name']}]({p['url']}) · {p['days_on']}天" for p in steady) or "(无)"
    embeds = [
        {"title": f"🎯 本周焦点 · {len(week)} 个上榜 / {new} 个新面孔", "description": _clip(focus, 1500)},
        {"title": f"🔥 本周 Top {len(top)}", "description": _clip(top_lines, 3000)},
        {"title": "🏃 持续霸榜", "description": _clip(steady_lines, 800)},
    ]
    content = f"📅 **GitHub 热门周报** {start:%m/%d} ~ {end:%m/%d}"
    if report_url:
        content += f"\n完整报告:{report_url}"
    return {"content": content, "embeds": embeds}


def build_top(week: list[dict], weekly_trending: list[dict], translate=None) -> list[dict]:
    """Top N 以 GitHub 周榜「本周新增 star」为准;拿不到周榜时退回按本周上榜天数排序。"""
    by_name = {p["full_name"]: p for p in week}
    if weekly_trending:
        ranked = sorted(weekly_trending, key=lambda t: t.get("stars_today") or 0, reverse=True)[:TOP_N]
        top = [dict(by_name.get(t["full_name"], t), week_stars=t.get("stars_today"),
                    stars=t.get("stars") or by_name.get(t["full_name"], {}).get("stars", 0))
               for t in ranked]
    else:
        top = sorted(week, key=lambda p: (-p["days_on"], -p.get("stars", 0)))[:TOP_N]
    # 周榜里没进过本地库的项目没有中文描述,补翻
    missing = [p for p in top if not p.get("description_zh") and p.get("description")]
    if missing and translate:
        zh = translate([p["description"] for p in missing])
        for p, z in zip(missing, zh or []):
            p["description_zh"] = z
    return top


def main() -> None:
    cfg = load_config()
    report_day = datetime.now(timezone.utc).date()
    start, end = week_window(report_day)
    projects, _ = load(cfg.data_dir)
    week = weekly_projects(projects, start, end)

    base = cfg.translate_api_base or cfg.llm_api_base
    model = cfg.translate_model or cfg.llm_model
    key = get_translate_api_key()
    try:
        weekly_trending = fetch_trending(since="weekly")
    except Exception as e:
        print(f"⚠ 拉取 GitHub 周榜失败,Top 改按本周上榜天数排序: {e}")
        weekly_trending = []
    top = build_top(week, weekly_trending,
                    translate=lambda texts: llm_translate_batch(texts, base, model, api_key=key))
    themes = summarize_focus(week, base, model, api_key=key)
    if not themes:
        print("⚠ 本周焦点生成失败")

    iso_year, iso_week, _ = start.isocalendar()
    name = f"{iso_year}-W{iso_week:02d}.md"
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", name), "w", encoding="utf-8") as f:
        f.write(render_markdown(start, end, week, top, themes))
    print(f"已生成 reports/{name}:本周 {len(week)} 个项目,焦点 {len(themes)} 个")

    repo = os.environ.get("GITHUB_REPOSITORY")
    report_url = f"https://github.com/{repo}/blob/master/reports/{name}" if repo else None
    resp = requests.post(get_webhook_url(),
                         json=build_discord_payload(start, end, week, top, themes, report_url),
                         timeout=30)
    resp.raise_for_status()
    print("Discord 周报推送完成。")


if __name__ == "__main__":
    main()

import re

import requests
from bs4 import BeautifulSoup

TRENDING_URL = "https://github.com/trending"

# 形如 "1,234 stars today" / "56 stars this week"
_STARS_PERIOD = re.compile(r"([\d,]+)\s+stars?\s+(today|this week|this month)", re.I)


def _stars_in_period(article) -> int | None:
    el = article.select_one("span.float-sm-right")
    if not el:
        return None
    m = _STARS_PERIOD.search(el.get_text(" ", strip=True))
    return int(m.group(1).replace(",", "")) if m else None


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
            "stars_today": _stars_in_period(article),
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

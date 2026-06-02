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

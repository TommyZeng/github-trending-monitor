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

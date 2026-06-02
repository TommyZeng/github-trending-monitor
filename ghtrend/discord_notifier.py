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

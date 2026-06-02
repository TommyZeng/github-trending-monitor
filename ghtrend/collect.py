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

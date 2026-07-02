from datetime import timezone, datetime

from . import store
from .config import (Config, load_config, get_webhook_url, get_github_token,
                     get_llm_api_key)
from .embedder import Embedder, build_text
from .trending_fetcher import fetch_trending
from .github_enricher import enrich
from .discord_notifier import send
from .translator import translate_to_zh, llm_translate_batch


def run(config: Config, webhook_url: str, github_token, embedder, today=None,
        fetcher=fetch_trending, enricher=enrich, notifier=send,
        translate=translate_to_zh, translate_batch=None) -> None:
    today = today or datetime.now(timezone.utc).date().isoformat()

    raw = fetcher(since=config.trending_since, language=config.languages[0]
                  if config.languages else None)
    enriched = []
    for item in raw:
        meta = enricher(item["full_name"], token=github_token)
        if meta:
            meta["stars_today"] = item.get("stars_today")   # 今日新增 star(来自 trending 页)
            enriched.append(meta)

    if not enriched:
        print("没有可用项目,跳过本次。")
        return

    # 中文描述:优先 LLM 批量翻译(一次调用),失败回退逐条翻译
    descs = [p.get("description") for p in enriched]
    zh = translate_batch(descs) if translate_batch else None
    if zh is None:
        zh = [translate(d) for d in descs]
    for p, z in zip(enriched, zh):
        p["description_zh"] = z

    texts = [build_text(p) for p in enriched]
    vecs = embedder.encode(texts)

    projects, embeddings = store.load(config.data_dir)
    projects, embeddings = store.upsert(projects, embeddings, enriched, vecs, today)
    store.save(config.data_dir, projects, embeddings)
    print(f"已写入 {len(enriched)} 个项目,库总量 {len(projects)}。")

    # 按「今日新增 star」降序推送(无该数据的回退用总 star)
    top = sorted(enriched, key=lambda p: (p.get("stars_today") or 0, p.get("stars", 0)),
                 reverse=True)[:config.daily_top_n]
    notifier(webhook_url, top, f"🔥 GitHub Trending {today}")
    print("Discord 推送完成。")


def main() -> None:
    cfg = load_config()
    embedder = Embedder(cfg.embedding_model)
    # 配了 LLM(公网可达,如 DashScope)就用它批量翻译;失败自动回退 Google 逐条翻译
    translate_batch = None
    llm_key = get_llm_api_key()
    if cfg.llm_api_base and llm_key:
        translate_batch = lambda texts: llm_translate_batch(
            texts, cfg.llm_api_base, cfg.llm_model, api_key=llm_key)
        print(f"描述翻译使用 LLM: {cfg.llm_model} @ {cfg.llm_api_base}")
    run(cfg, get_webhook_url(), get_github_token(), embedder,
        translate_batch=translate_batch)


if __name__ == "__main__":
    main()

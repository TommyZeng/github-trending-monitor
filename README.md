# GitHub 热门项目监控 + 语义搜索

每日通过 GitHub Actions 抓取 GitHub Trending,补元数据、算向量、推送 Discord,并把数据写回仓库;
本机网页对累积库做语义搜索 + GitHub 全站关键词搜索。

## 一次性准备
1. 把本项目推到一个 GitHub 仓库。
2. Discord 频道 → 编辑频道 → 整合 → Webhook → 新建 → 复制 URL。
3. 仓库 Settings → Secrets and variables → Actions 新增:
   - `DISCORD_WEBHOOK_URL`(上一步的 URL)
   - (内置 `GITHUB_TOKEN` 已自动提供,无需手动加)
4. 本机:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

## 每日推送
由 `.github/workflows/daily.yml` 定时(默认北京时间 09:00)自动运行。
也可在 Actions 页手动 `Run workflow` 触发。

## 本机搜索
```bash
export GITHUB_TOKEN=<你的 PAT>   # 关键词搜索用,提高速率
git pull                         # 拉取最新数据
./run_web.sh
```
浏览器打开 http://127.0.0.1:8000 。

## 配置
见 `config.yaml`(推送数量、语言过滤、embedding 模型等)。

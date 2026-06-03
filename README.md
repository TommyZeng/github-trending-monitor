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
1. 在项目根创建 `.env`(已被 `.gitignore` 忽略,不会提交):
   ```
   EMBEDDING_API_KEY=sk-xxx        # 在线 embedding 服务的 key
   # GITHUB_TOKEN=ghp_xxx          # 可选,关键词搜索提速
   ```
2. 启动:
   ```bash
   ./run_web.sh                    # 自动载入 .env、git pull 最新数据、起服务
   ```
   浏览器打开 http://127.0.0.1:8000 。

### 三种搜索模式
- **本地语义**:在累积的 trending 库里按向量相似度找(库越大越准)。
- **实时语义**:一句话 → LLM 提取多组关键词 → 每组实时搜 GitHub top N → bge-reranker 重排 → 返回最相关的(覆盖全 GitHub、实时)。需配 `llm_api_base` + `reranker_api_base`,key 放 `.env`(`LLM_API_KEY`、`RERANKER_API_KEY`)。**强烈建议配 `GITHUB_TOKEN`**(每次会发多个搜索请求)。
- **GitHub 关键词**:把输入直接当查询词调 GitHub Search API。

### 语义搜索的 embedding 后端
- **在线服务(推荐)**:在 `config.yaml` 设 `embedding_api_base`(OpenAI 兼容,如自建 vLLM bge-m3),
  key 放 `.env` 的 `EMBEDDING_API_KEY`。本机不下载模型,查询即时。
- **本地模型**:把 `embedding_api_base` 留空 `""`,首次搜索会下载 `embedding_model`(约 2.3GB)。

> 注意:Actions 云端采集时用的是本地 `embedding_model`(够不着内网在线服务),
> 但只要在线服务与之同为 bge-m3,产出的向量一致,检索结果不受影响。

## 配置
见 `config.yaml`(推送数量、语言过滤、embedding 模型与在线服务地址等)。

# GitHub 热门项目监控 + 语义搜索 — 设计文档

- 日期:2026-06-02
- 状态:已与用户确认设计,准备进入实现计划

## 1. 目标

一个个人用的 GitHub 热门项目监控应用,满足两个需求:

1. **每日推送**:每天定时把当天热门的 GitHub 项目通过 Discord 推送给用户查看。
2. **语义搜索**:用户在本地网页输入一句自然语言(可中文),从累积的项目库里找出最相关的项目;同时支持直接调 GitHub Search API 做全站关键词搜索。

## 2. 整体架构

系统分两部分,通过一个 git 仓库解耦(仓库既是代码也是数据存储):

### 2.1 采集端 — GitHub Actions(云端定时)

每天定时运行,流程:

1. 抓取 `github.com/trending` 日榜(主榜单)。
2. 对榜单中每个项目调用 GitHub Search/Repos API 补充元数据(stars、topics、language、README 摘要)。
3. 用本地 BGE 模型给每个项目算 embedding(输入文本 = 名称 + 描述 + topics + README 摘要)。
4. 追加/更新到数据文件,按 `owner/repo` 去重:已存在的只更新 star 数与上榜记录,不重复添加。
5. 发送 Discord Webhook(当天 Top N 项目卡片)。
6. `git commit` 把更新后的数据文件写回仓库。

**理由**:重活和定时都在云端,不依赖用户机器开机;数据(含向量)存仓库,作为单一数据源。

### 2.2 搜索端 — 本机网页服务(FastAPI + 单页 HTML)

随用随开,功能:

- **语义搜索**:用户输入查询 → 用与采集端**同一个 BGE 模型**算查询向量 → 在累积库的向量矩阵上算余弦相似度 → 返回 Top K 最相关项目。
- **关键词搜索**:把输入直接传给 GitHub Search API 搜全站(请求走用户本机代理)。
- 结果以**卡片**展示:名称、stars、描述、topics、链接、首次/最近上榜时间。
- 数据来源:启动时(或手动/定时)`git pull` 拉取采集端写回的数据文件。

## 3. 数据存储(纯文件,不用数据库服务)

个人量级(每天约 25 条、去重累积后一年几千条),无需 Milvus。两个文件存在仓库 `data/` 下:

- `data/projects.jsonl` — 每行一个项目的元数据:`full_name`、`description`、`stars`、`language`、`topics`、`readme_excerpt`、`first_seen`(首次上榜日期)、`trending_history`(上榜日期列表)。
- `data/embeddings.npy` — 与 `projects.jsonl` 行顺序对齐的 numpy 向量矩阵。

去重键 = `owner/repo`。两个文件必须保持行对齐(更新时一起维护)。

## 4. 技术栈

- **语言**:Python(与用户 doc_qa 项目一致,BGE/sentence-transformers 生态在 Python)。
- **Embedding**:`sentence-transformers` + 多语言 BGE 模型。描述多为英文、用户查询可能为中文,需多语言模型保证跨语言匹配。默认型号 `BAAI/bge-m3`(多语言强);若 Actions 运行时间/体积敏感,可降级为更轻的多语言 MiniLM(如 `paraphrase-multilingual-MiniLM-L12-v2`)。型号在实现时最终确定并写入配置,采集端与搜索端必须一致。
- **网页**:FastAPI + 一个静态 HTML 单页(查询输入框 + 结果卡片列表),无前端框架。
- **Trending 抓取**:`requests` + HTML 解析(`selectolax` 或 `beautifulsoup4`)。
- **GitHub API**:用于元数据补充和关键词搜索,带 token 提高速率限制。
- **推送**:Discord Webhook(不做 Discord Bot)。

## 5. 配置项(默认值,均可改)

| 配置 | 默认 | 说明 |
|------|------|------|
| 推送时间 | 每天 09:00(北京时间 = UTC 01:00) | Actions cron 用 UTC |
| 每日推送数量 | Top 10 | Discord 卡片数量 |
| 语言过滤 | 不过滤(全部) | 可配置为只看指定语言(如 Python/Rust/TS) |
| 语义搜索返回 | Top 10 | 网页结果数量 |
| Trending 周期 | 日榜 | 可扩展周榜 |

配置集中在一个文件(如 `config.yaml` 或环境变量),采集端与搜索端共享。

## 6. 用户一次性准备

1. 新建一个 GitHub 仓库存放本项目(Actions 需要)。
2. 在目标 Discord 频道创建 Webhook,把 URL 存为仓库 Actions Secret(如 `DISCORD_WEBHOOK_URL`)。
3. (推荐)创建一个 GitHub Personal Access Token,存为 Actions Secret 用于提高 API 速率;本机搜索端也需要一个 token 做关键词搜索。
4. 本机首次 `pip install` 依赖并下载 BGE 模型。

## 7. 组件划分(单一职责)

| 组件 | 职责 | 输入 | 输出 | 依赖 |
|------|------|------|------|------|
| `trending_fetcher` | 抓 trending 页并解析出仓库列表 | 日期/语言/周期 | repo 标识列表 | requests + 解析库 |
| `github_enricher` | 调 GitHub API 补元数据 | repo 列表 | 元数据字典列表 | GitHub API + token |
| `embedder` | 文本 → 向量(采集端与搜索端共用) | 文本 | 向量 | sentence-transformers |
| `store` | 读写 jsonl + npy、去重、行对齐维护 | 元数据 + 向量 | 持久化文件 | numpy |
| `discord_notifier` | 组装并发送 Discord 卡片 | Top N 项目 | webhook 调用 | Discord Webhook |
| `collect`(采集主流程) | 串起上面采集端各步 | 配置 | 更新数据 + 推送 | 以上组件 |
| `web`(FastAPI 应用) | 语义搜索 + 关键词搜索 + 卡片页面 | 查询 | HTML/JSON | embedder + store + GitHub API |
| `.github/workflows/daily.yml` | 定时触发 collect 并提交数据 | cron | 运行 collect | Actions |

每个组件可独立测试:fetcher/enricher 可对固定 HTML/API 响应做单测;store 的去重与行对齐可单测;embedder 接口稳定;web 的两类搜索可分别测试。

## 8. 数据流

**采集(每日)**:cron → collect → trending_fetcher → github_enricher → embedder → store(去重/对齐写入)→ discord_notifier → git commit。

**搜索(随时)**:浏览器 → web →
- 语义:embedder(查询)→ store 加载向量 → 余弦相似度 → Top K → 卡片;
- 关键词:GitHub Search API → 卡片。

## 9. 错误处理

- Trending 抓取失败 / 页面结构变化:记录错误,采集流程跳过该步但仍尝试已有数据,不让整次运行崩溃。
- GitHub API 速率限制:带 token;命中限制时退避重试,超限则本次少补元数据而非失败。
- Discord 推送失败:重试有限次,失败写日志(Actions 日志可见)。
- 数据文件行不对齐:store 写入前后做行数一致性校验,不一致则报错中止写回,避免污染数据。
- 本机搜索端拉不到数据:提示用户先 `git pull` / 数据为空时给出友好提示。

## 10. 测试策略

- 单元测试:trending_fetcher(对固定 HTML 样本)、github_enricher(mock API 响应)、store(去重 + 行对齐 + 更新已有项目)、discord_notifier(mock webhook)、语义搜索相似度排序(小型固定向量集)。
- 集成测试:collect 端到端(用 mock 的 fetcher/enricher,真实 store + embedder),验证数据文件正确生成与去重。
- web:用 FastAPI TestClient 测两类搜索端点。

## 11. 明确不做(YAGNI)

- 不做 Discord 交互机器人(仅单向 Webhook 推送)。
- 不引入数据库服务(Milvus/Postgres),纯文件足够。
- 不做用户系统/多用户(个人单用户)。
- 不做复杂前端框架(单页 HTML)。
- 不做 LLM 生成式摘要/点评(初版仅相似度检索;后续可加)。

#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
# 载入 .env(含 EMBEDDING_API_KEY、可选 GITHUB_TOKEN),文件已被 .gitignore 忽略
if [ -f .env ]; then set -a; . ./.env; set +a; fi
git pull --rebase 2>/dev/null || true
.venv/bin/python -m ghtrend.web

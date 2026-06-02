#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
git pull --rebase 2>/dev/null || true
.venv/bin/python -m ghtrend.web

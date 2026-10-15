#!/usr/bin/env bash
# Copy code (not data) to lab01 so it can run inside ~/AICUP_2026-1/.venv there.
set -e
cd "$(dirname "$0")/.."
scp -q -r src scripts tests pyproject.toml lab01:~/AICUP_2026-1/
echo synced

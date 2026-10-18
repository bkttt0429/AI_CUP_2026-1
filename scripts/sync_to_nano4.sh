#!/usr/bin/env bash
# Copy code (not data) to NCHC nano4; asks for the password and OTP once. Run from Git Bash:
#   bash scripts/sync_to_nano4.sh                                        # into ~/AICUP_2026-1
#   NANO4_DIR=/path/on/nano4/AICUP_2026-1 bash scripts/sync_to_nano4.sh  # somewhere else
set -e
cd "$(dirname "$0")/.."
scp -q -r src scripts tests viz pyproject.toml requirements.txt "nano4:${NANO4_DIR:-AICUP_2026-1}/"
echo synced

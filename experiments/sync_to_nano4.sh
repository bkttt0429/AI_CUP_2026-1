#!/usr/bin/env bash
# Copy code (not data) to NCHC nano4; asks for the 2FA method and password once. Run from Git Bash:
#   bash scripts/sync_to_nano4.sh                                        # into ~/AICUP_2026-1
#   NANO4_DIR=/path/on/nano4/AICUP_2026-1 bash scripts/sync_to_nano4.sh  # somewhere else
# Files go through the data transfer node, port 2222 of the same host: port 22 (login node)
# has no sftp subsystem, so scp there closes the connection right after 2FA.
set -e
cd "$(dirname "$0")/.."
scp -q -P 2222 -r src scripts tests viz pyproject.toml requirements.txt "nano4:${NANO4_DIR:-AICUP_2026-1}/"
echo synced

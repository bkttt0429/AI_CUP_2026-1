#!/bin/bash
set -euo pipefail
cd /home/u2507115/.cache/rptwt
python3 -I reports/go_report/build.py
PLAYWRIGHT_BROWSERS_PATH=$HOME/.cache/report_shot/browsers $HOME/.cache/report_shot/venv/bin/python -I reports/go_report/shot.py

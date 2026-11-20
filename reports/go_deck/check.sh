#!/usr/bin/env bash
# Run the browser check of the deck on a compute node: sbatch ... scripts/nano4/run.sbatch bash reports/go_deck/check.sh [--only NN]
set -euo pipefail
T=$HOME/.cache/go_deck_tools
export PYTHONPATH=$T/py PLAYWRIGHT_BROWSERS_PATH=$T/browsers
exec python "$(dirname "$0")/deck_check.py" "$@"

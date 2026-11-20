#!/usr/bin/env bash
# M262 part C: test-side outcomes, then the Public fair likelihood (design doc M262).
main() {
  set -euo pipefail
  python -I logs/nano4/m262_outcome.py --test-only
  python -I logs/nano4/m262_like.py --perms 500
}
main "$@"; exit $?

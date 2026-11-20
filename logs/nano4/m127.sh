#!/usr/bin/env bash
# M127: fuse two identifiers keeping the top-20 values and the per-source read-outs, then the label-free comparison of statistics.
set -uo pipefail
main() {
  python -I logs/nano4/m127_conf_family.py --stage fuse --tags SP2e SP4t || return 1
  python -I logs/nano4/m127_conf_family.py --stage report --tags SP2e SP4t
}
main "$@"; exit $?

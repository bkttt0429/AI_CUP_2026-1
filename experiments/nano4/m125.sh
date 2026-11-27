#!/usr/bin/env bash
# M125: fuse three identifiers once more keeping several confidence statistics, then the label-free comparison.
set -uo pipefail
main() {
  python -I logs/nano4/m125_conf_stat.py --stage fuse --tags SP2e SP4 SP4t || return 1
  python -I logs/nano4/m125_conf_stat.py --stage report --tags SP2e SP4 SP4t
}
main "$@"; exit $?

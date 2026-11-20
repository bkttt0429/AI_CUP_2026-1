#!/usr/bin/env bash
# M122 preparation (after the second training job): orientation-averaged queries for seeds 0-2, the identifiers with one, two and
# three new seeds next to the old members, and the label-free shares under m12 / m15 (design doc M124: the rule for more seeds).
set -uo pipefail
main() {
  python -I logs/nano4/m124_tta_queries.py --seeds 0,1,2 --name S4et || return 1
  for t in SP4et SP2y4t1 SP2y4t2 SP2y4et; do python -I logs/nano4/m84_variant.py --stage eval --tag "$t" || return 1; done
  python -I logs/nano4/m127_conf_family.py --stage fuse --tags SP4et SP2y4t1 SP2y4t2 SP2y4et || return 1
  python -I logs/nano4/m127_conf_family.py --stage report --tags SP2y4t1 SP2y4t2 SP2y4et SP4et
}
main "$@"; exit $?

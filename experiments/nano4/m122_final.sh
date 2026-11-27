#!/usr/bin/env bash
# M122 (design doc M122 / M124 / M125): the most complete identifier and its registered test.
#   new seeds 0-3 with orientation-averaged queries (S4ft), with or without the old members (the label-free share under m15
#   decides), banded by m15; then m121 (recipe level against v5 by M37, file level against the ct75u rule) -> ct101 only if both hold.
# Optional first argument: the seeds to use (default 0,1,2,3).
set -uo pipefail
main() {
  local seeds="${1:-0,1,2,3}"
  python -I logs/nano4/m124_tta_queries.py --seeds "$seeds" --name S4ft || return 1
  for t in SP4ft SP2y4ft; do python -I logs/nano4/m84_variant.py --stage eval --tag "$t" || return 1; done
  python -I logs/nano4/m127_conf_family.py --stage fuse --tags SP4ft SP2y4ft || return 1
  python -I logs/nano4/m127_conf_family.py --stage report --tags SP4ft SP2y4ft --write m15 || return 1
  local pick
  pick=$(python -I - <<'PY'
import json
d = json.load(open("results/r_variants/m127_conf_family_SP4ft_SP2y4ft.json"))
m = {t: d[t]["m15"]["mass"] for t in ("SP4ft", "SP2y4ft")}
print(max(m, key=m.get))
PY
) || return 1
  local other=SP4ft; [ "$pick" = SP4ft ] && other=SP2y4ft
  echo "== identifier with the larger label-free share under m15: $pick (the other one, $other, is reported only)"
  python -I logs/nano4/m121_sym_anchor_test.py --id "${pick}~m15" --also "${other}~m15" "$pick" "SP2e~m15" --name ct101 --build
}
main "$@"; exit $?

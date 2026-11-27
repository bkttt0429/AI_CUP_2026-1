#!/usr/bin/env bash
# M161 step 2, secondary reading (design doc M161, registered 2026-11-15 09:45): the same identifiers with fusion weights fitted
# by activity; pool test without building a file.
#   bash logs/nano4/m161_step2w.sh S6at SPA6aw SP6aw P,S2y,S4ft,S4Lt SPA6a
set -uo pipefail
main() {
  local var="$1" new="$2" also="$3" srcs="$4" ctl="$5"
  echo "== $(date '+%F %H:%M') identifiers with activity-weighted fusion: $also, $new"
  python -I logs/nano4/m161_ident_w.py --tag "$also:P,$var" --tag "$new:$srcs,$var" || return 1
  echo "== $(date '+%F %H:%M') label-free recognition shares"
  python -I logs/nano4/m120_diag_mass.py SPAL SPAL~m15 "$ctl" "$ctl~m15" "$also" "$also~m15" "$new" "$new~m15" || echo "(the recognition-share report failed; the test goes on)"
  echo "== $(date '+%F %H:%M') pool test (report only, no file)"
  python -I logs/nano4/m143_pool.py --new "$new" --var "$var" --also "$also" --control "$ctl" --name ct109w --no-build
}
main "$@"; exit $?

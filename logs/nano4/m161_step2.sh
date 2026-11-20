#!/usr/bin/env bash
# M161 step 2 (design doc M161, registered 2026-11-15 09:05): orientation-averaged queries of the by-games identity networks ->
# identifiers -> label-free recognition shares -> the registered pool test against the ct106 rule.
#   bash logs/nano4/m161_step2.sh <seeds> <var> <new tag> <also tag> <candidate name> <sources before the new var>
#   bash logs/nano4/m161_step2.sh 20 S6at SPA6a SP6a ct109 P,S2y,S4ft,S4Lt
set -uo pipefail
main() {
  local seeds="$1" var="$2" new="$3" also="$4" name="$5" srcs="$6"
  echo "== $(date '+%F %H:%M') orientation-averaged queries -> var_$var"
  python -I logs/nano4/m161p_tta.py --seeds "$seeds" --name "$var" || return 1
  echo "== $(date '+%F %H:%M') identifiers $also ($var with P) and $new ($srcs,$var)"
  python -I logs/nano4/m143_ident.py --tag "$also:P,$var" --tag "$new:$srcs,$var" || return 1
  echo "== $(date '+%F %H:%M') label-free recognition shares"
  python -I logs/nano4/m120_diag_mass.py SP2e SP2e~m15 SPAL SPAL~m15 "$also" "$also~m15" "$new" "$new~m15" || echo "(the recognition-share report failed; the test goes on)"
  echo "== $(date '+%F %H:%M') registered pool test"
  python -I logs/nano4/m143_pool.py --new "$new" --var "$var" --also "$also" --control SPAL --name "$name"
}
main "$@"; exit $?

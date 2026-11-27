#!/usr/bin/env bash
# Section 4M M216 step 2 (2026-11-17; registered in the design doc before any output of m216_prod.py existed) - as m200_step2.sh /
# m161r_step2.sh, for the networks that read games up to 384 plies:
#   orientation-averaged queries -> var_S10t (prototypes) and var_S10tc (learned centres);
#   identifier compositions ST1..ST3 (prototype sources only, simplest first; the selection list) and ST4..ST5 (with the
#   learned centres; report only), each with the activity prior; the label-free choice among ST1..ST3 (m164_select.py);
#   the recognised-share report; the paired pool comparison against SQ4 (= ct112's / ct116's identifier) on both seed sets
#   (m165_pool.py, report); then the registered pool test of the chosen identifier (m163_pool.py; candidate file <name>).
#   Whether <name> replaces ct112 is read from the paired comparison: >= +0.001 on both seed sets, plus the ledger
#   back-test against ct112 (not run here).
#   bash logs/nano4/m216_step2.sh 30 S10t ct118
main() {
  local seeds="$1" var="$2" name="$3" c="results/m84_cache" sel=(ST1 ST2 ST3) rep=(ST4 ST5) specs best t also=()
  python -I logs/nano4/m216_tta.py --seeds "$seeds" --name "$var" || return 1
  [ -f "$c/var_S7t.npz" ] && [ -f "$c/var_S6at.npz" ] || { echo "var_S7t / var_S6at missing"; return 1; }
  specs=("ST1:$var" "ST2:S7t,$var" "ST3:S6at,S7t,$var" "ST4:S7t,S7tc,$var,${var}c" "ST5:S6at,S7t,S7tc,$var,${var}c")
  echo "== identifiers: ${specs[*]}"
  python -I logs/nano4/m164_ident.py --prior fit $(printf -- '--tag %s ' "${specs[@]}") || return 1
  python -I logs/nano4/m164_select.py "${sel[@]}" "${rep[@]}" > /dev/null || true   # the table of all five (stderr); report only
  best=$(python -I logs/nano4/m164_select.py "${sel[@]}") || return 1             # the registered choice (last writer of m164_selected.json)
  echo "== chosen identifier (label-free rule, prototype compositions only): $best"
  for t in "${sel[@]}" "${rep[@]}"; do [ "$t" != "$best" ] && also+=("$t"); done
  python -I logs/nano4/m120_diag_mass.py SPALm~m15 SQ4~m15 $(printf -- '%s~m15 ' "${sel[@]}" "${rep[@]}") || echo "(the recognition-share report failed; the test goes on)"
  python -I logs/nano4/m165_pool.py --sets SQ4 "$best" "${also[@]}" --json "m216_pool.json" --dump "m216_dump.npz" || echo "(the paired comparison failed; the test goes on)"
  python -I logs/nano4/m163_pool.py --new "$best" --also "${also[@]}" --control SPALm --name "$name"
}
main "$@"; exit $?

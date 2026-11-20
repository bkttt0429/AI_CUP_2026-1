#!/usr/bin/env bash
# Section 4M M181 step 2 (2026-11-16; registered in the design doc before any output of the third round existed):
#   orientation-averaged queries of the third round -> var_S8t (prototypes) and var_S8tc (learned centres);
#   identifier compositions SR1..SR3 (prototype sources only, simplest first; the selection list) and SR4..SR5 (with the
#   learned centres; report only - M179: the centres carry no reliable gain in cross-validation), each with the activity prior;
#   the label-free choice among SR1..SR3 (m164_select.py); the recognised-share report;
#   the paired pool comparison against SQ4 (= ct112's identifier) on both seed sets (m165_pool.py, report);
#   then the registered pool test of the chosen identifier (m163_pool.py; candidate file <name> on an M37 pass against the
#   ct106 rule). Whether <name> replaces ct112 is read from the paired comparison: >= +0.001 on both seed sets, plus the
#   ledger back-test against ct112 (not run here).
#   bash logs/nano4/m161r_step2.sh 40 S8t ct113
main() {
  local seeds="$1" var="$2" name="$3" c="results/m84_cache" sel=(SR1 SR2 SR3) rep=(SR4 SR5) specs best t also=()
  python -I logs/nano4/m161r_tta.py --seeds "$seeds" --name "$var" || return 1
  [ -f "$c/var_S7t.npz" ] && [ -f "$c/var_S6at.npz" ] || { echo "var_S7t / var_S6at missing"; return 1; }
  specs=("SR1:$var" "SR2:S7t,$var" "SR3:S6at,S7t,$var" "SR4:S7t,S7tc,$var,${var}c" "SR5:S6at,S7t,S7tc,$var,${var}c")
  echo "== identifiers: ${specs[*]}"
  python -I logs/nano4/m164_ident.py --prior fit $(printf -- '--tag %s ' "${specs[@]}") || return 1
  python -I logs/nano4/m164_select.py "${sel[@]}" "${rep[@]}" > /dev/null || true   # the table of all five (stderr); report only
  best=$(python -I logs/nano4/m164_select.py "${sel[@]}") || return 1             # the registered choice (last writer of m164_selected.json)
  echo "== chosen identifier (label-free rule, prototype compositions only): $best"
  for t in "${sel[@]}" "${rep[@]}"; do [ "$t" != "$best" ] && also+=("$t"); done
  python -I logs/nano4/m120_diag_mass.py SPALm~m15 SQ4~m15 $(printf -- '%s~m15 ' "${sel[@]}" "${rep[@]}") || echo "(the recognition-share report failed; the test goes on)"
  python -I logs/nano4/m165_pool.py --sets SQ4 "$best" "${also[@]}" --json "m181_pool.json" --dump "m181_dump.npz" || echo "(the paired comparison failed; the test goes on)"
  python -I logs/nano4/m163_pool.py --new "$best" --also "${also[@]}" --control SPALm --name "$name"
}
main "$@"; exit $?

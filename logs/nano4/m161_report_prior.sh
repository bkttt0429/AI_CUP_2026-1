#!/usr/bin/env bash
# Section 4M M161 (2026-11-15), REPORT ONLY (design doc "M161q 第二步的規格": the a / f production sets with the activity
# prior are reported, no file is built): for one production set's source <var>, the identifiers <tag>1 = block + <var> and
# <tag>5 = block + the old sources + <var>, both with the prior fitted on the held-out own games; the recognised-share report;
# the pool test with the identifier as the new source (look-alike reading unchanged), --no-build.
#   bash logs/nano4/m161_report_prior.sh S6at SQa
main() {
  local var="$1" tag="$2"
  python -I logs/nano4/m164_ident.py --prior fit --tag "${tag}1:$var" --tag "${tag}5:P,S2y,S4ft,S4Lt,$var" || return 1
  python -I logs/nano4/m120_diag_mass.py SPALm~m15 "${tag}1~m15" "${tag}5~m15" || echo "(the recognition-share report failed; the test goes on)"
  python -I logs/nano4/m163_pool.py --new "${tag}1" --also "${tag}5" --control SPALm --no-build --json "m163_pool_${tag}_report.json"
}
main "$@"; exit $?

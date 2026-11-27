#!/usr/bin/env bash
# M143 step 2 (design doc M143 / M143b): orientation-averaged queries of the 40,000-step networks, the identifiers that include
# them, the label-free recognition shares, the registered pool test (and, on a pass, the candidate file).
#   m143_step2.sh <seeds> <variant> <new identifier> <also identifier> <candidate name>
#   m143_step2.sh 7     S4Lt SPAL SP4L ct107     seed 7 alone
#   m143_step2.sh 7,8,9 S4Mt SPAM SP4M ct108     three seeds: the final reading of this line
# Needs GPUs (one per network for the query pass, one for the identifiers). Never touches upload_next.
set -uo pipefail
main() {
  local seeds="$1" var="$2" new="$3" also="$4" name="$5"
  echo "== $(date '+%F %H:%M') step 1 threshold (registered: mean held-out top-1 of the five seed-7 networks >= 0.365)"
  python -I - "$seeds" <<'PY' || { echo "step 1 threshold not met, or a network is missing: step 2 is not run"; return 0; }
import json, sys
ok = True
for s in [int(x) for x in sys.argv[1].split(",")]:
    t = [json.load(open(f"results/m120/train_{v}{f'_s{s}' if s else ''}.json"))["best"] for v in (1, 2, 3, 4, -1)]; m = sum(x["top1"] for x in t) / 5
    print(f"seed {s}: best held-out top-1 of networks 1-4 / all " + " / ".join(f"{x['top1']:.3f} (step {x['step']})" for x in t) + f"; mean {m:.4f}")
    if s == 7:
        ok = m >= 0.365
sys.exit(0 if ok else 1)
PY
  echo "== $(date '+%F %H:%M') orientation-averaged queries -> var_$var"
  python -I logs/nano4/m124_tta_queries.py --seeds "$seeds" --name "$var" || return 1
  echo "== $(date '+%F %H:%M') identifiers $also and $new"
  python -I logs/nano4/m143_ident.py --tag "$also:P,$var" --tag "$new:P,S2y,S4ft,$var" || return 1
  echo "== $(date '+%F %H:%M') label-free recognition shares"
  python -I logs/nano4/m120_diag_mass.py SP2e SP2e~m15 SP2y4ft SP2y4ft~m15 "$also" "$also~m15" "$new" "$new~m15" || echo "(the recognition-share report failed; the test goes on)"
  echo "== $(date '+%F %H:%M') registered pool test"
  python -I logs/nano4/m143_pool.py --new "$new" --var "$var" --also "$also" --control SP2y4ft --name "$name"
}
main "$@"; exit $?

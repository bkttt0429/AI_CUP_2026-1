#!/usr/bin/env bash
# M38: pc seeds 6-29 (192 members, 24 x 8-GPU packs as fast as the QOS allows) -> protocol check -> A (s0-14 vs s0-2), B (s15-29 vs s3-5), curve.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
A=/home/u2507115/.cache/report_audit; TR="python -I scripts/trunk_seq_check.py"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() { local j=""; for i in $(seq 1 600); do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 60; done; return 1; }
echo "== $(date '+%F %T') M38 stage 1: 192 members"
rm -f $A/m38_cmds_*
for s in $(seq 6 29); do for v in 1 2 3 4; do for arm in C T; do
  echo "$TR train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --with-fold0 --extra-games 40 --opp-hsl --player-contrast 0.1 --save-dir results/models/ct_cv_m38 --out results/trunk_seq/es" >> $A/m38_cmds_all.txt
done; done; done
echo "$(wc -l < $A/m38_cmds_all.txt) member commands"
split -l 8 -d -a 2 $A/m38_cmds_all.txt $A/m38_cmds_
P=""; for f in $A/m38_cmds_[0-9]*; do P="$P,$(submit m38p$(basename $f | sed 's/.*_//') "--gres=gpu:8 -c 96 -t 05:00:00" bash $A/pack8.sh $f)"; echo "submitted $(basename $f) $(date '+%T')"; done
echo "pack jobs ${P#,}"; wait_jobs "$P"; for j in ${P//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(sacct -n -X -j $j -o Elapsed | head -1)"; done
ok_all "$P" || echo "a pack failed (the protocol check will name what is missing)"
echo "== $(date '+%F %T') M38 stage 2: protocol check"
C=$(submit m38chk "-c 12 -t 00:20:00" python -I logs/nano4/m38_protocol_check.py); wait_jobs "$C"
grep -v "^==\|^+" logs/nano4/m38chk-$C.out | tail -12
[ "$(state $C)" = "COMPLETED" ] || { echo "protocol check FAILED: comparisons not run"; exit 1; }
echo "== $(date '+%F %T') M38 stage 3: comparisons"
PC="C-pc-f0-x40-oh T-pc-f0-x40-oh"
EA=$(submit m38A "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old $PC --seeds 0 1 2 --new $PC --new-seeds $(seq -s ' ' 0 14) --out results/trunk_seq/es/ens_m38_A.json)
EB=$(submit m38B "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old $PC --seeds 3 4 5 --new $PC --new-seeds $(seq -s ' ' 15 29) --out results/trunk_seq/es/ens_m38_B.json)
EC=""; for k in 6 9 12 15 21 30; do EC="$EC,$(submit m38k$k "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old $PC --seeds 0 1 2 --new $PC --new-seeds $(seq -s ' ' 0 $((k - 1))) --out results/trunk_seq/es/ens_m38_k$k.json)"; done
wait_jobs "$EA,$EB$EC"
python3 -I - <<'PY'
import json
for name in ["A", "B"] + [f"k{k}" for k in (6, 9, 12, 15, 21, 30)]:
    d = json.load(open(f"results/trunk_seq/es/ens_m38_{name}.json"))
    (key, p), = [(k, v) for k, v in d["pairs"].items() if "_new vs" in k]
    pf = p["per_fold_diff"]; ok = p["ci95"][0] > 0 and sum(e > 0 for e in pf) >= 3 and p["diff"] >= 0.0025
    sc = {k: round(v["score"], 4) for k, v in d["ensembles"].items() if k.endswith(("_new", "_old"))}
    print(f"{name:4s} {key}: {p['diff']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + f"  {sc}" + ("  -> PASS (new rule)" if ok else "  -> no"))
PY
echo "== $(date '+%F %T') M38 done"

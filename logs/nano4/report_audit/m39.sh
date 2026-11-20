#!/usr/bin/env bash
# M39 (6 contrastive variants) + M40 (soft histograms in R): stage 1 seeds 0-2 (168 members) -> protocol check -> 6v6 vs pc s0-2 under the
# new rule -> stage 2: the best passing M39 variant and M40 (if it passes) with seeds 3-5 -> protocol check -> 6v6 vs pc s3-5.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
A=/home/u2507115/.cache/report_audit; TR="python -I scripts/trunk_seq_check.py"
REC="--inner-frac 0.1 --with-fold0 --extra-games 40 --opp-hsl --save-dir results/models/ct_cv_m39 --out results/trunk_seq/es"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() { local j=""; for i in $(seq 1 600); do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 60; done; return 1; }
declare -A FLAGS=( [pc0.05]="--player-contrast 0.05" [pc0.2]="--player-contrast 0.2" [pc0.3]="--player-contrast 0.3" [pct0.05]="--player-contrast 0.1 --pc-temp 0.05"
                   [pct0.2]="--player-contrast 0.1 --pc-temp 0.2" [pcv4]="--player-contrast 0.1 --pc-views 4" [rh]="--player-contrast 0.1 --r-extra-npz results/trunk_seq/r_extra_opphist_f0_x40_val{v}.npz" )
tagof() { [ "$1" = rh ] && echo "$2-pc-f0-x40-oh-rh" || echo "$2-$1-f0-x40-oh"; }
run_members() {  # $1 = label, $2 = seeds, rest = variants
  local label=$1 seeds=$2; shift 2; rm -f $A/${label}_cmds_*
  for var in "$@"; do for v in 1 2 3 4; do for arm in C T; do for s in $seeds; do
    echo "$TR train --arm $arm --val-fold $v --seed $s $REC ${FLAGS[$var]}" >> $A/${label}_cmds_all.txt
  done; done; done; done
  echo "$(wc -l < $A/${label}_cmds_all.txt) member commands ($label)"
  split -l 8 -d -a 2 $A/${label}_cmds_all.txt $A/${label}_cmds_
  local P=""; for f in $A/${label}_cmds_[0-9]*; do P="$P,$(submit ${label}p$(basename $f | sed 's/.*_//') "--gres=gpu:8 -c 96 -t 05:00:00" bash $A/pack8.sh $f)"; done
  echo "pack jobs ${P#,}"; wait_jobs "$P"; ok_all "$P" || echo "a pack failed (the protocol check names what is missing)"
}
check_and_compare() {  # $1 = label, $2 = seeds, $3 = base seeds, rest = variants
  local label=$1 seeds=$2 bseeds=$3; shift 3; local tags=""; for var in "$@"; do tags="$tags $(tagof $var C) $(tagof $var T)"; done
  local C=$(submit ${label}chk "-c 12 -t 00:20:00" python -I logs/nano4/m39_protocol_check.py --tags $tags --seeds $seeds); wait_jobs "$C"
  grep -v "^==\|^+" logs/nano4/${label}chk-$C.out | tail -20
  [ "$(state $C)" = "COMPLETED" ] || { echo "protocol check FAILED: comparisons not run"; return 1; }
  local E=""; for var in "$@"; do E="$E,$(submit ${label}e$var "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C-pc-f0-x40-oh T-pc-f0-x40-oh --seeds $bseeds --new $(tagof $var C) $(tagof $var T) --new-seeds $seeds --out results/trunk_seq/es/ens_${label}_$var.json)"; done
  wait_jobs "$E"
}
echo "== $(date '+%F %T') M39/M40 stage 0: soft-histogram columns"
wait_jobs "$(cat $A/m40feat.jobs)"; ok_all "$(cat $A/m40feat.jobs)" || { echo "M40 features failed: stop"; exit 1; }
ls -la results/trunk_seq/r_extra_opphist_f0_x40_val*.npz | awk '{print $5, $9}'
echo "== $(date '+%F %T') stage 1: seeds 0-2"
VARS="pc0.05 pc0.2 pc0.3 pct0.05 pct0.2 pcv4 rh"
run_members m39a "0 1 2" $VARS
check_and_compare m39a "0 1 2" "0 1 2" $VARS || exit 1
python3 -I - <<'PY' > $A/m39_stage1.txt
import json
rows = []
for var in "pc0.05 pc0.2 pc0.3 pct0.05 pct0.2 pcv4 rh".split():
    d = json.load(open(f"results/trunk_seq/es/ens_m39a_{var}.json")); (k, p), = [(k, v) for k, v in d["pairs"].items() if k.startswith("6_new vs")]
    pf = p["per_fold_diff"]; ok = p["ci95"][0] > 0 and sum(e > 0 for e in pf) >= 3 and p["diff"] >= 0.0025
    rows.append((var, p["diff"], p["ci95"], pf, p["by_n"], ok))
    print(f"STAGE1 {var:8s} {p['diff']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + " by n " + " ".join(f"{m}: {x:+.4f}" for m, x in p["by_n"].items()) + ("  -> passes stage 1" if ok else ""))
best = max([r for r in rows if r[0] != "rh" and r[5]], key=lambda r: r[1], default=None)
print("SELECT " + " ".join(([best[0]] if best else []) + (["rh"] if any(r[0] == "rh" and r[5] for r in rows) else [])))
PY
cat $A/m39_stage1.txt
SEL=$(grep '^SELECT' $A/m39_stage1.txt | cut -d' ' -f2-)
if [ -z "$SEL" ]; then echo "== $(date '+%F %T') no variant passes stage 1: stop (pre-registered)"; echo "== M39 done"; exit 0; fi
echo "== $(date '+%F %T') stage 2 (fresh seeds 3-5): $SEL"
run_members m39b "3 4 5" $SEL
check_and_compare m39b "3 4 5" "3 4 5" $SEL || exit 1
for var in $SEL; do python3 -I -c "
import json; d = json.load(open('results/trunk_seq/es/ens_m39b_$var.json')); (k, p), = [(k, v) for k, v in d['pairs'].items() if k.startswith('6_new vs')]
pf = p['per_fold_diff']; ok = p['ci95'][0] > 0 and sum(e > 0 for e in pf) >= 3 and p['diff'] >= 0.0025
print(f'STAGE2 $var {p[\"diff\"]:+.4f} [{p[\"ci95\"][0]:+.4f}, {p[\"ci95\"][1]:+.4f}] folds ' + ' '.join(f'{e:+.4f}' for e in pf) + ('  -> ADOPT (new rule, both stages)' if ok else '  -> no'))"; done
echo "== $(date '+%F %T') M39 done"

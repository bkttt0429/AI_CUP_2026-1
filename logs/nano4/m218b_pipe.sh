#!/usr/bin/env bash
# M218b pipeline (2026-11-18 12:2x; registered 01:29): primary test on the seeds (30, 31); the two replications (30, 32) and
# (31, 32) run only when the primary's chosen composition is >= +0.0010 against SQ4 on both seed sets.
#   bash logs/nano4/m218b_pipe.sh
main() {
  local D=results/m218/g25_ckpt J=${SLURM_JOB_ID:-local} n=0 f
  mkdir -p $D || return 1
  for f in /work/u2507115/m218_ckpt/g25/ckpt_*_s30.pt; do [ -e $D/$(basename $f) ] || ln -s $f $D/$(basename $f); done
  until [ "$(ls $D/ckpt_*_s3[012].pt 2>/dev/null | wc -l)" = 15 ]; do n=$((n + 1)); [ $n -gt 150 ] && { echo "the checkpoints of the seeds 31 / 32 did not arrive in $D"; return 1; }; sleep 10; done
  echo "== $(date '+%F %T') 15 checkpoints in $D; primary test (30, 31) on $(hostname)"
  bash logs/nano4/m218_step2b.sh 30,31 S11u SZ ct121 $D 2>&1 | tee logs/nano4/m218b_primary-$J.log
  python3 -I - logs/nano4/m218b_primary-$J.log <<'PY'
import re, sys
t = open(sys.argv[1]).read(); m = re.search(r"chosen identifier \(label-free rule[^)]*\): (\w+)", t)
if not m: print("== no chosen identifier in the primary's output"); sys.exit(2)
best = m.group(1); d = [float(x) for x in re.findall(r"\(%s\) minus \(SQ4\): ([+-]\d+\.\d+)" % best, t)][:2]
print(f"== primary: chosen {best}; paired differences against SQ4: {d}; threshold +0.0010 on both -> {'PASS' if len(d) == 2 and min(d) >= 0.0010 else 'not passed'}")
sys.exit(0 if len(d) == 2 and min(d) >= 0.0010 else 1)
PY
  if [ $? = 0 ]; then
    bash logs/nano4/m218_step2b.sh 30,32 S11v SA - $D 2>&1 | tee logs/nano4/m218b_rep3032-$J.log
    bash logs/nano4/m218_step2b.sh 31,32 S11w SB - $D 2>&1 | tee logs/nano4/m218b_rep3132-$J.log
  else echo "== the replications (30, 32) and (31, 32) are not run"; fi
  echo "== $(date '+%F %T') M218b pipeline finished"
}
main "$@"; exit $?

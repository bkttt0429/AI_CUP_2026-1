#!/usr/bin/env bash
# M219 report-only step 2 for the arm w9p (2026-11-18): copy this node's checkpoints to results/m219/w9p_ckpt, wait for the
# two from the other node, then m219_step2r.sh (no pool test against the ct106 rule, no file).
main() {
  local D=results/m219/w9p_ckpt n=0 f b
  mkdir -p $D || return 1
  for f in /dev/shm/m219/w9p/ckpt_*_s3[01].pt; do b=$(basename $f); [ -e $D/$b ] || { cp $f $D/.$b.part && mv $D/.$b.part $D/$b || return 1; }; done
  until [ "$(ls $D/ckpt_*_s3[01].pt 2>/dev/null | wc -l)" = 10 ]; do n=$((n + 1)); [ $n -gt 150 ] && { echo "the ten checkpoints did not arrive in $D"; return 1; }; sleep 10; done
  echo "== $(date '+%F %T') ten checkpoints in $D ($(du -shL $D | cut -f1)); /home free $(df -BM --output=avail /home/u2507115 | tail -1 | tr -dc 0-9) MB"
  bash logs/nano4/m219_step2r.sh w9p 1 $D S12p
}
main "$@"; exit $?

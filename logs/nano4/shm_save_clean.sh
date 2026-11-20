#!/usr/bin/env bash
# End-of-job hand-over of one node's checkpoints (2026-11-18): copy the files <pattern> of <shm ckpt dir> that are not yet in
# <dest> (when /home has >= <min MB> free), check the weights of every copy, then remove <shm root> from the node.
# With <wait dir> <count> <HHMM> the script first waits until <wait dir> holds <count> checkpoints or the clock reaches HHMM.
#   bash logs/nano4/shm_save_clean.sh <shm ckpt dir> <dest> '<pattern>' <shm root> <min MB> [<wait dir> <count> <HHMM>]
main() {
  local src=${1:?} dst=${2:?} pat=${3:?} root=${4:?} min=${5:?} wd=${6:-} wn=${7:-0} wt=${8:-0000} f b free
  if [ -n "$wd" ]; then until [ "$(ls "$wd"/ckpt_*.pt 2>/dev/null | wc -l)" -ge "$wn" ] || [ "$(date +%H%M)" -ge "$wt" ]; do sleep 20; done; fi
  free=$(df -BM --output=avail /home/u2507115 | tail -1 | tr -dc 0-9); echo "== $(date '+%F %T') $(hostname): /home free $free MB; $(ls $src/$pat 2>/dev/null | wc -l) file(s) match $src/$pat"
  if ls $src/$pat > /dev/null 2>&1; then
    if [ "$free" -ge "$min" ]; then
      mkdir -p "$dst" || return 1
      for f in $src/$pat; do b=$(basename "$f"); [ -e "$dst/$b" ] || { cp "$f" "$dst/.$b.part" && mv "$dst/.$b.part" "$dst/$b" || return 1; }; done
      python -I logs/nano4/ckpt_same.py "$src" "$dst" "$pat" || { echo "== copies NOT verified: $root stays on the node"; return 1; }
    else echo "== NOT copied (/home below $min MB): the checkpoints of $src are given up"; fi
  fi
  bash logs/nano4/shm_clean.sh "$root"
}
main "$@"; exit $?

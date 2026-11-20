#!/usr/bin/env bash
# Section 4M M161q (2026-11-15): the second 8-GPU job of the production set - seven trainers of the registered recipe on the
# first seven GPUs and, on the eighth, one pilot (all-players network, 64 players x 32 games per episode, same recipe otherwise,
# outputs in results/m161q/e64). The job's status is the production trainers'.
main() {
  local gpus first rc=0
  IFS=, read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
  [ "${#gpus[@]}" -ge 8 ] || { echo "8 GPUs expected, got '${CUDA_VISIBLE_DEVICES:-}'"; return 2; }
  first=$(IFS=,; echo "${gpus[*]:0:7}")
  CUDA_VISIBLE_DEVICES="$first" python -I logs/nano4/m161q_prod.py --jobs 4:31 all:31 1:32 2:32 3:32 4:32 all:32 --steps 40000 &
  local p1=$!
  CUDA_VISIBLE_DEVICES="${gpus[7]}" python -I logs/nano4/m161q_prod.py --jobs all:30 --ep 64 32 --out e64 --steps 40000 &
  local p2=$!
  wait "$p1" || rc=1
  wait "$p2" || echo "(the 64 x 32 pilot failed; the production trainers are what counts)"
  return $rc
}
main "$@"; exit $?

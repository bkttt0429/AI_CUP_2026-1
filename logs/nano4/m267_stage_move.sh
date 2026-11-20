#!/usr/bin/env bash
# Closing (2026-11-21): a staging area for moving the rank project off nano4. NOTHING is deleted or modified in the project:
#   - one tar of the many small files (code, records, logs, git history, small results) written to /work
#   - hard links (no extra space) to the large files that must be kept: weights, identifier outputs, caches of the final chain
#   - a sha256 manifest of every staged file
# The large REGENERABLE inputs (per-move feature vectors, score tables), the Python environment, external tools and models
# are NOT staged; LEFT_OUT.txt in the staging area lists them with their sizes.
#   sbatch ... scripts/nano4/run.sbatch bash logs/nano4/m267_stage_move.sh
main() {
  set -euo pipefail
  ROOT=/home/u2507115/BK_AICUP_2026-1; SH=/home/u2507115/AICUP1_MOVE_20261121; SW=/work/u2507115/AICUP1_MOVE_20261121
  [ -e "$SH" ] || [ -e "$SW" ] && { echo "staging directory exists already: $SH or $SW"; exit 1; }
  free=$(df -m --output=avail /work/u2507115 | tail -1); echo "free on /work before: $free MB"
  mkdir -p "$SH/BK_AICUP_2026-1/results/l1b" "$SW/work"
  BIG="m82 m77 m88 m90 m101 m101c m120 m161 m161p m161q m161r m161r_tta m200 m200_tta m218 m219 m221 ct_submission"
  REGEN="trunk_vec trunk_vec_x40 m90_ply1 m90_ply2 scores_hd scores_opp_hd scores_hd_x40 scores_opp_hd_x40"
  cd "$ROOT"
  # ---- 1. the tar of everything small
  ex=(--anchored --exclude=./.venv --exclude=./tools --exclude=./repo --exclude=./models --exclude=./.claude --exclude=./.superpowers --exclude=./data/training --exclude=./data/mock_tests
      --exclude=./reports/backup_20261118/pack1_release_A.tar --exclude='./results/l1b/scores*' --exclude='./results/l1b/*.ckpt' --exclude='./results/l1b/*.trunk.pt')
  for d in $BIG $REGEN; do ex+=(--exclude=./results/$d); done
  ex+=(--no-anchored --exclude=__pycache__ --exclude=.pytest_cache --exclude='*.pyc')
  tar -cf "$SW/aicup1_code_records.tar" "${ex[@]}" -C /home/u2507115 --transform='s#^\./#BK_AICUP_2026-1/#' -C "$ROOT" . 2> "$SW/tar_warnings.txt" || { echo "tar exit $? (see tar_warnings.txt)"; }
  tar -rf "$SW/aicup1_code_records.tar" -C /home/u2507115/.cache --transform='s#^report_audit#BK_AICUP_2026-1/_from_cache/report_audit#' report_audit
  echo "tar: $(du -m "$SW/aicup1_code_records.tar" | cut -f1) MB, $(tar -tf "$SW/aicup1_code_records.tar" | wc -l) entries; warnings: $(wc -l < "$SW/tar_warnings.txt")"
  # ---- 2. hard links on /home: weights and outputs of the final chain and of the recorded experiments
  for d in $BIG; do cp -al "$ROOT/results/$d" "$SH/BK_AICUP_2026-1/results/$d"; done
  cp -al "$ROOT"/results/l1b/*.trunk.pt "$ROOT"/results/l1b/*.ckpt "$SH/BK_AICUP_2026-1/results/l1b/"
  # ---- 3. hard links on /work: the caches behind the project's symlinks and the identity checkpoints
  mkdir -p "$SW/work/aicup1_moved"
  for d in m84_cache m95; do cp -al "/work/u2507115/aicup1_moved/$d" "$SW/work/aicup1_moved/$d"; done
  for d in m161q_ckpt m161p_ckpt m218_ckpt; do cp -al "/work/u2507115/$d" "$SW/work/$d"; done
  # ---- 4. what is left out, with sizes
  {
    echo "Left on nano4, not staged (MB):"
    for d in $REGEN; do printf '%8d  results/%s   (regenerable input: per-move vectors / score tables)\n' "$(du -sm "$ROOT/results/$d" | cut -f1)" "$d"; done
    printf '%8d  results/l1b/scores*   (regenerable: L1b score tables)\n' "$(du -smc "$ROOT"/results/l1b/scores* | tail -1 | cut -f1)"
    printf '%8d  /work/u2507115/aicup1_moved/trunk_vec_x80   (regenerable input, behind results/trunk_vec_x80)\n' "$(du -sm /work/u2507115/aicup1_moved/trunk_vec_x80 | cut -f1)"
    printf '%8d  /work/u2507115/aicup1_moved/m90_ply0   (regenerable input, behind results/m90_ply0)\n' "$(du -sm /work/u2507115/aicup1_moved/m90_ply0 | cut -f1)"
    for d in .venv tools repo models data/training data/mock_tests reports/backup_20261118/pack1_release_A.tar; do printf '%8d  %s\n' "$(du -sm "$ROOT/$d" | cut -f1)" "$d"; done
    printf '%8d  /work/u2507115/m262_ai   (strong-net states of 2026-11-20, regenerable)\n' "$(du -sm /work/u2507115/m262_ai | cut -f1)"
  } > "$SW/LEFT_OUT.txt"
  # ---- 5. manifest
  ( cd "$SH" && find . -type f -print0 | sort -z | xargs -0 -P 12 -n 4 sha256sum ) | sort -k2 > "$SH/MANIFEST_home.sha256"
  ( cd "$SW" && find . -type f ! -name 'MANIFEST_work.sha256' -print0 | sort -z | xargs -0 -P 12 -n 4 sha256sum ) | sort -k2 > "$SW/MANIFEST_work.sha256"
  echo "staged on /home: $(find "$SH" -type f | wc -l) files, $(du -sm "$SH" | cut -f1) MB (hard links); on /work: $(find "$SW" -type f | wc -l) files, $(du -sm "$SW" | cut -f1) MB (of which the tar is real space)"
  echo "free on /work after: $(df -m --output=avail /work/u2507115 | tail -1) MB; free on /home: $(df -m --output=avail /home/u2507115 | tail -1) MB"; cat "$SW/LEFT_OUT.txt"
}
main "$@"; exit $?

#!/usr/bin/env bash
# Closing (2026-11-21), second step of the move staging: the small files (code, records, logs, git history, small results) as HARD
# LINKS under the /home staging directory - no archive, no extra space (the tar of m267_stage_move.sh hit the /work quota and was
# removed). Then the sha256 manifests of both staging roots. Nothing in the project is deleted or modified.
#   sbatch ... scripts/nano4/run.sbatch bash logs/nano4/m267b_stage_small.sh
main() {
  set -euo pipefail
  ROOT=/home/u2507115/BK_AICUP_2026-1; SH=/home/u2507115/AICUP1_MOVE_20261121; SW=/work/u2507115/AICUP1_MOVE_20261121; D="$SH/BK_AICUP_2026-1"
  BIG="m82 m77 m88 m90 m101 m101c m120 m161 m161p m161q m161r m161r_tta m200 m200_tta m218 m219 m221 ct_submission"
  REGEN="trunk_vec trunk_vec_x40 m90_ply1 m90_ply2 scores_hd scores_opp_hd scores_hd_x40 scores_opp_hd_x40"
  ex=(--exclude=/.venv --exclude=/tools --exclude=/repo --exclude=/models --exclude=/.claude --exclude=/.superpowers --exclude=/data/training --exclude=/data/mock_tests
      --exclude=/reports/backup_20261118/pack1_release_A.tar --exclude='/results/l1b/scores*' --exclude='/results/l1b/*.ckpt' --exclude='/results/l1b/*.trunk.pt' --exclude=__pycache__ --exclude=.pytest_cache --exclude='*.pyc')
  for d in $BIG $REGEN; do ex+=(--exclude=/results/$d); done
  before=$(df -m --output=avail /home/u2507115 | tail -1)
  rsync -a --link-dest="$ROOT/" "${ex[@]}" "$ROOT/" "$D/"
  mkdir -p "$D/_from_cache"; cp -al /home/u2507115/.cache/report_audit "$D/_from_cache/report_audit"
  echo "small files staged: $(find "$D" -type f | wc -l) files in the /home staging tree, $(du -sm "$SH" | cut -f1) MB by du (hard links); free on /home before / after: $before / $(df -m --output=avail /home/u2507115 | tail -1) MB"
  echo "files in the staging tree that are NOT hard links (link count 1; these take real space): $(find "$SH" -type f -links 1 | wc -l)"
  ( cd "$SH" && find . -type f ! -name 'MANIFEST_home.sha256' -print0 | sort -z | xargs -0 -P 12 -n 16 sha256sum ) | sort -k2 > "$SH/MANIFEST_home.sha256.tmp"; mv "$SH/MANIFEST_home.sha256.tmp" "$SH/MANIFEST_home.sha256"
  ( cd "$SW" && find . -type f ! -name 'MANIFEST_work.sha256' -print0 | sort -z | xargs -0 -P 12 -n 4 sha256sum ) | sort -k2 > "$SW/MANIFEST_work.sha256.tmp"; mv "$SW/MANIFEST_work.sha256.tmp" "$SW/MANIFEST_work.sha256"
  echo "manifests: home $(wc -l < "$SH/MANIFEST_home.sha256") files, work $(wc -l < "$SW/MANIFEST_work.sha256") files"
  echo "sizes (MB): home staging $(find "$SH" -type f -printf '%s\n' | awk '{s+=$1} END {printf "%.0f", s/1e6}'), work staging $(find "$SW" -type f -printf '%s\n' | awk '{s+=$1} END {printf "%.0f", s/1e6}')"
  echo "largest directories of the home staging (MB):"; du -sm "$D"/* "$D"/results/* 2>/dev/null | sort -rn | head -28
}
main "$@"; exit $?

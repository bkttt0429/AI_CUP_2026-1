#!/usr/bin/env bash
# Backup batch 1 (2026-11-18): release/ and the intermediates of closed experiments (item A of the 11-17 list) into one tar
# (no compression: arrays and weights do not compress) with a file list and sha256. Nothing is moved or deleted.
main() {
  local O=reports/backup_20261118 L n
  L=$O/pack1_files.txt; mkdir -p $O || return 1
  [ "$(df -BM --output=avail /home/u2507115 | tail -1 | tr -dc 0-9)" -ge 5500 ] || { echo "/home below 5.5 GB: not packed"; return 3; }
  { find release -type f; find results/m200_tta results/m161r_tta -type f; ls results/m200/out_*.npz results/m161r/out_*.npz; find results/m161 results/m161q -mindepth 2 -type f -name '*.npz'; } | sort > $L
  n=$(wc -l < $L); echo "== $(date '+%F %T') $n files, $(tr '\n' '\0' < $L | du -ch --files0-from=- | tail -1 | cut -f1) on $(hostname)"
  tar -cf $O/pack1_release_A.tar.part -T $L && mv $O/pack1_release_A.tar.part $O/pack1_release_A.tar || return 1
  [ "$(tar -tf $O/pack1_release_A.tar | wc -l)" = "$n" ] || { echo "the archive does not list $n files"; return 1; }
  tar -xOf $O/pack1_release_A.tar | sha256sum | cut -d' ' -f1 > /dev/shm/pack1_a.$$; tr '\n' '\0' < $L | xargs -0 cat | sha256sum | cut -d' ' -f1 > /dev/shm/pack1_b.$$
  cmp -s /dev/shm/pack1_a.$$ /dev/shm/pack1_b.$$ && echo "== the archive's contents equal the files on disk (stream sha256 $(cat /dev/shm/pack1_a.$$ | cut -c1-16))" || { echo "CONTENT MISMATCH"; rm -f /dev/shm/pack1_[ab].$$; return 1; }
  rm -f /dev/shm/pack1_[ab].$$; (cd $O && sha256sum pack1_release_A.tar > pack1_release_A.tar.sha256); cat $O/pack1_release_A.tar.sha256; ls -la $O | cut -c24-120
  echo "== $(date '+%F %T') /home free $(df -BM --output=avail /home/u2507115 | tail -1 | tr -dc 0-9) MB"
}
main "$@"; exit $?

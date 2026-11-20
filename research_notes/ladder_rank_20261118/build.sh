#!/usr/bin/env bash
set -euo pipefail
research_dir=$(cd -- "$(dirname -- "$0")" && pwd)
katago_src="$research_dir/../../repo/KataGo/cpp"
objects=(game/board.cpp core/rand.cpp core/rand_helpers.cpp core/hash.cpp core/sha2.cpp core/md5.cpp core/global.cpp)
sources=()
for object in "${objects[@]}"; do
  sources+=("$katago_src/$object")
done
g++ -std=c++17 -O2 -ffunction-sections -fdata-sections -Wl,--gc-sections -pthread \
  -I "$katago_src" "$research_dir/read_ladders.cpp" "${sources[@]}" -o "${1:?output path required}"

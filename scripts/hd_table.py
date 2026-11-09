#!/usr/bin/env python
"""Write the HD table of the training set (issue #42): game_key, player_id, rank, color, hd for every game of data/training
(dataset v1.1; hd empty for a v1.0 record). Read once by scripts/hd_check.py (the HD features) and, filtered to hd >= 2, by the
--game-keys re-scoring of scripts/score_folds.py / scripts/l1b_score.py. CPU, a few minutes.

  python -I scripts/hd_table.py --out results/data/hd_table.csv --keys-out results/data/hd2_keys.csv
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.handicap import hd_summary, hd_table  # noqa: E402
from aicup.scoring import TRAIN_DIR  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-dir", default=str(TRAIN_DIR))
    ap.add_argument("--out", required=True, help="the HD table (CSV)")
    ap.add_argument("--keys-out", default=None, help="also write the game keys with HD >= --min-hd (a CSV with game_key, color, hd) for --game-keys")
    ap.add_argument("--min-hd", type=int, default=2)
    a = ap.parse_args(argv)
    t0 = time.time()
    try:
        t = hd_table(a.train_dir)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    s = hd_summary(t)
    print(f"{s['games']} games; HD {s['hd']}; share HD>=1 {s['share_hd1']:.4f}, HD>=2 {s['share_hd2']:.4f} ({time.time() - t0:.0f} s)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, index=False)
    print(f"wrote {a.out}")
    if a.keys_out:
        sel = t[t["hd"] >= a.min_hd][["game_key", "color", "hd"]]
        Path(a.keys_out).parent.mkdir(parents=True, exist_ok=True)
        sel.to_csv(a.keys_out, index=False)
        print(f"wrote {a.keys_out}: {len(sel)} games with HD >= {a.min_hd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""8-symmetry test-time augmentation, data side: rewrite every SGF coordinate of a test CSV under board symmetry k (0-7) so the
unchanged pipeline (shards -> trunk extract -> predict) scores the transformed games; averaging the 8 probability files is the TTA.

  python -I logs/nano4/sgf_symmetry.py --inputs data/mock_tests/fold1_100/rank_prediction_test_public.csv ... --k 3 --out-dir data/mock_tests/fold1_100_sym3
"""
import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

COORD = re.compile(r"(;?[BW]\[|AB\[|AW\[|AE\[)([a-s])([a-s])(\])")
SYMS = [lambda x, y: (x, y), lambda x, y: (18 - x, y), lambda x, y: (x, 18 - y), lambda x, y: (18 - x, 18 - y),
        lambda x, y: (y, x), lambda x, y: (18 - y, x), lambda x, y: (y, 18 - x), lambda x, y: (18 - y, 18 - x)]


def transform_sgf(sgf, k):
    s = SYMS[k]

    def sub(m):
        x, y = ord(m.group(2)) - 97, ord(m.group(3)) - 97
        nx, ny = s(x, y)
        return f"{m.group(1)}{chr(97 + nx)}{chr(97 + ny)}{m.group(4)}"
    return COORD.sub(sub, sgf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--copy", nargs="*", default=[], help="other files to copy unchanged (e.g. answers.csv)")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in a.inputs:
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        cols = [c for c in df.columns if c.startswith("sgf_")]
        n = 0
        for c in cols:
            df[c] = [transform_sgf(v, a.k) if v else v for v in df[c]]
            n += int((df[c] != "").sum())
        df.to_csv(out / Path(p).name, index=False)
        print(f"{Path(p).name}: {len(df)} questions, {n} games transformed under symmetry {a.k} -> {out}")
    for p in a.copy:
        shutil.copy(p, out / Path(p).name)


if __name__ == "__main__":
    main()

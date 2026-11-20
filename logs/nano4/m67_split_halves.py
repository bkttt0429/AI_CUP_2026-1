"""Section 4M M67 (2026-11-10; user: 合成一個 public 高且 private 也高的版本). ct36 = Public half of ct35 (ct13 + 4D->5D + B->C, Public 0.7874880
expected) + Private half of ct28 (ct13 + 4D->5D, the pre-registered final). Private predictions are byte-identical to ct28's.
  python -I logs/nano4/m67_split_halves.py
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.submission import write_submission  # noqa: E402

a = pd.read_csv(ROOT / "submission_rank_ct35.csv", dtype=str).set_index("question_id").pred_rank
b = pd.read_csv(ROOT / "submission_rank_ct28.csv", dtype=str).set_index("question_id").pred_rank
assert list(a.index) == list(b.index)
out = b.copy(); pub = out.index.str.startswith("pub_"); out[pub] = a[pub]
write_submission(list(out.items()), ROOT / "submission_rank_ct36.csv")
c = pd.read_csv(ROOT / "submission_rank_ct36.csv", dtype=str).set_index("question_id").pred_rank
for ref, s in (("ct28", b), ("ct35", a)):
    ch = c[c != s.reindex(c.index)]
    print(f"ct36 vs {ref}: public {int(ch.index.str.startswith('pub_').sum())}, private {int(ch.index.str.startswith('priv_').sum())} changed; moves {pd.Series([f'{s[i]}->{c[i]}' for i in ch.index]).value_counts().to_dict()}")
print("private half identical to ct28:", bool((c[~pub] == b[~pub]).all()), "| rows", len(c), "| md5", hashlib.md5((ROOT / "submission_rank_ct36.csv").read_bytes()).hexdigest()[:12])

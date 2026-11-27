"""Per-run inner early-stopping diagnostics of the ladder CV set rungs (for GitHub issue #2).

Writes logs/nano4/inner_table.csv (one row per label x seed x split) and logs/nano4/inner_table.md
(per-label summary + the long table). Run from the project root through Slurm.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from aicup.ladder import INNER_KEYS, load_cv  # noqa: E402

OUT = Path("logs/nano4")
rows, first = [], None
for path in sorted(Path("results/ladder/cv").glob("*_s[0-9].npz")):
    if path.name.startswith("base_"):
        continue
    _, meta = load_cv(path)
    if meta.get("best_epoch") is None:  # static rungs have no inner early stopping
        continue
    if first is None:
        first = {k: meta.get(k) for k in INNER_KEYS}
        print("structure of the first set-rung file", path.name, json.dumps(first, default=str)[:600], flush=True)
    label = meta["rung"] + ("" if meta["counts"] else "@nocounts")
    for split in meta["best_epoch"]:
        rows.append({"label": label, "seed": int(meta["seed"]), "val_fold": split,
                     **{k: meta[k][split] for k in INNER_KEYS}})

df = pd.DataFrame(rows).sort_values(["label", "seed", "val_fold"]).reset_index(drop=True)
df["ce_gain"] = df["inner_ce_init"] - df["inner_ce_best"]
df.to_csv(OUT / "inner_table.csv", index=False)

g = df.groupby("label", sort=False)
summ = pd.DataFrame({
    "runs": g.size(),
    "best_epoch median": g["best_epoch"].median(),
    "mean": g["best_epoch"].mean(),
    "min": g["best_epoch"].min(),
    "max": g["best_epoch"].max(),
    "=0": g["best_epoch"].apply(lambda s: int((s == 0).sum())),
    ">1.0": g["best_epoch"].apply(lambda s: int((s > 1.0).sum())),
    "stopped_early": g["stopped_early"].apply(lambda s: int(np.sum(s.astype(bool)))),
    "inner_ce_init": g["inner_ce_init"].mean(),
    "inner_ce_best": g["inner_ce_best"].mean(),
    "ce_gain": g["ce_gain"].mean(),
})
order = sorted(summ.index, key=lambda s: (s.endswith("@nocounts"), s))
summ = summ.loc[order]


def md(frame, floatfmt="{:.4f}"):
    cols = [frame.index.name or ""] + list(frame.columns) if frame.index.name or not isinstance(frame.index, pd.RangeIndex) else list(frame.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + " --- |" * len(cols)]
    for idx, r in frame.iterrows():
        vals = [] if isinstance(frame.index, pd.RangeIndex) else [f"`{idx}`"]
        for v in r.values:
            vals.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


text = ["## per label (4 val folds x 3 seeds)", "", md(summ), "", "## all runs", "",
        md(df.assign(stopped_early=df["stopped_early"].astype(bool)))]
(OUT / "inner_table.md").write_text("\n".join(text) + "\n")
print(summ.to_string(), flush=True)
print("rows", len(df), "labels", summ.shape[0], flush=True)

import json, numpy as np
reps = [json.load(open(f"results/r_variants/importance_val{v}.json")) for v in (1, 2, 3, 4)]
print("base scores", [round(r["base"], 4) for r in reps])
rows = []
for kind in ("token_groups", "r_blocks", "r_columns"):
    for k in reps[0][kind]:
        vals = [r[kind][k] for r in reps]; rows.append((float(np.mean(vals)), kind, k, vals))
for kind in ("token_groups", "r_blocks"):
    print(f"== {kind}")
    for mval, kd, k, vals in sorted([r for r in rows if r[1] == kind], reverse=True):
        print(f"  {k:24s} {mval:+.4f}  folds " + " ".join(f"{x:+.4f}" for x in vals))
print("== top 15 single items (token groups + R columns)")
for mval, kd, k, vals in sorted([r for r in rows if r[1] != "r_blocks"], reverse=True)[:15]:
    print(f"  {k:24s} {mval:+.4f}  ({kd})  folds " + " ".join(f"{x:+.4f}" for x in vals))

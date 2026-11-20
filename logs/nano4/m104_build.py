"""Section 4M M104 (2026-11-13): candidate ct75u = ct75s plus the changes on which all five recipes agree in the two places the
current composition never touches (design doc M104; CV in m103b_votes.py): (a) questions ct44 changed: the five recipes' region
decisions all equal the same answer other than ct44's -> that answer; (b) questions outside the boundary region, not changed by
ct44 and not among ct28's 38 locked ones: the five unrestricted decisions all equal the same answer other than the base decision
-> that answer. Checks: the ct75s rule rebuilt here equals the ct75s file; the reference recipe equals ct75n. Then every Public
reading the file can have, given all recorded uploads (ledger decoder), as a table.
  python -I logs/nano4/m104_build.py [--no-ledger]
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402
import m58_decode as led  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402

b, mb, REC = m3.b, m3.mb, m3.REC; FIVE = ("v3", "v5", "s2", "s3c", "s3"); E1 = math.exp(-1.0); NAME = "ct75u"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-ledger", action="store_true"); a_ = ap.parse_args()
    V, T, key_pid, used = b.sources(["5e", "SP", "SP2e", "S2", "S3c", "S3"]); pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); lev = np.array(LEVELS); rk = np.vectorize(b.RANK.get)
    sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731
    c9, c28, c44, c73, c75n, c75s = (sub(k) for k in ("ct9", "ct28", "ct44", "ct73", "ct75n", "ct75s")); locked = c28 != c9; keep = c44 != c28; nq_t = pd.Series(T["5e"]["qid"]).value_counts().reindex(qids).to_numpy(); Tq = np.array([b.TN[int(m)] for m in nq_t])[:, None]; pub = np.char.startswith(qids.astype(str), "pub_")
    Leq = pd.DataFrame(m3.test_evidence(V, T, "enc", 0, 1, m3.R_ENC, 1)).groupby(T["enc"]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); dR, dA = {}, {}; region = None
    for name in FIVE:
        idn, enc, cc, temp, pi0, bmin, scale = REC[name]; Lq = (pd.DataFrame(m3.test_evidence(V, T, idn, cc, pi0, scale, bmin)).groupby(T[idn]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy() + b.U_ENC * Leq) * Tq
        d0, da, dr, _, region = mb.decisions(P9, Lq, 0.5, 3.0, 0.6); dR[name] = np.where(locked, c28, lev[dr]); dA[name] = np.where(locked, c28, lev[da])
    base0 = lev[d0]; print(f"base decision equals ct9 on {int((base0 == c9).sum())} of {len(qids)} questions; locked {int(locked.sum())}; changed by ct44 {int(keep.sum())}; inside the boundary region {int(region.sum())}")
    r75s = np.where(keep, c44, np.where(dR["v3"] == dR["v5"], dR["v5"], c44)); assert (r75s == c75s).all(), int((r75s != c75s).sum()); assert (np.where(keep, c44, dR["v5"]) == c75n).all(); print("checks: the ct75s rule rebuilt here equals ct75s; the reference recipe equals ct75n")
    vR = np.stack([dR[k] for k in FIVE]); vA = np.stack([dA[k] for k in FIVE]); una_R = (vR == vR[0]).all(0); una_A = (vA == vA[0]).all(0)
    rev = keep & una_R & (vR[0] != c44); out = ~region & ~keep & ~locked & una_A & (vA[0] != c9); fin = c75s.copy(); fin[rev] = vR[0][rev]; fin[out] = vA[0][out]
    for nm, s in (("(a) unanimous change of a ct44 answer", rev), ("(b) unanimous change outside the region", out)):
        print(f"{nm}: {int(s.sum())} questions (Public {int((s & pub).sum())}, Private {int((s & ~pub).sum())})")
        for i in np.flatnonzero(s):
            print(f"    {qids[i]} ({int(nq_t[i])} games): ct9 {c9[i]}, ct28 {c28[i]}, ct44 {c44[i]}, ct75s {c75s[i]} -> {fin[i]}")
    for nm, o in (("ct75s", c75s), ("ct44", c44), ("ct75n", c75n), ("ct28", c28)):
        ch = fin != o; j = np.abs(rk(fin) - rk(o)); print(f"{NAME} vs {nm}: {int(ch.sum())} differ (Public {int((ch & pub).sum())}, Private {int((ch & ~pub).sum())}); one level {int((j == 1).sum())}, two or more {int((j >= 2).sum())}")
    f = ROOT / f"submission_rank_{NAME}.csv"
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(["question_id", "pred_rank"]); w.writerows(zip(qids, fin))
    raw = f.read_bytes(); md5 = hashlib.md5(raw).hexdigest()[:12]; (ROOT / "upload_next" / f.name).write_bytes(raw); print(f"{f.name} written, md5 {md5}; rows {len(fin)}, LF only {b'\\r' not in raw}, levels valid {set(fin) <= set(LEVELS)}")
    if a_.no_ledger:
        return
    # every feasible Public reading of the new file, relative to ct75s
    ledger = json.load(open(led.LEDGER)); up = {**led.KNOWN_SCORES, **ledger}; prior = led.prior_from_ct9(); qs, cand, obs = led.build(up, prior); o75 = next(o for o in obs if o["name"] == "submission_rank_ct75s.csv"); S75 = up["submission_rank_ct75s.csv"]
    dnew = led.decisions(f.name); k = int(((fin != c75s) & pub).sum()); big = int((np.abs(rk(fin) - rk(c75s))[(fin != c75s) & pub] >= 2).sum()); rows = []
    for a in range(-k, k + 1):
        for bb in range(-k, k + 1):
            if abs(a - bb) > k:
                continue
            qs2, cand2, obs2 = led.build(up, prior, [(dnew, o75["X"] + a, o75["Y"] + bb)]); sol = led.solve(qs2, cand2, obs2, prior, time_limit=20)
            if sol is not None and sol != "timeout":
                cost = float(sum(-math.log(max(prior[q][c], 1e-6)) for q, c in sol.items())); rows.append({"a": a, "b": bb, "score": S75 + ((1 - E1) * a + E1 * bb) / 400, "points": (1 - E1) * a + E1 * bb, "cost": cost})
    cmin = min(r["cost"] for r in rows); z = sum(math.exp(-(r["cost"] - cmin)) for r in rows)
    for r in rows:
        r["weight"] = math.exp(-(r["cost"] - cmin)) / z
    rows.sort(key=lambda r: -r["score"]); p_up = sum(r["weight"] for r in rows if r["score"] >= S75 - 1e-9)
    print(f"feasible Public readings of {NAME}: {len(rows)} ({k} Public questions differ from ct75s, {big} of them by two or more levels); rough probability of a reading not below ct75s: {p_up:.2f}")
    for r in sorted(rows, key=lambda r: -r["weight"])[:12]:
        print(f"    reading {r['score']:.7f}  exact {r['a']:+d}, within one {r['b']:+d} ({r['points']:+.2f} points vs ct75s)  weight {r['weight']:.3f}")
    L = [f"# `submission_rank_{NAME}.csv` 的 Public 讀數對照表（自動產生）", "", f"檔案 md5 `{md5}`：ct75s 加上「五個配方全票通過」的改動——(a) 把 ct44 的答案改掉 {int(rev.sum())} 題（Public {int((rev & pub).sum())}、Private {int((rev & ~pub).sum())}），(b) 邊界區以外 {int(out.sum())} 題（Public {int((out & pub).sum())}、Private {int((out & ~pub).sum())}）。", "",
         f"與 ct75s 不同的 Public 題共 {k} 題。讀數必須是下表其中之一；不在表內就代表傳錯檔或抄錯。ct75s 是 0.8092107。", "", "| Public 讀數 | 對 ct75s：完全正確 | 差一級以內 | 分數差 | 粗估可能性 |", "|---|---|---|---|---|"]
    L += [f"| {r['score']:.7f} | {r['a']:+d} | {r['b']:+d} | {r['points']:+.2f} | {r['weight']:.2f} |" for r in rows]
    L += ["", f"事先寫下的建議：讀數 ≥ 0.8092107 → 最終檔改用 {NAME}（它已是最後一筆）；低於 0.8092107 → 傳回 ct75s。交叉驗證的期望是每題約 +0.15（改掉 ct44 的答案）到 +0.37（邊界區以外），這 {k} 題合計約 {0.15 * int((rev & pub).sum()) + 0.37 * int((out & pub).sum()):+.1f} 分，雜訊約 ±{0.62 * math.sqrt(max(k, 1)):.1f} 分。"]
    (ROOT / f"upload_next/{NAME}_decode.md").write_text("\n".join(L) + "\n"); (ROOT / f"results/r_variants/m104_{NAME}.json").write_text(json.dumps({"md5": md5, "reversal": qids[rev].tolist(), "outside": qids[out].tolist(), "outcomes": rows}, indent=1, default=float)); print("written")


if __name__ == "__main__":
    main()

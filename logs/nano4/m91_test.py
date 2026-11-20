"""Section 4M M91, test side (2026-11-13): mirror evidence for the 800 test questions. Candidates = the questions (prototype = sum of
the own-side vectors of a question's games); queries = the opponent side of all 1,000,000 training rows, every row under ONE network:
rows whose owner is in fold u (1-4) use network (u mod 4) + 1 (a trained-on row for that network, as in the simulation), the other
rows are spread over the four networks by the owner's index; null = the own-side rows of the fold that network left out (certainly
not test players). Band edges, a_b, K / P0 and temperatures come from the simulation (m91_<tag>_cal.npz). Safety checks: (1) the
test-side null's share in every band within a factor 2 of nominal; (2) without mirror evidence the builder reproduces ct75 exactly.
Density ratio r = min over the bands >= q99.9 of (test excess share) / (simulation excess share), clipped to [0, 1]; weights x r.
ct84 = ct75's recipe posterior x mirror evidence, same composition (ct28's 38 locked questions, decisions inside the boundary
region); ct84n = the no-reversal version (ct44 where ct44 changed ct28). Files are written only when the simulation passed and both
checks hold; --report-only never writes.
  python -I logs/nano4/m91_test.py [--tag mirror] [--report-only]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m71_step2 as s2  # noqa: E402
import m76_boundary as mb  # noqa: E402
import m83_classcond as m83  # noqa: E402
import m86_bundle as b  # noqa: E402
import m87_pi0 as m87  # noqa: E402
import m91_mirror as mm  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.submission import write_submission  # noqa: E402

RV, EMB, TEST = mm.RV, mm.EMB, ROOT / "results/ct_submission/vec"; NOM, NB, NSEED = mm.NOM, mm.NB, mm.NSEED


def v5_evidence(V, T, qids, nq_t):
    """Question-level log-evidence of bundle v5 on the test (m86_bundle.py's rules; density ratios as it printed them: 0.852 / 0.713)."""
    def one(name, cc, r, floor=1):
        v, tt = V[name], T[name]; ones = np.ones(len(v["y"]), bool); edges, cal, vband, vg = s2.fit(v, ones); cal_e = m87.corrected(v, ones)[1]
        tb = np.where(~tt["target_black"], np.searchsorted(edges[True], tt["opp_m"]), np.searchsorted(edges[False], tt["opp_m"])); tg = s2.group_of(tt["hd"], tt["target_black"]); R = m83.tables(v, ones, cal_e, vband, vg, m83.KAPPA) if cc else None; logE = np.zeros((len(tb), 10))
        for k in (0, 1, 2):
            s = tg == k; aw = (r * cal_e[k]["a"] * (np.arange(len(cal_e[k]["a"])) >= floor))[tb[s]][:, None]; lab = tt["opp_lab"][s]; ratio = R[k][:, lab].T if cc else cal_e[k]["ratio"][lab[:, None] - np.arange(10)[None, :] + 9]; logE[s] = np.log((1 - aw) + aw * ratio)
        return pd.DataFrame(logE).groupby(tt["qid"]).sum().reindex(qids).fillna(0.0).to_numpy()
    Tq = np.array([b.TN[int(m)] for m in nq_t])[:, None]
    return (one("SP2e", 1, 0.852) + b.U_ENC * one("enc", 0, 0.713)) * Tq


def main():
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="mirror"); ap.add_argument("--report-only", action="store_true"); a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    dev = "cuda"; cal = np.load(RV / f"m91_{a_.tag}_cal.npz"); sim = json.loads((RV / f"m91_{a_.tag}.json").read_text()); passed = bool(sim.get("passed_cv")); edges = {g: cal["edges"][g] for g in range(3)}
    calib = {g: {"a": cal["a"][g], "ratio": cal["ratio"][g]} for g in range(3)}; D = mm.load_meta(log); N = len(D["key"])
    tab = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(TEST.glob("games_s*_p*.csv"))]); tab["task"] = tab["task"].astype(int); tab = tab.sort_values("task"); assert np.array_equal(tab["task"].to_numpy(), np.arange(len(tab)))
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); qpos = {q: i for i, q in enumerate(qids)}; tq = np.array([qpos[x] for x in tab["question_id"]])
    qtasks = [torch.as_tensor(np.flatnonzero(tq == i), device=dev) for i in range(len(qids))]; net_of = np.where((D["fold"] >= 1) & (D["fold"] <= 4), D["fold"] % 4 + 1, 1 + D["owner"] % 4)
    band_of = lambda m, g: np.where(g == 0, np.searchsorted(edges[0], m), np.where(g == 1, np.searchsorted(edges[1], m), np.where(g == 2, np.searchsorted(edges[2], m), 0)))  # noqa: E731
    J = np.zeros(N, np.int32); MG = np.full(N, -np.inf, np.float32); cn = np.zeros((3, NB)); NEG = -float("inf")
    for v in (1, 2, 3, 4):
        rq = np.flatnonzero(net_of == v); rn = np.flatnonzero(D["fold"] == v); rq_t, rn_t = torch.as_tensor(rq, device=dev), torch.as_tensor(rn, device=dev); Z = Zn = 0.0; Eown, Eopp = [], []
        for s in NSEED:
            z = np.load(EMB / f"emb_{v}{f'_s{s}' if s else ''}.npz"); Eown.append(torch.as_tensor(z["own"].astype(np.float32), device=dev)); Eopp.append(torch.as_tensor(z["opp"].astype(np.float32), device=dev)); tow = torch.as_tensor(z["t_own"].astype(np.float32), device=dev); del z
            P = torch.stack([torch.nn.functional.normalize(tow[i].sum(0), dim=0) for i in qtasks]); S = Eopp[-1][rq_t] @ P.T; mu, sd = S.mean(0), S.std(0) + 1e-6; Z = Z + (S - mu) / sd; del S; Zn = Zn + (Eown[-1][rn_t] @ P.T - mu) / sd
        best, n_dis = mm.distractors(Eown, Eopp, D, v, rq, dev); dz_q = best(Eopp, rq, True); dz_n = best(Eown, rn, False)

        def stat(Zs, dz):
            t = (Zs / len(NSEED)).topk(2, dim=1); p1, p2 = t.values[:, 0], t.values[:, 1]; m = torch.where(p1 > dz, p1 - torch.maximum(p2, dz), torch.full_like(p1, NEG))
            return t.indices[:, 0].cpu().numpy().astype(np.int32), m.cpu().numpy()
        (J[rq], MG[rq]), (_, mn) = stat(Z, dz_q), stat(Zn, dz_n)
        gn = D["g_own"][rn]; bn = band_of(mn, gn); np.add.at(cn, (gn[gn >= 0], bn[gn >= 0]), 1); log(f"network {v}: {len(rq)} query rows, {len(rn)} null rows, {n_dis} seen training players as competing candidates; a test question wins {float(np.isfinite(MG[rq]).mean()):.4f} of the queries and {float(np.isfinite(mn).mean()):.4f} of the null rows"); del Z, Zn, Eown, Eopp, dz_q, dz_n
    g = D["g_opp"]; band = band_of(MG, g); ok = g >= 0; ct = np.zeros((3, NB)); np.add.at(ct, (g[ok], band[ok]), 1); null_ok = True; rep = {"passed_cv": passed}
    simc = sim["calibration"]
    for k in range(3):
        sh_t, sh_n, sh_v, sh_vn = ct[k] / ct[k].sum(), cn[k] / cn[k].sum(), np.array(simc[str(k)]["share"]), np.array(simc[str(k)]["share_null"]); chk = [(sh_n[x] / sh_vn[x]) for x in range(1, NB) if cn[k].sum() * sh_vn[x] >= 30]; good = all(0.5 <= x <= 2.0 for x in chk); null_ok &= good
        log(f"group {k}: test share by band " + " ".join(f"{x:.5f}" for x in sh_t) + " | simulation " + " ".join(f"{x:.5f}" for x in sh_v) + " | test-side null " + " ".join(f"{x:.5f}" for x in sh_n) + " | simulation null " + " ".join(f"{x:.5f}" for x in sh_vn) + f" -> null check (test null / simulation null within a factor 2 in every band with >= 30 expected rows) {'OK' if good else 'FAILED'}")
        rep[f"group{k}"] = {"share_test": sh_t.tolist(), "share_sim": sh_v.tolist(), "share_null": sh_n.tolist(), "share_null_sim": sh_vn.tolist(), "null_ok": bool(good)}
    sh_t, sh_n, sh_v, sh_vn = ct[0] / ct[0].sum(), cn[0] / cn[0].sum(), np.array(simc["0"]["share"]), np.array(simc["0"]["share_null"]); top = [x for x in range(3, NB) if np.isfinite(edges[0][x - 1]) and sh_v[x] - sh_vn[x] > 1e-6]; ratios = [(sh_t[x] - sh_n[x]) / (sh_v[x] - sh_vn[x]) for x in top]; r = float(np.clip(min(ratios), 0, 1)) if ratios else 0.0
    log(f"true-pair share (share minus the null's share) test / simulation in the HD0 bands >= q99.9: {' '.join(f'{x:.2f}' for x in ratios)} -> r = {r:.3f} (simulation half-pools had about {int(cal['n_cand'].mean())} candidates, the test has {len(qids)})")
    k = ok & (band >= 1); qd, od, bd, (ld, gd) = mm.dedup(J[k].astype(np.int64), D["owner"][k], band[k], [D["lab"][k], g[k]]); LT = np.zeros((len(qids), 10)); np.add.at(LT, qd, mm.evidence(calib, ld, gd, bd, scale=r)); m = np.zeros(len(qids), np.int64); np.add.at(m, qd[bd >= 2], 1); LT = LT * cal["T"][mm.bucket(m)][:, None]
    log(f"de-duplicated matches {len(qd)}; per question at >= q99: mean {m.mean():.1f}, median {np.median(m):.0f}, none {float((m == 0).mean()):.3f}, > 20 {float((m > 20).mean()):.3f}; temperatures {cal['T'].tolist()}")
    sub = lambda name: pd.read_csv(ROOT / f"submission_rank_{name}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731
    c9, c28, c44, c75, c75n = sub("ct9"), sub("ct28"), sub("ct44"), sub("ct75"), sub("ct75n"); lev = np.array(LEVELS); rk = np.vectorize(b.RANK.get); locked = c28 != c9
    cur = rk(c75)[qd]; h0 = gd == 0  # label-free look: how often the matched owner's level agrees with the current answer, by band (HD0)
    log("HD0 matches, owner's level vs ct75's answer by band (same / within one / count): " + " | ".join(f"b{x}: {float((ld[h0 & (bd == x)] == cur[h0 & (bd == x)]).mean()) if (h0 & (bd == x)).any() else float('nan'):.2f} / {float((np.abs(ld[h0 & (bd == x)] - cur[h0 & (bd == x)]) <= 1).mean()) if (h0 & (bd == x)).any() else float('nan'):.2f} / {int((h0 & (bd == x)).sum())}" for x in range(1, NB)))
    V, T, _, used = b.sources(["5e", "SP2e"]); nq_t = pd.Series(T["5e"]["qid"]).value_counts().reindex(qids).to_numpy(); L5 = v5_evidence(V, T, qids, nq_t)
    base = np.where(locked, c28, lev[mb.decisions(P9, L5, 0.5, 3.0, 0.6)[2]]); repro = bool((base == c75).all()); fin = np.where(locked, c28, lev[mb.decisions(P9, L5 + LT, 0.5, 3.0, 0.6)[2]]); finn = np.where(c44 != c28, c44, fin); pub = np.char.startswith(qids.astype(str), "pub_")
    log(f"reproduction of ct75 without mirror evidence: {'exact' if repro else f'FAILED ({int((base != c75).sum())} differ)'}; questions with any mirror evidence {int((np.abs(LT).sum(1) > 0).sum())} of {len(qids)}")
    for name, x in (("ct84 (full)", fin), ("ct84n (no reversal)", finn)):
        for o_name, o in (("ct75", c75), ("ct75n", c75n), ("ct44", c44), ("ct28", c28)):
            ch = x != o; jmp = np.abs(rk(x) - rk(o)); log(f"  {name} vs {o_name}: {int(ch.sum())} differ (public {int((ch & pub).sum())}, private {int((ch & ~pub).sum())}); one level {int((jmp == 1).sum())}, two or more {int((jmp >= 2).sum())}")
    rep.update(r=r, null_ok=bool(null_ok), repro_ct75=repro, matches=int(len(qd)), matches_per_question_mean=float(m.mean()), changed_vs_ct75=int((fin != c75).sum()), changed_n_vs_ct75n=int((finn != c75n).sum()))
    adopt = bool(passed and null_ok and repro and r > 0 and not a_.report_only); rep["adopt"] = adopt
    if adopt:
        for name, x in (("ct84", fin), ("ct84n", finn)):
            f = ROOT / f"submission_rank_{name}.csv"; write_submission(list(zip(qids, x)), f); rep[f"md5_{name}"] = hashlib.md5(f.read_bytes()).hexdigest()[:12]; log(f"{name} written, md5 {rep[f'md5_{name}']}")
    else:
        log(f"no file written (simulation passed: {passed}; null check: {null_ok}; reproduction: {repro}; r: {r:.3f}; report only: {a_.report_only})")
    np.savez(RV / f"m91_{a_.tag}_test.npz", qid=qids, LT=LT, m=m, fin=fin, finn=finn); (RV / f"m91_{a_.tag}_test.json").write_text(json.dumps(rep, indent=1, default=float)); log("written")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""M198 exploratory signal probe: conditional move-transition habits after ply 256.

Report only. No model/checkpoint/cache changes. See report section 11 preregistration.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.sgf import setup_and_moves
import m81_tactic_id as m81
from m188_confusions import std

DIM = 4096
BOUNDS = [128, 192, 256, 288, 320, 384, 512]
WEIGHTS = [0., .125, .25, .5, 1., 2.]


def tokens(sgf, color):
    """D4-invariant distance/edge/turn features, hashed consecutive own moves."""
    _, moves = setup_and_moves(sgf)
    previous = {"B": None, "W": None}; previous_token = 1022
    out = [[], []]
    def distance(a, b):
        if a is None or b is None:
            return 6
        d = max(abs(a[0]-b[0]), abs(a[1]-b[1]))
        return 0 if d == 0 else 1 if d == 1 else 2 if d == 2 else 3 if d <= 4 else 4 if d <= 8 else 5
    for t, (c, x, y) in enumerate(moves[:384]):
        point = None if x is None else (x, y)
        if c == color:
            own, opp = previous[c], previous["W" if c == "B" else "B"]
            if point is None:
                token = 1023
            else:
                edge = min(x, y, 18-x, 18-y, 4)
                angle = 1
                if own is not None and opp is not None:
                    a = (x-own[0], y-own[1]); b = (opp[0]-own[0], opp[1]-own[1])
                    dot = a[0]*b[0] + a[1]*b[1]
                    norm = (a[0]**2+a[1]**2)*(b[0]**2+b[1]**2)
                    if norm and 4*dot*dot > norm:
                        angle = 2 if dot > 0 else 0
                token = ((distance(point, opp)*7 + distance(point, own))*5 + edge)*3 + angle
            value = (previous_token*1009 + token*9176 + 197) % DIM
            if 128 <= t < 384:
                out[int(t >= 256)].append(value)
            previous_token = token
        previous[c] = point
    return [np.asarray(x, np.int16) for x in out], len(moves)


def one(task):
    owner, hidx, sgf, color = task
    arr, length = tokens(sgf, color)
    return owner, hidx, arr, int(np.searchsorted(BOUNDS, length, side="right")), length


def selftest():
    sgf = "(;SZ[19]HD[0]" + ";B[];W[]"*193 + ")"
    arr, length = tokens(sgf, "W")
    assert length == 386 and [len(x) for x in arr] == [64, 64]
    assert len(tokens("(;SZ[19]HD[0])", "B")[0][0]) == 0
    hd, _ = tokens("(;SZ[19]HD[2]" + ";B[];W[]"*193 + ")", "W")
    assert [len(x) for x in hd] == [64, 64]
    # Full sequence reflected/rotated: distances, edge and angle are unchanged.
    pts = [(i % 19, (i*7+i//19) % 19) for i in range(384)]
    def record(transform):
        return "(;SZ[19]HD[0]" + "".join(f";{'B' if i%2 == 0 else 'W'}[{chr(97+transform(x,y)[0])}{chr(97+transform(x,y)[1])}]" for i,(x,y) in enumerate(pts)) + ")"
    base, _ = tokens(record(lambda x,y:(x,y)), "B")
    for tr in [lambda x,y:(18-y,x), lambda x,y:(18-x,y)]:
        alt, _ = tokens(record(tr), "B")
        assert all(np.array_equal(a,b) for a,b in zip(base,alt))
    print("selftest passed: empty, pass, handicap colours, 128/256/384 boundaries, rotation/reflection", flush=True)


def baseline(q, ng):
    Q, H = int(q["Q"]), int(q["H"]); hq = np.arange(2*Q, 2*Q+H)
    cache = ROOT / "results/m84_cache"
    ev = json.loads((ROOT / "results/r_variants/m84_SQ4_eval.json").read_text())
    F = ev["weights"][0] * np.load(cache / "block_held.npy").astype(np.float32)
    for k, name in enumerate(ev["sources"]):
        with np.load(cache / f"var_{name}.npz") as v:
            proto, query = v["protos"], v["qvec"]
            z = np.zeros_like(F)
            for m in range(len(proto)):
                scores = (torch.as_tensor(query[m,hq].astype(np.float32), device="cuda") @
                          torch.as_tensor(proto[m,4].astype(np.float32), device="cuda").T).cpu().numpy()
                z += std(scores, v["has"][m,4] if "has" in v.files else None) / len(proto)
            F += ev["weights"][k+1] * z
    return F + ev["beta"] * np.log(ng)[None].astype(np.float32), ev


def score(counts, query):
    # Training-only Dirichlet background and fixed equivalent sample size 200 moves.
    bg = counts.sum(0).astype(np.float64) + 1.
    bg /= bg.sum()
    probs = (counts + 200.*bg[None]) / (counts.sum(1)[:,None] + 200.)
    llr = np.log(probs / bg[None]).astype(np.float32)
    out = np.empty((len(query), len(counts)), np.float32)
    matrix = torch.as_tensor(llr, device="cuda").T
    for i in range(0, len(query), 256):
        out[i:i+256] = (torch.as_tensor(query[i:i+256], device="cuda") @ matrix).cpu().numpy()
    n = query.sum(1)
    return std(out) * np.sqrt(np.minimum(n/32., 1.))[:,None]


def main(a):
    selftest(); start = time.time()
    q = np.load(ROOT / "results/m84_cache/queries.npz")
    players = q["players"].astype(str); pidx = {p:i for i,p in enumerate(players)}
    Q, H = int(q["Q"]), int(q["H"]); truth = q["held_truth"].astype(int)
    hkeys = q["qkey"][2*Q:2*Q+H].astype(str); hidx = {k:i for i,k in enumerate(hkeys)}
    assert len(hidx) == H
    held = set(json.loads((ROOT / "results/m82/held_keys.json").read_text()))
    assert set(hkeys) <= held
    ng = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id":str}).set_index("player_id").n_games.reindex(players).to_numpy(float)
    rows = m81.load_rows()
    counts = np.zeros((2,len(players),DIM), np.float32)
    queries = np.zeros((2,H,DIM), np.float32)
    length_counts = np.zeros((len(players),8), np.float32); hlength = np.zeros(H,int)
    hlbin = np.zeros(H,int); seen = np.zeros(H,bool)
    def tasks():
        for r in rows.itertuples(index=False):
            if r.game_key in held and r.game_key not in hidx:
                continue
            yield pidx[r.player_id], hidx.get(r.game_key,-1), r.sgf_content, r.color
    with mp.get_context("fork").Pool(a.workers) as pool:
        for j,(owner, hi, arr, lb, length) in enumerate(pool.imap(one,tasks(),chunksize=128)):
            if hi >= 0:
                assert owner == truth[hi] and not seen[hi]
                seen[hi] = True; hlength[hi] = length; hlbin[hi] = lb
                for phase in range(2):
                    np.add.at(queries[phase,hi], arr[phase], 1)
            else:
                length_counts[owner,lb] += 1
                for phase in range(2):
                    np.add.at(counts[phase,owner], arr[phase], 1)
            if (j+1) % 100000 == 0:
                print(f"{j+1} games tokenized ({time.time()-start:.0f}s)",flush=True)
    assert seen.all()
    del rows
    F, ev = baseline(q,ng); ar=np.arange(H); w=ng[truth]
    ow=np.unique(truth); rng=np.random.default_rng(198); rng.shuffle(ow)
    discover=np.isin(truth,ow[:len(ow)//2]); confirm=~discover
    rate=lambda hit,mask: float(np.average(hit[mask],weights=w[mask]))
    hit0=F.argmax(1)==truth
    # Length-only nuisance control, fitted on discovery owners only.
    lp=(length_counts+1)/(length_counts.sum(1)[:,None]+8)
    L=std(np.log(lp[:,hlbin].T).astype(np.float32))
    scores=[rate((F+c*L).argmax(1)==truth,discover) for c in WEIGHTS]
    wl=WEIGHTS[int(np.argmax(scores))]; Fctrl=F+wl*L; hc=Fctrl.argmax(1)==truth
    masks={"all":confirm,"has_tail":confirm & (queries[1].sum(1)>=16),"no_tail":confirm & (queries[1].sum(1)==0)}
    out=dict(experiment="M198",report_only=True,seconds=0,discovery_owners=int(discover.sum()),confirmation_queries=int(confirm.sum()),
             baseline_sources=ev["sources"],baseline_weights=ev["weights"],baseline_beta=ev["beta"],length_weight=wl,
             baseline_weighted_top1=rate(hit0,np.ones(H,bool)),length_control_confirmation=rate(hc,confirm),arms={})
    predictions={"truth":truth,"weight":w,"confirm":confirm,"length":hlength,"baseline_hit":hit0,"control_hit":hc}
    for phase,name in enumerate(["middle_128_255","tail_256_383"]):
        Z=score(counts[phase],queries[phase])
        discovery=[rate((Fctrl+c*Z).argmax(1)==truth,discover) for c in WEIGHTS]
        weight=WEIGHTS[int(np.argmax(discovery))]; hit=(Fctrl+weight*Z).argmax(1)==truth
        delta=hit.astype(float)-hc; owners, inverse=np.unique(truth[confirm],return_inverse=True)
        numer=np.bincount(inverse,weights=w[confirm]*delta[confirm]); denom=np.bincount(inverse,weights=w[confirm])
        brng=np.random.default_rng(198); bs=[]
        for _ in range(5000):
            ix=brng.integers(0,len(owners),len(owners)); bs.append(numer[ix].sum()/denom[ix].sum())
        ci=np.quantile(bs,[.025,.975]).tolist()
        arm=dict(chosen_weight=weight,discovery_curve=dict(zip(map(str,WEIGHTS),discovery)),confirmation_ci95=ci,by_group={})
        for group, mask in masks.items():
            if mask.any():
                arm["by_group"][group]=dict(n=int(mask.sum()),baseline=rate(hit0,mask),length_control=rate(hc,mask),new=rate(hit,mask),diff=rate(delta,mask))
        arm["screen_pass"]=bool(name=="tail_256_383" and arm["by_group"]["all"]["diff"]>=.01 and ci[0]>0 and arm["by_group"].get("has_tail",{}).get("diff",0)>=.02)
        out["arms"][name]=arm; predictions[name]=hit
        print(name,json.dumps(arm),flush=True)
    out["seconds"]=time.time()-start
    out["decision"]="continue_to_independent_open_set_check" if out["arms"]["tail_256_383"]["screen_pass"] else "close_this_tail_transition_representation"
    path=ROOT / "results/r_variants/m198_tail_style.json";path.write_text(json.dumps(out,indent=2)+"\n")
    np.savez(ROOT / "results/r_variants/m198_tail_style_predictions.npz",**predictions)
    print(json.dumps(out,indent=2),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=10);ap.add_argument("--selftest",action="store_true");a=ap.parse_args()
    if a.selftest:selftest()
    else:main(a)

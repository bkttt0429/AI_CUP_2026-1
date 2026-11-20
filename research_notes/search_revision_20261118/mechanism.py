"""Measure rank contrasts, opportunity counts and repeatability on the frozen two-move menus."""
import json

import numpy as np
from scipy.special import expit

from response import (GRADES, HERE, LEVELS, RV, SIZES, aggregate, design, events,
                      groups_of_questions, questions, save_json, sha, strong)

BOOT_REPEATS = 4000
N_STRATA = 36


def ratio(a, b):
    return float(a / b) if b else None


def counts(mask, response, delta):
    choices = np.bincount(response[mask], minlength=3)
    named = mask & (response < 2) & (delta != 0)
    wins = named & (response == (delta > 0))
    return {"menus": int(mask.sum()), "choices_A_B_other": choices.tolist(),
            "named_untied": int(named.sum()), "searched_winner": int(wins.sum()),
            "winner_given_named": ratio(wins.sum(), named.sum()),
            "B_among_all": ratio(choices[1], choices.sum())}


def make_strata(e):
    direction = (e["d24"] > 0).astype(int)
    odds = np.digitize(e["pA"] / e["pB"], [1.4, 2.2])
    phase = np.digitize(e["ply"], [84, 148])
    size = (abs(e["d24"]) >= 1.5).astype(int)
    return ((direction * 3 + odds) * 3 + phase) * 2 + size


def matched_difference(nlo, slo, nhi, shi):
    common = (nlo > 0) & (nhi > 0)
    weight = np.where(common, nlo * nhi / np.maximum(nlo + nhi, 1), 0)
    contrast = shi / np.maximum(nhi, 1) - slo / np.maximum(nlo, 1)
    total = weight.sum(axis=-1)
    return np.divide((weight * contrast).sum(axis=-1), total,
                     out=np.full_like(total, np.nan, dtype=float), where=total > 0)


def matched_by_grade(games, e, include, include_games, rng):
    response, delta = e["response"], e["d24"]
    mask = include & (response < 2) & strong(e["d0"], delta)
    return matched_proportions(games,e["game"],mask,response == (delta > 0),make_strata(e),
                               include_games,N_STRATA,rng)


def matched_proportions(games, gid, mask, outcome, cell, include_games, n_strata, rng):
    """Bind all of a person's positions to one weight; recompute cell support in each draw."""
    rank = games["y"][gid]
    points = {}
    bootstrap = {}
    grade_stats = {}
    for level in GRADES:
        owners = np.unique(games["pid"][include_games & (games["y"] == level)])
        chosen = mask & (rank == level)
        owner = np.searchsorted(owners, games["pid"][gid[chosen]])
        win = outcome[chosen].astype(float)
        bins = owner * n_strata + cell[chosen]
        n = np.bincount(bins, minlength=len(owners)*n_strata).reshape(-1, n_strata)
        s = np.bincount(bins, weights=win, minlength=n.size).reshape(n.shape)
        table = np.concatenate([n, s], axis=1)
        mult = rng.multinomial(len(owners), np.full(len(owners), 1/len(owners)), size=BOOT_REPEATS)
        bn, bs = np.split(mult @ table, 2, axis=1)
        bootstrap[level] = bn, bs
        points[level] = n.sum(axis=0), s.sum(axis=0)
        total, wins = bn.sum(axis=1), bs.sum(axis=1)
        samples = wins[total > 0] / total[total > 0]
        grade_stats[LEVELS[level]] = {"opportunities": int(n.sum()), "owners_all": len(owners),
            "owners_observed": int((n.sum(axis=1)>0).sum()), "wins": int(s.sum()),
            "rate": ratio(s.sum(), n.sum()), "ci95": np.quantile(samples, [.025,.975]).tolist()}
    contrasts = {}
    for lo, hi in ((4,5), (6,7), (0,9)):
        lower, higher = points[lo], points[hi]
        both = (lower[0] > 0) & (higher[0] > 0)
        sample = matched_difference(*bootstrap[lo], *bootstrap[hi])
        finite = np.isfinite(sample)
        contrasts[LEVELS[lo]+"/"+LEVELS[hi]] = {
            "exploratory": bool((lo,hi)==(0,9)),
            "difference": float(matched_difference(*lower,*higher)),
            "ci95": np.quantile(sample[finite], [.025,.975]).tolist(), "draws": int(finite.sum()),
            "strata_with_both": int(both.sum()), "opportunities_in_common_support":
            [int(lower[0][both].sum()), int(higher[0][both].sum())]}
    return grade_stats, contrasts


def differences_over_games(games, e, roots, shared):
    gid, resp = e["game"], e["response"]
    rev = strong(e["d0"], e["d24"])
    named = (resp < 2) & (e["d24"] != 0)
    captured = resp == (e["d24"] > 0)
    x = design({k:v[named] for k,v in e.items()})
    residual = resp[named] - expit(x @ shared)
    revision = x[:,7]
    def cnt(mask, weights=None):
        return np.bincount(gid[mask], weights=weights, minlength=len(roots))
    return np.column_stack([roots, cnt(slice(None)), cnt(resp < 2), cnt(rev), cnt(rev & named),
                           cnt(rev & named & captured), cnt(named), cnt(named & captured),
                           cnt(named, weights=residual*revision), cnt(named, weights=revision**2)])


def split_half(counts_game, twenty_games, grade):
    """Each of the twenty random ten-plus-ten splits uses disjoint games for a given player."""
    rng = np.random.default_rng(20261121)
    per_player = counts_game[twenty_games]
    scores = {}
    support = []
    def calc(total):
        return {"searched_choice_rate": (total[:,7]+.5)/(total[:,6]+1),
                "strong_reversal_choice_rate": (total[:,5]+.5)/(total[:,4]+1),
                "continuous_revision_response": total[:,8]/(4+total[:,9]),
                "named_pair_rate": (total[:,2]+.5)/(total[:,1]+1)}
    def demean(a, g):
        return a-(np.bincount(g,weights=a,minlength=10)/np.maximum(1,np.bincount(g,minlength=10)))[g]
    for _ in range(20):
        order = rng.random(twenty_games.shape).argsort(axis=1)
        halves = [np.take_along_axis(per_player, idx[:,:,None],axis=1).sum(axis=1)
                  for idx in (order[:,:10],order[:,10:])]
        left,right = map(calc,halves)
        valid = (halves[0][:,4] > 0) & (halves[1][:,4] > 0)
        support.append(float(valid.mean()))
        for name in left:
            a,b = left[name],right[name]
            output = scores.setdefault(name,{"all": [], "within_rank": []})
            output["all"].append(float(np.corrcoef(a,b)[0,1]))
            output["within_rank"].append(float(np.corrcoef(demean(a,grade),demean(b,grade))[0,1]))
            if name=="strong_reversal_choice_rate" and valid.sum()>2:
                output.setdefault("both_halves_observed",[]).append(float(np.corrcoef(a[valid],b[valid])[0,1]))
    return {"players":len(twenty_games),"splits":20,"median_both_halves_have_a_strong_named_move":float(np.median(support)),
            "median_person_correlations":{k:{a:float(np.median(v)) for a,v in vv.items()} for k,vv in scores.items()}}


def main():
    games,e,_,roots = events()
    qs,_ = questions()
    groups = groups_of_questions(games,qs)
    rank = json.loads((HERE/"rank_results.json").read_text())
    assert rank["protocol_sha256"] == sha(HERE/"protocol.json")
    shared = np.array(rank["fits"]["0"]["common"]["shared"])
    per_game = differences_over_games(games,e,roots,shared)
    game,resp,delta=e["game"],e["response"],e["d24"]
    rev=strong(e["d0"],delta)
    stable=(e["d0"]*delta>0)&(np.minimum(abs(e["d0"]),abs(delta))>=.5)
    spatial_dist=abs(e["A"]%19-e["B"]%19)+abs(e["A"]//19-e["B"]//19)
    same_side=games["fold"]>=3
    results = {}
    for tag,game_set,seed in (("question_games",same_side & ~games["fresh"],20261122),
                              ("new_games",games["fresh"],20261123)):
        m=game_set[game]
        grades,adjacent=matched_by_grade(games,e,m,game_set,np.random.default_rng(seed))
        outcome = {name:counts(m&mask,resp,delta) for name,mask in (
            ("eligible",np.ones(len(game),bool)),("strong_reversal",rev),("strong_stable",stable),
            ("raw_A_to_search_B",rev&(delta>0)),("raw_B_to_search_A",rev&(delta<0)),
            ("exploratory_strong_manhattan_below_8",rev&(spatial_dist<8)),
            ("exploratory_strong_manhattan_at_least_8",rev&(spatial_dist>=8)))}
        results[tag]={"games":int(game_set.sum()),"owners":len(set(games["pid"][game_set])),"roots":int(roots[game_set].sum()),
                      "choices":outcome, "observed_by_grade":grades,"matched_strong_reversal":adjacent}
    with np.load(RV/"m183_dump.npz") as z:
        f,pids,y,n = (z[k] for k in ("fold","pid","y","n"))
    per_q=aggregate(per_game[:,:5],groups)
    fixed={}
    for size in SIZES:
        sel=(f>=3)&(n==size);col=per_q[sel]
        fixed[str(size)]={"questions":int(sel.sum()),
            "medians":dict(zip(["sampled","eligible","named","strong","strong_named"],np.median(col,axis=0).tolist())),
            "strong_named_at_least_1":float((col[:,4]>=1).mean()),"strong_named_at_least_2":float((col[:,4]>=2).mean())}
    rows,idx=groups[20];valid=f[rows]>=3
    assert len(set(pids[rows][valid]))==valid.sum()
    result={"samples":results, "question_coverage":fixed, "repeatability":split_half(per_game,idx[valid],y[rows][valid]),
            "bootstrap_draws":BOOT_REPEATS,"sha256":{p.name:sha(p) for p in (HERE/"protocol.json",HERE/"rank_results.json",HERE/"mechanism.py")}}
    save_json(HERE/"mechanism.json",result)
    print(json.dumps({"coverage":fixed,"repeatability":result["repeatability"],"matched":
        {k:v["matched_strong_reversal"] for k,v in results.items()}},indent=2),flush=True)


if __name__=="__main__":
    main()

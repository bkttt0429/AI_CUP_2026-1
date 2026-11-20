"""Borrow strength across players for sparse defensive and attacking choices."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import betaln, logsumexp

from extract_games import HERE, RV, questions, save_json, sha
from evaluate import RANKS, SIZES, balanced_mean_over_n, balanced_weights, cluster_intervals, decide, guard2, score


def counts(cells, task, rng=None):
    if task == "defense":
        total, favor, chose, joint = np.moveaxis(cells, -1, 0)
    else:
        _, _, total, favor, chose, joint = np.moveaxis(cells, -1, 0)
    if rng is not None:
        joint = np.zeros_like(joint)
        active = total > 0
        joint[active] = rng.hypergeometric(favor[active], (total-favor)[active], chose[active])
    fine_n = np.stack([total-favor, favor], axis=-1).reshape(len(cells), -1)
    fine_a = np.stack([chose-joint, joint], axis=-1).reshape(len(cells), -1)
    assert np.all((fine_a >= 0) & (fine_a <= fine_n))
    np.testing.assert_array_equal(fine_a.reshape(len(cells), -1, 2).sum(axis=2), chose)
    return (total, chose), (fine_n, fine_a)


def combine(roles):
    return [tuple(np.column_stack([part[level][i] for part in roles]) for i in (0,1)) for level in (0,1)]


def fit_tables(train, rank, person_counts):
    tables = []
    for trials, yes in person_counts:
        t, s = trials[train], yes[train]
        prior = (s.sum(axis=0) + .5) / (t.sum(axis=0) + 1.)
        means = np.stack([(s[rank[train]==k].sum(axis=0) + 24*prior) /
                          (t[rank[train]==k].sum(axis=0) + 24) for k in RANKS])
        assert np.all((means > 0) & (means < 1))
        tables.append(means)
    return tables


def predict_ll(tables, groups, index):
    out = []
    for means, (trials, yes) in zip(tables, groups):
        successes = yes[index]
        failures = trials[index] - successes
        grade_ll = []
        for mu in means:
            a, b = 12*mu, 12*(1-mu)
            terms = betaln(a+successes, b+failures) - betaln(a,b)
            grade_ll.append(terms.sum(axis=1))
        ll = np.column_stack(grade_ll)
        # The common likelihood offset cancels across candidate grades.
        out.append(ll - ll.mean(axis=1, keepdims=True))
    return out


def calibrate(logp, evidence, y, sizes):
    w = balanced_weights(y, sizes)
    idx = np.arange(len(y))
    def loss(a):
        lp = logp + a*evidence
        return w @ (logsumexp(lp, axis=1)-lp[idx, y])
    optimum = minimize_scalar(loss, bounds=(0,1), method="bounded", options={"xatol":1e-9})
    assert optimum.success
    return float(min((0.,float(optimum.x),1.), key=loss))


def main():
    questions_all, _ = questions()
    select = np.array([q[0] in (1,2) for q in questions_all])
    qs = [q for q, ok in zip(questions_all, select) if ok]
    with np.load(RV/"m183_dump.npz") as z:
        n, y, fold, pid, ref, raw, frozen = [z[k][select] for k in ("n","y","fold","pid","ct106_rule","anat_plain","SQ4")]
        p = z["anat_post"][select].astype(float)
    p /= p.sum(axis=1,keepdims=True)
    np.testing.assert_array_equal(decide(p),raw)
    np.testing.assert_array_equal(guard2(raw,ref),frozen)
    logp = np.log(p)
    owners, owner_idx = np.unique(pid, return_inverse=True)
    person_fold = np.zeros(len(owners),int)
    person_fold[owner_idx] = fold
    person_rank = np.zeros(len(owners),int)
    person_rank[owner_idx] = y
    np.testing.assert_array_equal(person_rank[owner_idx],y)
    np.testing.assert_array_equal(person_fold[owner_idx],fold)
    question_roles, null_roles, player_roles, hashes = [], [], [], {}
    rng = np.random.default_rng(20261120)
    for role, fname, meta in (("defense","game_counts.npz","extraction.json"),("attack","attack_counts.npz","attack_extraction.json")):
        filename = HERE/fname
        hashes[fname] = sha(filename)
        assert hashes[fname] == json.loads((HERE/meta).read_text())["counts_sha256"]
        with np.load(filename) as z:
            lookup = {k:i for i,k in enumerate(z["keys"])}
            valid, per_game = z["valid"],z["cells"]
        game_indices = [[lookup[k] for k in q[4]] for q in qs]
        own_games = [set() for _ in owners]
        for i,games in zip(owner_idx,game_indices):
            own_games[i].update(games)
        all_unique = set.union(*own_games)
        assert sum(map(len,own_games)) == len(all_unique)
        assert valid[list(all_unique)].all()
        pooled = np.stack([per_game[sorted(games)].sum(axis=0,dtype=np.int64) for games in own_games])
        qcounts = np.stack([per_game[games].sum(axis=0,dtype=np.int64) for games in game_indices])
        player_roles.append(counts(pooled,role))
        question_roles.append(counts(qcounts,role))
        null_roles.append(counts(qcounts,role,rng))
    query, null, people = map(combine,(question_roles,null_roles,player_roles))
    for i in (0,1):
        np.testing.assert_array_equal(query[0][i], null[0][i])
    preds={"SQ4":frozen}
    losses={"SQ4":-logp[np.arange(len(y)),y]}
    names=["coarse","full","shuffle"]
    for name in names:
        preds[name]=np.full_like(y,-1)
        losses[name]=np.full(len(y),np.nan)
    meta = {}
    for held in (1,2):
        training, evaluation = fold != held, fold == held
        fit_people = person_fold != held
        inner = np.full(len(owners),-1)
        splitter = np.random.default_rng(20261118+held)
        for k in RANKS:
            group = np.flatnonzero(fit_people & (person_rank==k))
            shuffled = splitter.permutation(group)
            inner[shuffled] = np.arange(len(group))%3
        inner_ll = [np.full_like(p,np.nan) for _ in (0,1)]
        for inner_test in range(3):
            tr = fit_people & (inner!=inner_test)
            val = inner[owner_idx]==inner_test
            assert np.all(training[val]) and not np.any(tr[owner_idx[val]])
            tables = fit_tables(tr,person_rank,people)
            for dest, source in zip(inner_ll,predict_ll(tables,query,val)):
                dest[val]=source
        assert all(np.isfinite(ll[training]).all() and np.isnan(ll[evaluation]).all() for ll in inner_ll)
        coarse, fine = (ll[training] for ll in inner_ll)
        alpha0 = calibrate(logp[training], coarse, y[training], n[training])
        alpha1 = calibrate(logp[training]+alpha0*coarse, fine-coarse, y[training], n[training])
        tabs = fit_tables(fit_people,person_rank,people)
        coarse, fine = predict_ll(tabs,query,evaluation)
        coarse_null, shuffled = predict_ll(tabs,null,evaluation)
        np.testing.assert_array_equal(coarse_null,coarse)
        lp0 = logp[evaluation]+alpha0*coarse
        for name, lp in zip(names,(lp0,lp0+alpha1*(fine-coarse),lp0+alpha1*(shuffled-coarse))):
            lpn = lp-logsumexp(lp,axis=1)[:,None]
            preds[name][evaluation]=guard2(decide(np.exp(lpn)),ref[evaluation])
            losses[name][evaluation]=-lpn[np.arange(evaluation.sum()),y[evaluation]]
        meta[str(held)]={"temperatures":[alpha0,alpha1],"training_players":int(fit_people.sum()),
                        "evaluation_players":int((~fit_people).sum()),"question_counts":int(evaluation.sum()),
                        "means_coarse":tabs[0].tolist(),"means_fine":tabs[1].tolist()}
    assert all(np.all((pr>=0) & (pr<10)) for pr in preds.values())
    assert all(np.isfinite(l).all() for l in losses.values())
    sc = {k:score(v,y) for k,v in preds.items()}
    pairs = [("full","SQ4"),("full","coarse"),("full","shuffle"),("coarse","SQ4")]
    differences=np.column_stack([sc[a]-sc[b] for a,b in pairs])
    intervals=cluster_intervals(differences,y,n,pid)
    paired={}
    for (a,b),ci in zip(pairs,intervals):
        d = sc[a]-sc[b]
        paired[a+"_minus_"+b]={"difference":balanced_mean_over_n(d,y,n),"ci95":ci.tolist(),
            "folds":{str(f):balanced_mean_over_n(d[fold==f],y[fold==f],n[fold==f]) for f in (1,2)},
            "by_n":{str(s):balanced_mean_over_n(d,y,n,sizes=[s]) for s in SIZES},
            "changed":int((preds[a]!=preds[b]).sum()),"improved":int((d>0).sum()),"worsened":int((d<0).sum()),
            "nll_difference":balanced_mean_over_n(losses[a]-losses[b],y,n)}
    primary=paired["full_minus_SQ4"]
    lead=primary["difference"]>=.001 and min(primary["folds"].values())>0 and primary["ci95"][0]>0 and paired["full_minus_coarse"]["ci95"][0]>0
    dest=HERE/"pooled_predictions.npz"
    np.savez_compressed(dest,fold=fold,n=n,y=y,pid=pid,**{"pred_"+k:v for k,v in preds.items()})
    hashes.update({p.name:sha(p) for p in (HERE/"pooled_protocol.json", HERE/"evaluate.py", Path(__file__), dest, RV/"m183_dump.npz")})
    report={"development_lead":bool(lead),"players":len(owners),"questions":len(qs),"unique_games":len(all_unique),"models":meta,
            "paired":paired,"evaluation":{k:{"score":balanced_mean_over_n(sc[k],y,n),"nll":balanced_mean_over_n(losses[k],y,n)} for k in preds},
            "sha256":hashes}
    save_json(HERE/"pooled_results.json",report)
    print(json.dumps({"temperatures":{k:v["temperatures"] for k,v in meta.items()}, "paired":paired,"lead":lead},indent=2))


if __name__=="__main__":
    main()

from pathlib import Path
p = Path("scripts/trunk_seq_check.py"); s = p.read_text()
R = [
('''Everything from the loss back to the blocks trains together; the trunk vectors are fixed (scripts/trunk_extract.py).
C's hidden size is chosen so its blocks hold about as many parameters as T's. The final epoch is the arm's answer; the per-epoch
validation scores are diagnostics and select nothing. Fold 0 is refused.
''', '''Everything from the loss back to the blocks trains together; the trunk vectors are fixed (scripts/trunk_extract.py).
C's hidden size is chosen so its blocks hold about as many parameters as T's. With --inner-frac the stopping point is chosen on
held-out training players only and the validation fold is scored once, at that point. Fold 0 is refused.
'''),
('''  python -I scripts/trunk_seq_check.py compare --val-fold 1 --out results/trunk_seq
''', '''  python -I scripts/trunk_seq_check.py rfeat --val-fold 2
  python -I scripts/trunk_seq_check.py compare --val-folds 2 3 4 --out results/trunk_seq/es
'''),
('''    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in FOLDS:''', '''    out = Path(a.cache)
    out.mkdir(parents=True, exist_ok=True)
    for f in FOLDS:'''),
('''    path = Path(a.out) / f"r_val{v}.npz"
    meta_path = Path(a.out) / f"r_val{v}.json"''', '''    path = Path(a.cache) / f"r_val{v}.npz"
    meta_path = Path(a.cache) / f"r_val{v}.json"'''),
('''    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez(path, X_tr=Xt''', '''    Path(a.cache).mkdir(parents=True, exist_ok=True)
    np.savez(path, X_tr=Xt'''),
('''        X, offsets, stats = load_tokens([f for f in FOLDS if f != a.val_fold], a.out, device)
        Xv, offsets_v, _ = load_tokens([a.val_fold], a.out, device, stats)''', '''        X, offsets, stats = load_tokens([f for f in FOLDS if f != a.val_fold], a.cache, device)
        Xv, offsets_v, _ = load_tokens([a.val_fold], a.cache, device, stats)'''),
('''    if a.limit:  # smoke runs only
''', '''    R_in, y_in, keys_in, q_in = None, None, None, None
    if a.inner_frac > 0:  # the same held-out players for every arm and seed of this validation fold
        pid_tr = q_tr["player_id"].to_numpy(dtype=str)
        players = np.array(sorted(set(pid_tr)))
        held = set(np.random.RandomState(12345 + a.val_fold).permutation(players)[:int(a.inner_frac * len(players))])
        is_in = np.array([x in held for x in pid_tr])
        pick = np.random.RandomState(a.val_fold).permutation(np.flatnonzero(is_in))[:a.inner_questions]
        R_in, y_in, keys_in, q_in = R_tr[pick], y_tr[pick], [keys_tr[i] for i in pick], q_tr.iloc[pick]
        keep = np.flatnonzero(~is_in)
        R_tr, y_tr, keys_tr = R_tr[keep], y_tr[keep], [keys_tr[i] for i in keep]
        print(f"inner: {len(held)} of {len(players)} training players held out, {len(pick)} questions score the stopping point", flush=True)
    if a.limit:  # smoke runs only
'''),
('''    history, step = [], 0
    for epoch in range(a.epochs):
        model.train()
''', '''    history, step = [], 0
    every = max(1, (len(R_tr) // a.batch) // a.evals_per_epoch)
    best, best_state, best_step, since = -1.0, None, 0, 0
    stop = False
    for epoch in range(a.epochs):
        if stop:
            break
        model.train()
'''),
('''            run, n_run, step = run + loss.item(), n_run + 1, step + 1
        P = predict(model, R_va, keys_va, Xv, offsets_v, device)
        res, scores = evaluate(P, y_va, q_va)
        history.append({"epoch": epoch + 1, "train_loss": run / max(n_run, 1), "val": res["balanced_mean_over_n"], "val_nll": res["nll_balanced"], "s": time.time() - t_ep})
        print(f"{a.arm} seed {a.seed} epoch {epoch + 1}/{a.epochs}: loss {run / max(n_run, 1):.4f}  val {res['balanced_mean_over_n']:.4f}  NLL {res['nll_balanced']:.4f}  "
              f"{time.time() - t_ep:.0f} s  peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
''', '''            run, n_run, step = run + loss.item(), n_run + 1, step + 1
            if a.inner_frac > 0 and step % every == 0:
                Pi = predict(model, R_in, keys_in, X, offsets, device)
                _, si = evaluate(Pi, y_in, q_in, sizes=tuple(range(5, 21)))
                inner = float(np.mean([si[y_in == c].mean() for c in np.unique(y_in)]))
                history.append({"step": step, "epoch": step / (len(R_tr) // a.batch), "train_loss": run / max(n_run, 1), "inner": inner})
                print(f"{a.arm} seed {a.seed} step {step}/{steps} (epoch {history[-1]['epoch']:.2f}): loss {run / max(n_run, 1):.4f}  inner {inner:.4f}  {time.time() - t0:.0f} s", flush=True)
                run, n_run = 0.0, 0
                if inner > best:
                    best, best_step, since = inner, step, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    since += 1
                    if since >= a.patience:
                        stop = True
                        break
                model.train()
        if a.inner_frac <= 0:
            history.append({"epoch": epoch + 1, "train_loss": run / max(n_run, 1)})
            print(f"{a.arm} seed {a.seed} epoch {epoch + 1}/{a.epochs}: loss {run / max(n_run, 1):.4f}  {time.time() - t_ep:.0f} s  peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    P = predict(model, R_va, keys_va, Xv, offsets_v, device)
    res, scores = evaluate(P, y_va, q_va)
    print(f"{a.arm} seed {a.seed}: stopped at step {best_step or step} (inner {best:.4f}); validation {res['balanced_mean_over_n']:.4f}  NLL {res['nll_balanced']:.4f}", flush=True)
'''),
('''"dropout": a.dropout, "history": history,
                                                             "final": history[-1]["val"], "wall_s": time.time() - t0}, indent=1))''', '''"dropout": a.dropout, "history": history, "inner_frac": a.inner_frac,
                                                             "best_step": best_step, "best_inner": best, "steps_per_epoch": len(R_tr) // a.batch,
                                                             "final": res["balanced_mean_over_n"], "final_nll": res["nll_balanced"], "wall_s": time.time() - t0}, indent=1))'''),
('''def compare(a):
    _, _, _, y, _, q = r_features(a, a.val_fold)
    d = Path(a.out) / f"val{a.val_fold}"
    pid, n = q["player_id"].to_numpy(dtype=str), q["n_games"].to_numpy()
    sc, per_seed, rep = {}, {}, {}
    for arm in ("R0", "C", "T"):
        files = sorted(d.glob(f"{arm}_s*.npz"))
        if not files:
            continue
        s = [np.load(f)["scores"] for f in files]
        sc[arm] = np.mean(s, axis=0)
        per_seed[arm] = [balanced_mean_over_n(x, y, n) for x in s]
        rep[arm] = {"seeds": per_seed[arm], "mean_of_seeds": float(np.mean(per_seed[arm])), "by_n": {str(m): balanced_mean_over_n(sc[arm], y, n, sizes=(m,)) for m in SIZES}}
        print(f"{arm}: {rep[arm]['mean_of_seeds']:.4f}  seeds " + " ".join(f"{x:.4f}" for x in per_seed[arm]) + "  by n " + " ".join(f"{m}: {rep[arm]['by_n'][str(m)]:.4f}" for m in SIZES))
    pairs = {}
    for x, z in (("T", "C"), ("C", "R0"), ("T", "R0")):
        if x in sc and z in sc:
            dd = paired_diff(sc[x], sc[z], y, pid, n, n=a.n_boot)
            pairs[f"{x} vs {z}"] = dd
            print(f"{x} vs {z}: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  by n " + " ".join(f"{m}: {v:+.4f}" for m, v in dd["by_n"].items()))
    (d / "compare.json").write_text(json.dumps({"val_fold": a.val_fold, "arms": rep, "pairs": pairs}, indent=1, default=float))
    return 0
''', '''def compare(a):
    """Per-question scores averaged over seeds, per fold and pooled; conditions (1) CI lower bound > 0, (3) > 0 in all but one fold, (4) >= 0.005."""
    folds = a.val_folds or [a.val_fold]
    ys, pids, ns, sc, per_fold, seeds, stops = [], [], [], {}, {}, {}, {}
    for v in folds:
        _, _, _, y, _, q = r_features(a, v)
        d = Path(a.out) / f"val{v}"
        pid, n = q["player_id"].to_numpy(dtype=str), q["n_games"].to_numpy()
        ys.append(y); pids.append(pid); ns.append(n)
        for arm in ARMS:
            files = sorted(d.glob(f"{arm}_s*.npz"))
            s = [np.load(f)["scores"] for f in files]
            sc.setdefault(arm, []).append(np.mean(s, axis=0))
            per_fold.setdefault(arm, []).append(float(np.mean([balanced_mean_over_n(x, y, n) for x in s])))
            seeds.setdefault(arm, {})[str(v)] = [balanced_mean_over_n(x, y, n) for x in s]
            metas = [json.loads(f.with_suffix(".json").read_text()) for f in files]
            stops.setdefault(arm, {})[str(v)] = [m["best_step"] / m["steps_per_epoch"] for m in metas]
    y, pid, n = np.concatenate(ys), np.concatenate(pids), np.concatenate(ns)
    S = {arm: np.concatenate(sc[arm]) for arm in ARMS}
    rep = {}
    for arm in ARMS:
        rep[arm] = {"score": float(np.mean(per_fold[arm])), "per_fold": per_fold[arm], "seeds": seeds[arm], "stop_epoch": stops[arm], "by_n": {str(m): balanced_mean_over_n(S[arm], y, n, sizes=(m,)) for m in SIZES}}
        print(f"{arm}: {rep[arm]['score']:.4f}  folds " + " ".join(f"{x:.4f}" for x in per_fold[arm]) + "  by n " + " ".join(f"{m}: {rep[arm]['by_n'][str(m)]:.4f}" for m in SIZES)
              + "  stop epoch " + " ".join(f"{np.mean(v):.2f}" for v in stops[arm].values()))
    pairs, need = {}, max(1, len(folds) - 1)
    for x, z in (("T", "C"), ("C", "R0"), ("T", "R0")):
        dd = paired_diff(S[x], S[z], y, pid, n, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]
        cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= need), "4": bool(dd["diff"] >= 0.005)}
        pairs[f"{x} vs {z}"] = {**dd, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}
        print(f"{x} vs {z}: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {v:+.4f}" for m, v in dd["by_n"].items())
              + f"  -> {'PASS' if pairs[f'{x} vs {z}']['pass'] else 'no'}")
    Path(a.out, f"compare_{'_'.join(map(str, folds))}.json").write_text(json.dumps({"folds": folds, "arms": rep, "pairs": pairs}, indent=1, default=float))
    return 0


def rfeat(a):
    r_features(a, a.val_fold)
    print(f"R features of validation fold {a.val_fold} cached", flush=True)
    return 0
'''),
('''    ap.add_argument("cmd", choices=("build", "train", "compare"))
    ap.add_argument("--out", default=str(ROOT / "results/trunk_seq"))
''', '''    ap.add_argument("cmd", choices=("build", "rfeat", "train", "compare"))
    ap.add_argument("--out", default=str(ROOT / "results/trunk_seq"))
    ap.add_argument("--cache", default=str(ROOT / "results/trunk_seq"), help="the score arrays and R features")
    ap.add_argument("--val-folds", type=int, nargs="+", default=None)
    ap.add_argument("--inner-frac", type=float, default=0.0)
    ap.add_argument("--inner-questions", type=int, default=4000)
    ap.add_argument("--evals-per-epoch", type=int, default=4)
    ap.add_argument("--patience", type=int, default=8)
'''),
('''    return {"build": build, "train": train, "compare": compare}[a.cmd](a)''', '''    if 0 in (a.val_folds or []):
        ap.error("fold 0 is not used here")
    return {"build": build, "rfeat": rfeat, "train": train, "compare": compare}[a.cmd](a)'''),
('''GAME_DIM, Q_DIM = 64, 32
''', '''GAME_DIM, Q_DIM = 64, 32
ARMS = ("R0", "C", "T")
'''),
]
for x, y in R:
    assert s.count(x) == 1, x[:60]
    s = s.replace(x, y)
p.write_text(s)
print("patched", len(R))

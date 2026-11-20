from pathlib import Path
p = Path("scripts/trunk_seq_check.py"); s = p.read_text()
R = [
('def r_features(a, v):\n', 'def r_features(a, v, pool=0):\n', 1),
('    path = Path(a.cache) / f"r_val{v}.npz"\n    meta_path = Path(a.cache) / f"r_val{v}.json"\n',
 '    tag = f"r_val{v}" + (f"_p{pool}" if pool else "")  # pool k > 0: another draw of the 32 training questions per player\n    path = Path(a.cache) / f"{tag}.npz"\n    meta_path = Path(a.cache) / f"{tag}.json"\n', 1),
('    rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)\n    train_q', '    rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v + 1000 * pool)\n    train_q', 1),
('    X_tr, y_tr, X_va, y_va, q_tr, q_va = r_features(a, a.val_fold)\n    mean, std', '    X_tr, y_tr, X_va, y_va, q_tr, q_va = r_features(a, a.val_fold)\n    y_all = y_tr.copy()\n    held = set()\n    mean, std', 1),
('''    if a.limit:  # smoke runs only
        keep = np.random.RandomState(0).permutation(len(R_tr))[:a.limit]
        R_tr, y_tr, keys_tr = R_tr[keep], y_tr[keep], [keys_tr[i] for i in keep]
''', '''    bagged = None
    if a.bag < 1:  # bagging over players: this model sees a seeded share of the training players
        cand = np.array(sorted(set(q_tr["player_id"].astype(str)) - held))
        bagged = set(np.random.RandomState(777 + a.seed + 100 * a.val_fold).permutation(cand)[:int(a.bag * len(cand))])

    def prep(Xp, yp, qp):
        """One draw of training questions without the held-out players (and, with --bag, only the bagged ones), standardised as pool 0;
        weights are class-balanced, times each question's mean game uniqueness (1 / the number of questions using the game) with --uniqueness."""
        pid = qp["player_id"].to_numpy(dtype=str)
        ok = np.array([(x not in held) and (bagged is None or x in bagged) for x in pid])
        Rp = ((Xp[ok] - mean) / std).astype(np.float32)
        kp = [k for k, o in zip(qp["game_keys"].tolist(), ok) if o]
        w = class_weights(yp[ok])
        if a.uniqueness:
            count = {}
            for ks in kp:
                for g in ks:
                    count[g] = count.get(g, 0) + 1
            u = np.array([np.mean([1.0 / count[g] for g in ks]) for ks in kp])
            w = w * u / u.mean()
            w = w / w.mean()
        return Rp, yp[ok], kp, w

    pools = [prep(X_tr, y_all, q_tr)]
    for k in range(1, a.resample):  # --resample K: epoch e trains on draw e mod K
        Xk, yk, _, _, qk, _ = r_features(a, a.val_fold, k)
        pools.append(prep(Xk, yk, qk))
    if a.limit:  # smoke runs only
        keep = np.random.RandomState(0).permutation(len(pools[0][1]))[:a.limit]
        pools[0] = (pools[0][0][keep], pools[0][1][keep], [pools[0][2][i] for i in keep], pools[0][3][keep])
    R_tr, y_tr, keys_tr, _ = pools[0]
    print(f"training draws: {[len(x[1]) for x in pools]} questions" + (f"; {len(bagged)} bagged players" if bagged else ""), flush=True)
''', 1),
('''    weight = torch.as_tensor(class_weights(y_tr), device=device, dtype=torch.float32)
    y_t = torch.as_tensor(y_tr, device=device).long()
    R_t = torch.as_tensor(R_tr, device=device)
''', '''    pool_t = [(torch.as_tensor(Rp, device=device), torch.as_tensor(yp, device=device).long(), torch.as_tensor(wp, device=device, dtype=torch.float32), kp)
              for Rp, yp, kp, wp in pools]
''', 1),
('''        model.train()
        perm = rng.permutation(len(R_tr))
''', '''        model.train()
        R_t, y_t, weight, keys_tr = pool_t[epoch % len(pool_t)]
        perm = rng.permutation(len(keys_tr))
''', 1),
('    tag = a.arm + (f"-no{a.drop}" if a.drop else "")\n', '    tag = a.arm + (f"-no{a.drop}" if a.drop else "") + ("-rs" if a.resample > 1 else "") + ("-uq" if a.uniqueness else "") + ("-bg" if a.bag < 1 else "")\n', 1),
('"arm": tag, "drop": a.drop,', '"arm": tag, "drop": a.drop, "resample": a.resample, "uniqueness": a.uniqueness, "bag": a.bag,', 1),
('    r_features(a, a.val_fold)\n    print(f"R features of validation fold {a.val_fold} cached", flush=True)', '    r_features(a, a.val_fold, a.pool)\n    print(f"R features of validation fold {a.val_fold}, draw {a.pool} cached", flush=True)', 1),
('    ap.add_argument("--inner-frac", type=float, default=0.0)\n', '''    ap.add_argument("--inner-frac", type=float, default=0.0)
    ap.add_argument("--resample", type=int, default=1, help="number of training-question draws, cycled by epoch")
    ap.add_argument("--uniqueness", action="store_true", help="weight questions by the mean uniqueness of their games")
    ap.add_argument("--bag", type=float, default=1.0, help="share of training players this model sees")
    ap.add_argument("--pool", type=int, default=0, help="rfeat: which draw of training questions")
''', 1),
]
for x, y, n in R:
    assert s.count(x) == n, (x[:70], s.count(x))
    s = s.replace(x, y)
p.write_text(s); print("patched", len(R))

from pathlib import Path
p = Path("scripts/trunk_seq_check.py"); s = p.read_text()
R = [
("def load_tokens(folds, cache, device, stats=None):\n", "def load_tokens(folds, cache, device, stats=None, drop=None):\n", 1),
("""    X[:, :D_TOKEN - 4] = ((X[:, :D_TOKEN - 4].astype(np.float32) - stats[0]) / stats[1]).astype(np.float16)
    return torch.as_tensor(X, device=device), offsets, stats""", """    X[:, :D_TOKEN - 4] = ((X[:, :D_TOKEN - 4].astype(np.float32) - stats[0]) / stats[1]).astype(np.float16)
    if drop == "trunk":  # ablation: the columns stay, so every arm has the same parameters
        X[:, :128] = 0
    elif drop == "scores":
        X[:, 128:128 + N_SCORES] = 0
    return torch.as_tensor(X, device=device), offsets, stats""", 1),
("X, offsets, stats = load_tokens([f for f in FOLDS if f != a.val_fold], a.cache, device)", "X, offsets, stats = load_tokens([f for f in FOLDS if f != a.val_fold], a.cache, device, drop=a.drop)", 1),
("Xv, offsets_v, _ = load_tokens([a.val_fold], a.cache, device, stats)", "Xv, offsets_v, _ = load_tokens([a.val_fold], a.cache, device, stats, drop=a.drop)", 1),
('np.savez(out / f"{a.arm}_s{a.seed}.npz"', 'tag = a.arm + (f"-no{a.drop}" if a.drop else "")\n    np.savez(out / f"{tag}_s{a.seed}.npz"', 1),
('(out / f"{a.arm}_s{a.seed}.json").write_text(json.dumps({"arm": a.arm,', '(out / f"{tag}_s{a.seed}.json").write_text(json.dumps({"arm": tag, "drop": a.drop,', 1),
('    folds = a.val_folds or [a.val_fold]\n', '    folds = a.val_folds or [a.val_fold]\n    arms = a.arms\n    pairs_spec = [tuple(x.split(":")) for x in a.pairs]\n', 1),
("for arm in ARMS:", "for arm in arms:", 2),
("S = {arm: np.concatenate(sc[arm]) for arm in ARMS}", "S = {arm: np.concatenate(sc[arm]) for arm in arms}", 1),
('for x, z in (("T", "C"), ("C", "R0"), ("T", "R0")):\n        dd = paired_diff(S[x], S[z]', 'for x, z in pairs_spec:\n        dd = paired_diff(S[x], S[z]', 1),
("""            files = sorted(d.glob(f"{arm}_s*.npz"))
            s = [np.load(f)["scores"] for f in files]
            sc.setdefault""", """            files = [f for f in sorted(d.glob(f"{arm}_s*.npz")) if a.seeds is None or int(f.stem.rsplit("_s", 1)[1]) in a.seeds]
            s = [np.load(f)["scores"] for f in files]
            sc.setdefault""", 1),
('Path(a.out, f"compare_{\'_\'.join(map(str, folds))}.json")', 'Path(a.out, f"compare_{a.tag or \'\'}{\'_\'.join(map(str, folds))}.json")', 1),
('    ap.add_argument("--val-folds", type=int, nargs="+", default=None)\n', '''    ap.add_argument("--val-folds", type=int, nargs="+", default=None)
    ap.add_argument("--drop", choices=("trunk", "scores"), default=None, help="ablation: zero the trunk vectors or the per-move scores")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--pairs", nargs="+", default=["T:C", "C:R0", "T:R0"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="compare: only these seeds")
    ap.add_argument("--tag", default=None, help="compare: prefix of the output file")
''', 1),
]
for x, y, n in R:
    assert s.count(x) == n, (x[:70], s.count(x))
    s = s.replace(x, y)
p.write_text(s); print("patched", len(R))

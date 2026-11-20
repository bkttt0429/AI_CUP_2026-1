"""Exploratory reanalysis of the saved M197 probe; no model training or GPU use.

Run from repository root with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1:
  .venv/bin/python research_notes/choice_signal_20261116.py
Results are exploratory: the same 300 players were used in the original probe.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/r_variants/m197_probe.npz"


def distances(query, prototypes, covariance):
    whitening = np.linalg.cholesky(np.linalg.inv(covariance))
    q, p = query @ whitening, prototypes @ whitening
    return (q * q).sum(1)[:, None] - 2 * q @ p.T + (p * p).sum(1)[None, :]


def main():
    with np.load(SOURCE, allow_pickle=False) as data:
        x, y = data["X"].astype(np.float64), data["y"]
    assert x.shape == (12000, 390) and np.isfinite(x).all()
    assert np.array_equal(np.unique(y), np.arange(300))
    assert np.all(np.bincount(y) == 40)
    u = np.concatenate([x[:, :64], x[:, 195:259]], axis=1)
    r = np.concatenate([x[:, 64:128], x[:, 259:323]], axis=1)
    mu = u - r
    features = {"u": u, "r": r, "mu": mu,
                "u+r": np.concatenate([u, r], 1),
                "u+mu": np.concatenate([u, mu], 1)}
    output = {
        "status": "exploratory_closed_set_reanalysis",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "protocol": {"players": 300, "games_per_player": 40,
                     "enrol_per_player": 30, "query_per_player": 10,
                     "standardize": "enrol-only mean and std plus 1e-9",
                     "covariance": "enrol within-player covariance, ddof=1",
                     "ridge": 0.05, "bootstrap": "5000 paired player clusters",
                     "caveat": "Repeated splits of the same players are not independent replications; not an open-set or downstream score test."},
        "splits": [],
    }
    for seed in (20261116, 20261117, 20261118):
        rng = np.random.default_rng(seed)
        enrol = np.zeros(len(y), dtype=bool)
        for player in range(300):
            enrol[rng.permutation(np.flatnonzero(y == player))[:30]] = True
        result, correct = {"seed": seed}, {}
        for name, a in features.items():
            z = (a - a[enrol].mean(0)) / (a[enrol].std(0) + 1e-9)
            prototypes = np.zeros((300, z.shape[1]))
            np.add.at(prototypes, y[enrol], z[enrol])
            prototypes /= 30
            cov = np.cov((z[enrol] - prototypes[y[enrol]]).T)
            cov += 0.05 * np.eye(z.shape[1])
            d = distances(z[~enrol], prototypes, cov)
            rank = (d < d[np.arange(len(d)), y[~enrol], None]).sum(1)
            correct[name] = rank == 0
            result[name] = {"top1": float((rank == 0).mean()),
                            "top5": float((rank < 5).mean())}
        diff = np.zeros(300)
        np.add.at(diff, y[~enrol], correct["u+r"].astype(float) - correct["u"].astype(float))
        diff /= 10
        boot_rng = np.random.default_rng(seed + 100)
        boot = diff[boot_rng.integers(0, 300, size=(5000, 300))].mean(1)
        result["ur_minus_u_player_bootstrap_95"] = np.quantile(boot, [0.025, 0.975]).tolist()
        output["splits"].append(result)
        if seed == 20261116:
            a, b = features["u+r"], features["u+mu"]
            prototypes = np.zeros((300, 256))
            np.add.at(prototypes, y[enrol], a[enrol])
            prototypes /= 30
            sd = a[enrol].std(0) + 1e-9
            cov = np.cov((a[enrol] - prototypes[y[enrol]]).T) + 0.05 * np.diag(sd**2)
            eye, zero = np.eye(128), np.zeros((128, 128))
            transform = np.block([[eye, eye], [zero, -eye]])
            d1 = distances(a[~enrol], prototypes, cov)
            d2 = distances(b[~enrol], prototypes @ transform, transform.T @ cov @ transform)
            assert np.allclose(a @ transform, b)
            assert np.allclose(d1, d2, atol=1e-7, rtol=1e-9)
            assert np.array_equal(d1.argmin(1), d2.argmin(1))
            output["transported_metric_check"] = {
                "seed": seed, "coordinate_transform_max_error": float(np.abs(a @ transform - b).max()),
                "distance_max_error": float(np.abs(d1 - d2).max()),
                "prediction_disagreements": int((d1.argmin(1) != d2.argmin(1)).sum()),
                "top1": float((d1.argmin(1) == y[~enrol]).mean()),
            }
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

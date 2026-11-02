"""Standardisation conditioned on the question size n (issue #53, architecture v2 section 8 item 5 (a)).

The question-level MLP standardises every column globally; the same column has a different spread for a 5-game question
than for a 20-game one (means over fewer games are noisier), so a noisy block weighs in for the small questions and the
mixed-n training set tunes the weights for the many-game ones (the fixed-5 training control of METHOD_NOTES 2.22 v2).
NStdMLP estimates the mean and scale of every column per group of n (5 ... 20, training players only), keeps the
`keep_global` columns (log n itself) on the global statistics, and feeds the result to MLPModel(standardise=False): same
features, model, questions, seed and training length as the control. At prediction time an n outside 5 ... 20 uses the
nearest group.
"""
import numpy as np

from aicup.rankmodel import MLPModel, _standardise

N_MIN, N_MAX = 5, 20
EPS = 1e-12


def group_of(n):
    """The group of every question size: n clipped to N_MIN ... N_MAX (the nearest group outside the range)."""
    return np.clip(np.asarray(n, dtype=int), N_MIN, N_MAX)


class NStdMLP:
    """MLPModel on per-n standardised inputs. keep_global: column indices standardised with the global statistics
    (the log n columns, constant within a group); **mlp_kw go to MLPModel (hidden, seed, ...)."""

    def __init__(self, keep_global=(), **mlp_kw):
        self.keep_global = tuple(int(c) for c in keep_global)
        self.mlp_kw = mlp_kw

    def fit(self, X, y, n, w=None):
        X, g = np.asarray(X, dtype=float), group_of(n)
        self.global_mean, self.global_std = _standardise(X)
        self.stats, self.counts, self.raw_std = {}, {}, {}
        for k in range(N_MIN, N_MAX + 1):
            m = g == k
            if m.sum() >= 2:
                mean, std = _standardise(X[m])
                raw = X[m].std(axis=0)
            else:  # no training question of this size: the global statistics
                mean, std, raw = self.global_mean.copy(), self.global_std.copy(), X.std(axis=0)
            for c in self.keep_global:
                mean[c], std[c] = self.global_mean[c], self.global_std[c]
            self.stats[k], self.counts[k], self.raw_std[k] = (mean, std), int(m.sum()), raw
        self.mlp = MLPModel(standardise=False, **self.mlp_kw).fit(self.transform(X, n), y, w)
        return self

    def transform(self, X, n):
        X, g = np.asarray(X, dtype=float), group_of(n)
        Z = np.empty_like(X)
        for k in np.unique(g):
            mean, std = self.stats[int(k)]
            Z[g == k] = (X[g == k] - mean) / std
        return Z

    def predict_proba(self, X, n):
        return self.mlp.predict_proba(self.transform(X, n))

    def group_table(self, names=None):
        """Per group: training questions, the raw standard deviation of every column (named when `names` is given) and the
        number of near-constant columns (std < EPS, excluded from the scale by _standardise)."""
        out = {}
        for k in range(N_MIN, N_MAX + 1):
            raw = self.raw_std[k]
            std = {str(i) if names is None else names[i]: float(v) for i, v in enumerate(raw)}
            out[str(k)] = {"questions": self.counts[k], "std": std, "n_constant": int((raw < EPS).sum()), "from_global": self.counts[k] < 2}
        return out

"""L3/L4 models: question features -> 10-class rank probabilities.

All models train with class-balanced weights by default (the test set's class mix is
unknown; the training set is balanced per rank by games, not by players) and expose
fit(X, y), predict_proba(X) and a JSON-able state_dict() restored by from_state().
"""
import numpy as np
import torch

from aicup.evidence import apply_thresholds, fit_thresholds

N_CLASSES = 10


def class_weights(y, n_classes=N_CLASSES):
    """Per-sample weights giving every class the same total weight, mean 1."""
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = 1.0 / counts[y]
    return w / w.mean()


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _standardise(X):
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-12] = 1.0
    return mean, std


class SoftmaxModel:
    """Multinomial logistic regression with L2 on the weights, fitted by L-BFGS in float64."""

    def __init__(self, l2=1e-3, max_iter=500, balanced=True):
        self.l2, self.max_iter, self.balanced = l2, max_iter, balanced

    def fit(self, X, y, w=None):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        w = class_weights(y) if (w is None and self.balanced) else (np.ones(len(y)) if w is None else np.asarray(w, float))
        self.mean, self.std = _standardise(X)
        Xt = torch.tensor((X - self.mean) / self.std, dtype=torch.float64)
        yt, wt = torch.tensor(y), torch.tensor(w, dtype=torch.float64)
        W = torch.zeros(X.shape[1], N_CLASSES, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(N_CLASSES, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=self.max_iter, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            ce = torch.nn.functional.cross_entropy(Xt @ W + b, yt, reduction="none")
            loss = (ce * wt).sum() / wt.sum() + self.l2 * (W ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        self.W, self.b = W.detach().numpy(), b.detach().numpy()
        return self

    def predict_proba(self, X):
        return _softmax(((np.asarray(X, dtype=float) - self.mean) / self.std) @ self.W + self.b)

    def state_dict(self):
        return {"type": "softmax", "l2": self.l2, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "W": self.W.tolist(), "b": self.b.tolist()}


class MLPModel:
    """One hidden ReLU layer, full-batch Adam in float64, fixed epochs, deterministic per seed."""

    def __init__(self, hidden=64, l2=1e-4, epochs=300, lr=1e-2, seed=0, balanced=True):
        self.hidden, self.l2, self.epochs, self.lr, self.seed, self.balanced = hidden, l2, epochs, lr, seed, balanced

    def _net(self, n_in):
        torch.manual_seed(self.seed)
        return torch.nn.Sequential(torch.nn.Linear(n_in, self.hidden), torch.nn.ReLU(),
                                   torch.nn.Linear(self.hidden, N_CLASSES)).double()

    def fit(self, X, y, w=None):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        w = class_weights(y) if (w is None and self.balanced) else (np.ones(len(y)) if w is None else np.asarray(w, float))
        self.mean, self.std = _standardise(X)
        Xt = torch.tensor((X - self.mean) / self.std, dtype=torch.float64)
        yt, wt = torch.tensor(y), torch.tensor(w, dtype=torch.float64)
        self.net = self._net(X.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            opt.zero_grad()
            ce = torch.nn.functional.cross_entropy(self.net(Xt), yt, reduction="none")
            loss = (ce * wt).sum() / wt.sum() + self.l2 * sum((p ** 2).sum() for n, p in self.net.named_parameters() if "weight" in n)
            loss.backward()
            opt.step()
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            z = self.net(torch.tensor((np.asarray(X, dtype=float) - self.mean) / self.std, dtype=torch.float64)).numpy()
        return _softmax(z)

    def state_dict(self):
        return {"type": "mlp", "hidden": self.hidden, "l2": self.l2, "epochs": self.epochs, "lr": self.lr, "seed": self.seed,
                "mean": self.mean.tolist(), "std": self.std.tolist(),
                "params": {k: v.tolist() for k, v in self.net.state_dict().items()}}


class ThresholdModel:
    """Monotone cut points on one feature column (the pilot's method), one-hot probabilities."""

    def __init__(self, feature, balanced=True):
        self.feature, self.balanced = feature, balanced

    def fit(self, X, y, w=None):
        y = np.asarray(y, dtype=int)
        w = class_weights(y) if (w is None and self.balanced) else w
        self.cuts = fit_thresholds(np.asarray(X, dtype=float)[:, self.feature], y, weights=w)
        return self

    def predict_proba(self, X):
        return np.eye(N_CLASSES)[apply_thresholds(np.asarray(X, dtype=float)[:, self.feature], self.cuts)]

    def state_dict(self):
        return {"type": "threshold", "feature": self.feature,
                "cuts": [c if np.isfinite(c) else ("inf" if c > 0 else "-inf") for c in self.cuts]}


def from_state(d):
    if d["type"] == "softmax":
        m = SoftmaxModel(l2=d["l2"])
        m.mean, m.std, m.W, m.b = (np.array(d[k], dtype=float) for k in ("mean", "std", "W", "b"))
        return m
    if d["type"] == "mlp":
        m = MLPModel(hidden=d["hidden"], l2=d["l2"], epochs=d["epochs"], lr=d["lr"], seed=d["seed"])
        m.mean, m.std = np.array(d["mean"], dtype=float), np.array(d["std"], dtype=float)
        m.net = m._net(len(m.mean))
        m.net.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in d["params"].items()})
        return m
    if d["type"] == "threshold":
        m = ThresholdModel(feature=d["feature"])
        m.cuts = np.array([float(c) for c in d["cuts"]])
        return m
    raise ValueError(f"unknown model type {d['type']!r}")

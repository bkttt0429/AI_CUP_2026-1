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


def ordinal_targets(y, eps, n_classes=N_CLASSES):
    """[N, n_classes] soft targets of ordinal label smoothing (issue #16): 1 - eps on the class, eps / 2 on each of its
    two neighbours; an edge class gives its whole eps to its one neighbour. Every row sums to 1; eps = 0 is one-hot."""
    y = np.asarray(y, dtype=int)
    if not 0.0 <= eps < 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1), got {eps}")
    T = np.zeros((len(y), n_classes))
    T[np.arange(len(y)), y] = 1.0 - eps
    if eps > 0:
        has_left, has_right = y - 1 >= 0, y + 1 < n_classes
        share_left = np.where(has_right, eps / 2, eps)
        share_right = np.where(has_left, eps / 2, eps)
        rows = np.arange(len(y))
        T[rows[has_left], y[has_left] - 1] += share_left[has_left]
        T[rows[has_right], y[has_right] + 1] += share_right[has_right]
    return T


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
    """One hidden ReLU layer, full-batch Adam in float64, fixed epochs, deterministic per seed. label_smoothing > 0
    trains on ordinal_targets' soft labels instead of the one-hot cross-entropy (the same loss when it is 0).

    Regularisation options (issue #41; the defaults leave the training loop bit-identical to before): l1 > 0 adds an L1
    penalty on the input layer's weights (sparse input selection); fit(..., val=(X_val, y_val)) evaluates the class-
    balanced cross-entropy on `val` every `eval_every` epochs, keeps the parameters of the best evaluation and stops after
    `patience` evaluations in a row without improvement (best_epoch records where it stopped; epochs is the cap)."""

    def __init__(self, hidden=64, l2=1e-4, epochs=300, lr=1e-2, seed=0, balanced=True, label_smoothing=0.0, l1=0.0):
        self.hidden, self.l2, self.epochs, self.lr, self.seed, self.balanced = hidden, l2, epochs, lr, seed, balanced
        self.label_smoothing, self.l1 = float(label_smoothing), float(l1)
        self.best_epoch = None
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        if self.l1 < 0:
            raise ValueError(f"l1 must be >= 0, got {l1}")

    def _net(self, n_in):
        torch.manual_seed(self.seed)
        return torch.nn.Sequential(torch.nn.Linear(n_in, self.hidden), torch.nn.ReLU(),
                                   torch.nn.Linear(self.hidden, N_CLASSES)).double()

    def fit(self, X, y, w=None, val=None, eval_every=10, patience=5):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        w = class_weights(y) if (w is None and self.balanced) else (np.ones(len(y)) if w is None else np.asarray(w, float))
        self.mean, self.std = _standardise(X)
        Xt = torch.tensor((X - self.mean) / self.std, dtype=torch.float64)
        yt, wt = torch.tensor(y), torch.tensor(w, dtype=torch.float64)
        self.net = self._net(X.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        soft = torch.tensor(ordinal_targets(y, self.label_smoothing), dtype=torch.float64) if self.label_smoothing > 0 else None
        if val is not None:
            Xv, yv = np.asarray(val[0], dtype=float), np.asarray(val[1], dtype=int)
            Xvt, yvt = torch.tensor((Xv - self.mean) / self.std, dtype=torch.float64), torch.tensor(yv)
            wvt = torch.tensor(class_weights(yv) if self.balanced else np.ones(len(yv)), dtype=torch.float64)
            best, best_state, bad = float("inf"), None, 0
        self.best_epoch = None
        for epoch in range(self.epochs):
            opt.zero_grad()
            if soft is None:
                ce = torch.nn.functional.cross_entropy(self.net(Xt), yt, reduction="none")
            else:
                ce = -(soft * torch.log_softmax(self.net(Xt), dim=1)).sum(dim=1)
            loss = (ce * wt).sum() / wt.sum() + self.l2 * sum((p ** 2).sum() for n, p in self.net.named_parameters() if "weight" in n)
            if self.l1 > 0:
                loss = loss + self.l1 * self.net[0].weight.abs().sum()
            loss.backward()
            opt.step()
            if val is not None and (epoch + 1) % eval_every == 0:
                with torch.no_grad():
                    cev = torch.nn.functional.cross_entropy(self.net(Xvt), yvt, reduction="none")
                    score = float((cev * wvt).sum() / wvt.sum())
                if score < best - 1e-12:
                    best, bad = score, 0
                    best_state, self.best_epoch = {k: v.detach().clone() for k, v in self.net.state_dict().items()}, epoch + 1
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if val is not None and best_state is not None:
            self.net.load_state_dict(best_state)
            self.val_ce = best
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            z = self.net(torch.tensor((np.asarray(X, dtype=float) - self.mean) / self.std, dtype=torch.float64)).numpy()
        return _softmax(z)

    def state_dict(self):
        return {"type": "mlp", "hidden": self.hidden, "l2": self.l2, "epochs": self.epochs, "lr": self.lr, "seed": self.seed,
                "label_smoothing": self.label_smoothing, "l1": self.l1, "best_epoch": self.best_epoch, "mean": self.mean.tolist(), "std": self.std.tolist(),
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
        m = MLPModel(hidden=d["hidden"], l2=d["l2"], epochs=d["epochs"], lr=d["lr"], seed=d["seed"], label_smoothing=d.get("label_smoothing", 0.0),
                     l1=d.get("l1", 0.0))
        m.best_epoch = d.get("best_epoch")
        m.mean, m.std = np.array(d["mean"], dtype=float), np.array(d["std"], dtype=float)
        m.net = m._net(len(m.mean))
        m.net.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in d["params"].items()})
        return m
    if d["type"] == "threshold":
        m = ThresholdModel(feature=d["feature"])
        m.cuts = np.array([float(c) for c in d["cuts"]])
        return m
    raise ValueError(f"unknown model type {d['type']!r}")

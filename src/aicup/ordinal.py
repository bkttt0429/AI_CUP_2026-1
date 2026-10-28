"""Ordinal alternatives to the 10-class MLP on question features (the strength-estimator comparison, option a).

Both models keep rankmodel.MLPModel's regime -- one hidden ReLU layer, full-batch Adam in float64, fixed epochs,
class-balanced weights, standardised inputs, deterministic per seed -- and its interface (fit / predict_proba /
state_dict restored by from_state), so they drop into the CV scripts and, if adopted, into the fused document.

CoralMLP (Cao, Mirjalili, Raschka 2020): the hidden layer feeds one scalar g(x) and K-1 thresholds b_0 > b_1 > ... >
b_{K-2}; P(y > k) = sigmoid(g(x) + b_k), one binary cross-entropy per threshold. The thresholds are parametrised as
decreasing (b_k = b_{k-1} - softplus(d_k)), so the cumulative probabilities are monotone and the class probabilities
P(y = k) = P(y > k-1) - P(y > k) are non-negative by construction.

PairwiseScoreMLP: the hidden layer feeds one scalar strength s(x) trained with the Bradley-Terry loss on pairs of
questions of different ranks, -log sigmoid((s_i - s_j) sign(y_i - y_j)) -- the strength estimator's objective (Chen,
Shih, Wu 2025) at the question level; every epoch pairs each question with `pairs` random partners. Class
probabilities come from a cumulative-logit link fitted afterwards on the training scores, P(y <= k) = sigmoid(theta_k
- a s) with theta increasing and a > 0, by weighted NLL; the link's thresholds start at the class-balanced quantiles
of s.
"""
import numpy as np
import torch

from aicup import rankmodel
from aicup.rankmodel import N_CLASSES, _standardise, class_weights

P_FLOOR = 1e-12


def _inverse_softplus(x):
    x = np.maximum(np.asarray(x, dtype=float), 1e-6)
    return np.where(x > 20, x, np.log(np.expm1(x)))


def _class_probs_from_exceed(c):
    """[N, K] class probabilities from P(y > k) [N, K-1] (monotone decreasing in k), clipped and renormalised."""
    p = torch.cat([1 - c[:, :1], c[:, :-1] - c[:, 1:], c[:, -1:]], dim=1)
    p = p.clamp_min(P_FLOOR)
    return p / p.sum(dim=1, keepdim=True)


class _CoralNet(torch.nn.Module):
    def __init__(self, n_in, hidden, n_classes=N_CLASSES):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
        self.b0 = torch.nn.Parameter(torch.zeros(()))
        self.d = torch.nn.Parameter(torch.zeros(n_classes - 2))

    def thresholds(self):
        """[K-1] decreasing biases b_0, b_0 - softplus(d_1), ..."""
        steps = torch.nn.functional.softplus(self.d)
        return self.b0 - torch.cat([torch.zeros(1, dtype=steps.dtype), torch.cumsum(steps, 0)])

    def forward(self, X):
        """[N, K-1] logits of P(y > k)."""
        return self.body(X)[:, 0:1] + self.thresholds()[None, :]


class CoralMLP:
    """CORAL ordinal MLP: one hidden ReLU layer, a shared scalar and K-1 decreasing thresholds; MLPModel's regime."""

    def __init__(self, hidden=32, l2=1e-4, epochs=300, lr=1e-2, seed=0, balanced=True):
        self.hidden, self.l2, self.epochs, self.lr, self.seed, self.balanced = hidden, l2, epochs, lr, seed, balanced

    def _net(self, n_in):
        torch.manual_seed(self.seed)
        return _CoralNet(n_in, self.hidden).double()

    def fit(self, X, y, w=None):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        w = class_weights(y) if (w is None and self.balanced) else (np.ones(len(y)) if w is None else np.asarray(w, float))
        self.mean, self.std = _standardise(X)
        Xt = torch.tensor((X - self.mean) / self.std, dtype=torch.float64)
        wt = torch.tensor(w, dtype=torch.float64)
        T = torch.tensor((y[:, None] > np.arange(N_CLASSES - 1)[None, :]).astype(float), dtype=torch.float64)  # 1[y > k]
        self.net = self._net(X.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            opt.zero_grad()
            bce = torch.nn.functional.binary_cross_entropy_with_logits(self.net(Xt), T, reduction="none").mean(dim=1)
            loss = (bce * wt).sum() / wt.sum() + self.l2 * sum((p ** 2).sum() for n, p in self.net.named_parameters() if "weight" in n)
            loss.backward()
            opt.step()
        return self

    def exceed_proba(self, X):
        """P(y > k) [N, K-1]."""
        with torch.no_grad():
            return torch.sigmoid(self.net(torch.tensor((np.asarray(X, dtype=float) - self.mean) / self.std, dtype=torch.float64)))

    def predict_proba(self, X):
        return _class_probs_from_exceed(self.exceed_proba(X)).numpy()

    def state_dict(self):
        return {"type": "coral", "hidden": self.hidden, "l2": self.l2, "epochs": self.epochs, "lr": self.lr, "seed": self.seed,
                "mean": self.mean.tolist(), "std": self.std.tolist(), "params": {k: v.tolist() for k, v in self.net.state_dict().items()}}


class _Link(torch.nn.Module):
    """Cumulative logit link on a scalar: P(y <= k) = sigmoid(theta_k - a s), theta increasing, a > 0."""

    def __init__(self, theta0, deltas, a):
        super().__init__()
        self.theta0 = torch.nn.Parameter(torch.tensor(float(theta0), dtype=torch.float64))
        self.d = torch.nn.Parameter(torch.tensor(_inverse_softplus(deltas), dtype=torch.float64))
        self.a_raw = torch.nn.Parameter(torch.tensor(float(_inverse_softplus(a)), dtype=torch.float64))

    def thresholds(self):
        return self.theta0 + torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(torch.nn.functional.softplus(self.d), 0)])

    def scale(self):
        return torch.nn.functional.softplus(self.a_raw) + 1e-6

    def class_proba(self, s):
        """[N, K] from scores s [N]: P(y <= k) differences."""
        cum = torch.sigmoid(self.thresholds()[None, :] - self.scale() * s[:, None])  # [N, K-1], increasing in k
        return _class_probs_from_exceed(1 - cum)


class PairwiseScoreMLP:
    """A scalar strength s(x) trained with the Bradley-Terry pairwise loss, plus a cumulative-logit link for class
    probabilities; MLPModel's regime. `pairs`: random partners per question and epoch."""

    def __init__(self, hidden=32, l2=1e-4, epochs=300, lr=1e-2, seed=0, balanced=True, pairs=4, link_epochs=300):
        self.hidden, self.l2, self.epochs, self.lr, self.seed, self.balanced = hidden, l2, epochs, lr, seed, balanced
        self.pairs, self.link_epochs = pairs, link_epochs

    def _net(self, n_in):
        torch.manual_seed(self.seed)
        return torch.nn.Sequential(torch.nn.Linear(n_in, self.hidden), torch.nn.ReLU(), torch.nn.Linear(self.hidden, 1)).double()

    def fit(self, X, y, w=None):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        w = class_weights(y) if (w is None and self.balanced) else (np.ones(len(y)) if w is None else np.asarray(w, float))
        self.mean, self.std = _standardise(X)
        Xt = torch.tensor((X - self.mean) / self.std, dtype=torch.float64)
        wt = torch.tensor(w, dtype=torch.float64)
        n = len(y)
        self.net = self._net(X.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            rng = np.random.RandomState(self.seed * 100003 + epoch)
            i = np.repeat(np.arange(n), self.pairs)
            j = rng.randint(0, n, size=len(i))
            keep = y[i] != y[j]
            i, j = i[keep], j[keep]
            if len(i) == 0:
                raise ValueError("Bradley-Terry training needs questions of at least two different ranks")
            sign = torch.tensor(np.sign(y[i] - y[j]).astype(float), dtype=torch.float64)
            it, jt = torch.tensor(i), torch.tensor(j)
            pw = torch.sqrt(wt[it] * wt[jt])
            opt.zero_grad()
            s = self.net(Xt)[:, 0]
            nll = -torch.nn.functional.logsigmoid(sign * (s[it] - s[jt]))
            loss = (nll * pw).sum() / pw.sum() + self.l2 * sum((p ** 2).sum() for nm, p in self.net.named_parameters() if "weight" in nm)
            loss.backward()
            opt.step()
        with torch.no_grad():
            s = self.net(Xt)[:, 0]
        self.link = self._fit_link(s, torch.tensor(y), wt)
        return self

    def _fit_link(self, s, yt, wt):
        s_np = s.numpy()
        q = np.quantile(s_np, [(k + 1) / N_CLASSES for k in range(N_CLASSES - 1)])
        deltas = np.maximum(np.diff(q), 1e-3)
        link = _Link(q[0], deltas, 1.0)
        opt = torch.optim.Adam(link.parameters(), lr=self.lr)
        for _ in range(self.link_epochs):
            opt.zero_grad()
            p = link.class_proba(s)
            nll = -torch.log(p[torch.arange(len(yt)), yt])
            loss = (nll * wt).sum() / wt.sum()
            loss.backward()
            opt.step()
        return link

    def score(self, X):
        """The strength score s(x) [N]."""
        with torch.no_grad():
            return self.net(torch.tensor((np.asarray(X, dtype=float) - self.mean) / self.std, dtype=torch.float64))[:, 0].numpy()

    def predict_proba(self, X):
        with torch.no_grad():
            s = torch.tensor(self.score(X), dtype=torch.float64)
            return self.link.class_proba(s).numpy()

    def state_dict(self):
        return {"type": "bt", "hidden": self.hidden, "l2": self.l2, "epochs": self.epochs, "lr": self.lr, "seed": self.seed,
                "pairs": self.pairs, "link_epochs": self.link_epochs, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "params": {k: v.tolist() for k, v in self.net.state_dict().items()},
                "link": {k: v.tolist() for k, v in self.link.state_dict().items()}}


def from_state(d):
    """Restore a CoralMLP / PairwiseScoreMLP, or delegate to rankmodel.from_state for its types."""
    if d["type"] == "coral":
        m = CoralMLP(hidden=d["hidden"], l2=d["l2"], epochs=d["epochs"], lr=d["lr"], seed=d["seed"])
        m.mean, m.std = np.array(d["mean"], dtype=float), np.array(d["std"], dtype=float)
        m.net = m._net(len(m.mean))
        m.net.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in d["params"].items()})
        return m
    if d["type"] == "bt":
        m = PairwiseScoreMLP(hidden=d["hidden"], l2=d["l2"], epochs=d["epochs"], lr=d["lr"], seed=d["seed"], pairs=d["pairs"],
                             link_epochs=d["link_epochs"])
        m.mean, m.std = np.array(d["mean"], dtype=float), np.array(d["std"], dtype=float)
        m.net = m._net(len(m.mean))
        m.net.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in d["params"].items()})
        m.link = _Link(0.0, np.ones(N_CLASSES - 2), 1.0)
        m.link.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in d["link"].items()})
        return m
    return rankmodel.from_state(d)


MODELS = {"mlp": lambda: rankmodel.MLPModel(hidden=32), "coral": lambda: CoralMLP(hidden=32), "bt": lambda: PairwiseScoreMLP(hidden=32)}

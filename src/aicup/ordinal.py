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
import numpy as np  # 數值運算
import torch  # 以 float64 張量訓練

from aicup import rankmodel  # 還原 MLPModel 等既有類型時委派
from aicup.rankmodel import N_CLASSES, _standardise, class_weights  # 與 MLPModel 共用的設定與工具

P_FLOOR = 1e-12  # 類別機率的下限，避免取對數時為 0


def _inverse_softplus(x):  # softplus 的反函數，用來設定參數初值
    x = np.maximum(np.asarray(x, dtype=float), 1e-6)  # 輸入至少 1e-6，避免 log(0)
    return np.where(x > 20, x, np.log(np.expm1(x)))  # 大值時 softplus 近似恆等，直接回傳


def _class_probs_from_exceed(c):  # 由超越機率轉成類別機率
    """[N, K] class probabilities from P(y > k) [N, K-1] (monotone decreasing in k), clipped and renormalised."""
    p = torch.cat([1 - c[:, :1], c[:, :-1] - c[:, 1:], c[:, -1:]], dim=1)  # 第一類 1 - P(y > 0)，中間相鄰相減，最後一類 P(y > K-2)
    p = p.clamp_min(P_FLOOR)  # 截掉數值誤差造成的非正值
    return p / p.sum(dim=1, keepdim=True)  # 重新正規化


class _CoralNet(torch.nn.Module):  # CORAL 網路本體
    def __init__(self, n_in, hidden, n_classes=N_CLASSES):  # 建立各層
        super().__init__()  # 初始化 Module
        self.body = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))  # 隱藏層後輸出一個共用純量 g(x)
        self.b0 = torch.nn.Parameter(torch.zeros(()))  # 第一個門檻 b_0
        self.d = torch.nn.Parameter(torch.zeros(n_classes - 2))  # 門檻間距的原始參數（經 softplus）

    def thresholds(self):  # 遞減的門檻
        """[K-1] decreasing biases b_0, b_0 - softplus(d_1), ..."""
        steps = torch.nn.functional.softplus(self.d)  # 正的間距
        return self.b0 - torch.cat([torch.zeros(1, dtype=steps.dtype), torch.cumsum(steps, 0)])  # 由 b_0 依序扣減

    def forward(self, X):  # 前向計算
        """[N, K-1] logits of P(y > k)."""
        return self.body(X)[:, 0:1] + self.thresholds()[None, :]  # 共用純量加上各門檻


class CoralMLP:  # CORAL 序數模型
    """CORAL ordinal MLP: one hidden ReLU layer, a shared scalar and K-1 decreasing thresholds; MLPModel's regime."""

    def __init__(self, hidden=32, l2=1e-4, epochs=300, lr=1e-2, seed=0, balanced=True):  # 設定超參數
        self.hidden, self.l2, self.epochs, self.lr, self.seed, self.balanced = hidden, l2, epochs, lr, seed, balanced  # 與 MLPModel 相同的超參數

    def _net(self, n_in):  # 建立網路
        torch.manual_seed(self.seed)  # 固定初始化，結果可重現
        return _CoralNet(n_in, self.hidden).double()  # float64 的 CORAL 網路

    def fit(self, X, y, w=None):  # 訓練
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)  # 特徵與標籤
        w = class_weights(y) if (w is None and self.balanced) else (np.ones(len(y)) if w is None else np.asarray(w, float))  # 未給權重時預設類別平衡
        self.mean, self.std = _standardise(X)  # 記下標準化統計量
        Xt = torch.tensor((X - self.mean) / self.std, dtype=torch.float64)  # 標準化後的特徵張量
        wt = torch.tensor(w, dtype=torch.float64)  # 權重張量
        T = torch.tensor((y[:, None] > np.arange(N_CLASSES - 1)[None, :]).astype(float), dtype=torch.float64)  # 二元目標 1[y > k]
        self.net = self._net(X.shape[1])  # 依輸入維度建立網路
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)  # 全批次 Adam
        for _ in range(self.epochs):  # 固定 epoch 數
            opt.zero_grad()  # 清除梯度
            bce = torch.nn.functional.binary_cross_entropy_with_logits(self.net(Xt), T, reduction="none").mean(dim=1)  # 每個門檻一個 BCE，對門檻取平均
            loss = (bce * wt).sum() / wt.sum() + self.l2 * sum((p ** 2).sum() for n, p in self.net.named_parameters() if "weight" in n)  # 加權平均加權重的 L2
            loss.backward()  # 反向傳播
            opt.step()  # 更新參數
        return self  # 回傳模型本身

    def exceed_proba(self, X):  # 超越機率
        """P(y > k) [N, K-1]."""
        with torch.no_grad():  # 推論不需梯度
            return torch.sigmoid(self.net(torch.tensor((np.asarray(X, dtype=float) - self.mean) / self.std, dtype=torch.float64)))  # 標準化後算 logits 再取 sigmoid

    def predict_proba(self, X):  # 預測類別機率
        return _class_probs_from_exceed(self.exceed_proba(X)).numpy()  # 超越機率相減得類別機率

    def state_dict(self):  # 可寫入 JSON 的狀態
        return {"type": "coral", "hidden": self.hidden, "l2": self.l2, "epochs": self.epochs, "lr": self.lr, "seed": self.seed,  # 模型類型與超參數
                "mean": self.mean.tolist(), "std": self.std.tolist(), "params": {k: v.tolist() for k, v in self.net.state_dict().items()}}  # 標準化統計量與網路參數


class _Link(torch.nn.Module):  # 累積 logit 連結
    """Cumulative logit link on a scalar: P(y <= k) = sigmoid(theta_k - a s), theta increasing, a > 0."""

    def __init__(self, theta0, deltas, a):  # 以初值建立參數
        super().__init__()  # 初始化 Module
        self.theta0 = torch.nn.Parameter(torch.tensor(float(theta0), dtype=torch.float64))  # 第一個門檻
        self.d = torch.nn.Parameter(torch.tensor(_inverse_softplus(deltas), dtype=torch.float64))  # 門檻間距的原始參數，經 softplus 後等於初值
        self.a_raw = torch.nn.Parameter(torch.tensor(float(_inverse_softplus(a)), dtype=torch.float64))  # 斜率 a 的原始參數

    def thresholds(self):  # 遞增的門檻 theta
        return self.theta0 + torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(torch.nn.functional.softplus(self.d), 0)])  # 由 theta0 依序加上正的間距

    def scale(self):  # 正的斜率 a
        return torch.nn.functional.softplus(self.a_raw) + 1e-6  # softplus 再加一點，保證 a > 0

    def class_proba(self, s):  # 由棋力分數得類別機率
        """[N, K] from scores s [N]: P(y <= k) differences."""
        cum = torch.sigmoid(self.thresholds()[None, :] - self.scale() * s[:, None])  # [N, K-1]，隨 k 遞增
        return _class_probs_from_exceed(1 - cum)  # 轉成超越機率後相減


class PairwiseScoreMLP:  # Bradley-Terry 成對棋力模型
    """A scalar strength s(x) trained with the Bradley-Terry pairwise loss, plus a cumulative-logit link for class
    probabilities; MLPModel's regime. `pairs`: random partners per question and epoch."""

    def __init__(self, hidden=32, l2=1e-4, epochs=300, lr=1e-2, seed=0, balanced=True, pairs=4, link_epochs=300):  # 設定超參數
        self.hidden, self.l2, self.epochs, self.lr, self.seed, self.balanced = hidden, l2, epochs, lr, seed, balanced  # 與 MLPModel 相同的超參數
        self.pairs, self.link_epochs = pairs, link_epochs  # 每題配對數與連結擬合的 epoch 數

    def _net(self, n_in):  # 建立網路
        torch.manual_seed(self.seed)  # 固定初始化，結果可重現
        return torch.nn.Sequential(torch.nn.Linear(n_in, self.hidden), torch.nn.ReLU(), torch.nn.Linear(self.hidden, 1)).double()  # 輸出一個純量棋力 s(x)

    def fit(self, X, y, w=None):  # 訓練
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)  # 特徵與標籤
        w = class_weights(y) if (w is None and self.balanced) else (np.ones(len(y)) if w is None else np.asarray(w, float))  # 未給權重時預設類別平衡
        self.mean, self.std = _standardise(X)  # 記下標準化統計量
        Xt = torch.tensor((X - self.mean) / self.std, dtype=torch.float64)  # 標準化後的特徵張量
        wt = torch.tensor(w, dtype=torch.float64)  # 權重張量
        n = len(y)  # 題目數
        self.net = self._net(X.shape[1])  # 依輸入維度建立網路
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)  # 全批次 Adam
        for epoch in range(self.epochs):  # 固定 epoch 數
            rng = np.random.RandomState(self.seed * 100003 + epoch)  # 每個 epoch 由種子決定的配對亂數
            i = np.repeat(np.arange(n), self.pairs)  # 每個題目重複 pairs 次
            j = rng.randint(0, n, size=len(i))  # 隨機的配對對象
            keep = y[i] != y[j]  # 只保留不同等級的配對
            i, j = i[keep], j[keep]  # 套用篩選
            if len(i) == 0:  # 沒有可用的配對
                raise ValueError("Bradley-Terry training needs questions of at least two different ranks")  # 至少要有兩個等級
            sign = torch.tensor(np.sign(y[i] - y[j]).astype(float), dtype=torch.float64)  # 哪一方的等級較高
            it, jt = torch.tensor(i), torch.tensor(j)  # 配對索引張量
            pw = torch.sqrt(wt[it] * wt[jt])  # 配對權重取兩題權重的幾何平均
            opt.zero_grad()  # 清除梯度
            s = self.net(Xt)[:, 0]  # 每個題目的棋力分數
            nll = -torch.nn.functional.logsigmoid(sign * (s[it] - s[jt]))  # Bradley-Terry 負對數似然
            loss = (nll * pw).sum() / pw.sum() + self.l2 * sum((p ** 2).sum() for nm, p in self.net.named_parameters() if "weight" in nm)  # 加權平均加權重的 L2
            loss.backward()  # 反向傳播
            opt.step()  # 更新參數
        with torch.no_grad():  # 擬合連結前不需梯度
            s = self.net(Xt)[:, 0]  # 訓練題目的最終棋力分數
        self.link = self._fit_link(s, torch.tensor(y), wt)  # 在訓練分數上擬合累積 logit 連結
        return self  # 回傳模型本身

    def _fit_link(self, s, yt, wt):  # 擬合分數到類別的連結
        s_np = s.numpy()  # 分數轉成 numpy
        q = np.quantile(s_np, [(k + 1) / N_CLASSES for k in range(N_CLASSES - 1)])  # s 的 10%–90% 分位數作為門檻初值
        deltas = np.maximum(np.diff(q), 1e-3)  # 門檻間距，至少 1e-3
        link = _Link(q[0], deltas, 1.0)  # 斜率初值為 1
        opt = torch.optim.Adam(link.parameters(), lr=self.lr)  # 只最佳化連結的參數
        for _ in range(self.link_epochs):  # 固定 epoch 數
            opt.zero_grad()  # 清除梯度
            p = link.class_proba(s)  # 類別機率
            nll = -torch.log(p[torch.arange(len(yt)), yt])  # 真實類別的負對數機率
            loss = (nll * wt).sum() / wt.sum()  # 加權 NLL
            loss.backward()  # 反向傳播
            opt.step()  # 更新參數
        return link  # 擬合好的連結

    def score(self, X):  # 棋力分數
        """The strength score s(x) [N]."""
        with torch.no_grad():  # 推論不需梯度
            return self.net(torch.tensor((np.asarray(X, dtype=float) - self.mean) / self.std, dtype=torch.float64))[:, 0].numpy()  # 標準化後算純量分數

    def predict_proba(self, X):  # 預測類別機率
        with torch.no_grad():  # 推論不需梯度
            s = torch.tensor(self.score(X), dtype=torch.float64)  # 棋力分數
            return self.link.class_proba(s).numpy()  # 經連結轉成類別機率

    def state_dict(self):  # 可寫入 JSON 的狀態
        return {"type": "bt", "hidden": self.hidden, "l2": self.l2, "epochs": self.epochs, "lr": self.lr, "seed": self.seed,  # 模型類型與超參數
                "pairs": self.pairs, "link_epochs": self.link_epochs, "mean": self.mean.tolist(), "std": self.std.tolist(),  # 配對設定與標準化統計量
                "params": {k: v.tolist() for k, v in self.net.state_dict().items()},  # 網路參數
                "link": {k: v.tolist() for k, v in self.link.state_dict().items()}}  # 連結參數


def from_state(d):  # 由 state_dict 還原模型
    """Restore a CoralMLP / PairwiseScoreMLP, or delegate to rankmodel.from_state for its types."""
    if d["type"] == "coral":  # CORAL 模型
        m = CoralMLP(hidden=d["hidden"], l2=d["l2"], epochs=d["epochs"], lr=d["lr"], seed=d["seed"])  # 還原超參數
        m.mean, m.std = np.array(d["mean"], dtype=float), np.array(d["std"], dtype=float)  # 標準化統計量
        m.net = m._net(len(m.mean))  # 依輸入維度建立網路
        m.net.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in d["params"].items()})  # 載入網路參數
        return m  # 還原好的模型
    if d["type"] == "bt":  # Bradley-Terry 模型
        m = PairwiseScoreMLP(hidden=d["hidden"], l2=d["l2"], epochs=d["epochs"], lr=d["lr"], seed=d["seed"], pairs=d["pairs"],  # 還原超參數
                             link_epochs=d["link_epochs"])  # 連結的 epoch 數
        m.mean, m.std = np.array(d["mean"], dtype=float), np.array(d["std"], dtype=float)  # 標準化統計量
        m.net = m._net(len(m.mean))  # 依輸入維度建立網路
        m.net.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in d["params"].items()})  # 載入網路參數
        m.link = _Link(0.0, np.ones(N_CLASSES - 2), 1.0)  # 先建立形狀正確的連結
        m.link.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in d["link"].items()})  # 再載入連結參數
        return m  # 還原好的模型
    return rankmodel.from_state(d)  # 其他類型交給 rankmodel


MODELS = {"mlp": lambda: rankmodel.MLPModel(hidden=32), "coral": lambda: CoralMLP(hidden=32), "bt": lambda: PairwiseScoreMLP(hidden=32)}  # experiments/ordinal_check.py 比較的三個最後一層

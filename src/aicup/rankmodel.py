"""L3/L4 models: question features -> 10-class rank probabilities.

All models train with class-balanced weights by default (the test set's class mix is
unknown; the training set is balanced per rank by games, not by players) and expose
fit(X, y), predict_proba(X) and a JSON-able state_dict() restored by from_state().
"""
import numpy as np  # 數值運算
import torch  # 以 float64 張量擬合模型

from aicup.evidence import apply_thresholds, fit_thresholds  # 門檻法的切點擬合與套用
from aicup.metric_loss import expected_score_loss_torch, kernel_targets  # 與官方評分對齊的損失

N_CLASSES = 10  # 等級數


def class_weights(y, n_classes=N_CLASSES):  # 類別平衡的樣本權重
    """Per-sample weights giving every class the same total weight, mean 1."""
    y = np.asarray(y, dtype=int)  # 類別索引
    counts = np.bincount(y, minlength=n_classes).astype(float)  # 各類別的樣本數
    w = 1.0 / counts[y]  # 每個樣本取其類別樣本數的倒數
    return w / w.mean()  # 正規化成平均 1


def _softmax(z):  # numpy 版 softmax
    z = z - z.max(axis=1, keepdims=True)  # 減去每列最大值，避免溢位
    e = np.exp(z)  # 取指數
    return e / e.sum(axis=1, keepdims=True)  # 每列正規化為機率


def _standardise(X):  # 輸入標準化的統計量
    mean, std = X.mean(axis=0), X.std(axis=0)  # 各欄平均與標準差
    std[std < 1e-12] = 1.0  # 常數欄不縮放，避免除以 0
    return mean, std  # 訓練與預測共用


def ordinal_targets(y, eps, n_classes=N_CLASSES):  # 序數標籤平滑的軟目標
    """[N, n_classes] soft targets of ordinal label smoothing (issue #16): 1 - eps on the class, eps / 2 on each of its
    two neighbours; an edge class gives its whole eps to its one neighbour. Every row sums to 1; eps = 0 is one-hot."""
    y = np.asarray(y, dtype=int)  # 類別索引
    if not 0.0 <= eps < 1.0:  # eps 超出範圍
        raise ValueError(f"label_smoothing must be in [0, 1), got {eps}")  # 拒絕
    T = np.zeros((len(y), n_classes))  # 軟目標矩陣
    T[np.arange(len(y)), y] = 1.0 - eps  # 本類別得 1 - eps
    if eps > 0:  # 需要分給鄰級
        has_left, has_right = y - 1 >= 0, y + 1 < n_classes  # 是否有較弱、較強的鄰級
        share_left = np.where(has_right, eps / 2, eps)  # 沒有右鄰時整個 eps 給左鄰
        share_right = np.where(has_left, eps / 2, eps)  # 沒有左鄰時整個 eps 給右鄰
        rows = np.arange(len(y))  # 列索引
        T[rows[has_left], y[has_left] - 1] += share_left[has_left]  # 加到左鄰
        T[rows[has_right], y[has_right] + 1] += share_right[has_right]  # 加到右鄰
    return T  # 每列和為 1


class SoftmaxModel:  # 多項邏輯迴歸
    """Multinomial logistic regression with L2 on the weights, fitted by L-BFGS in float64."""

    def __init__(self, l2=1e-3, max_iter=500, balanced=True):  # 設定超參數
        self.l2, self.max_iter, self.balanced = l2, max_iter, balanced  # L2 強度、迭代上限、是否類別平衡

    def fit(self, X, y, w=None):  # 擬合
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)  # 特徵與標籤
        w = class_weights(y) if (w is None and self.balanced) else (np.ones(len(y)) if w is None else np.asarray(w, float))  # 未給權重時預設類別平衡
        self.mean, self.std = _standardise(X)  # 記下標準化統計量
        Xt = torch.tensor((X - self.mean) / self.std, dtype=torch.float64)  # 標準化後的特徵張量
        yt, wt = torch.tensor(y), torch.tensor(w, dtype=torch.float64)  # 標籤與權重張量
        W = torch.zeros(X.shape[1], N_CLASSES, dtype=torch.float64, requires_grad=True)  # 權重矩陣，從 0 開始
        b = torch.zeros(N_CLASSES, dtype=torch.float64, requires_grad=True)  # 偏差
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=self.max_iter, line_search_fn="strong_wolfe",  # L-BFGS 最佳化器，強 Wolfe 線搜尋
                                tolerance_grad=1e-9, tolerance_change=1e-12)  # 收斂容差

        def closure():  # L-BFGS 每次求值時呼叫
            opt.zero_grad()  # 清除梯度
            ce = torch.nn.functional.cross_entropy(Xt @ W + b, yt, reduction="none")  # 逐樣本交叉熵
            loss = (ce * wt).sum() / wt.sum() + self.l2 * (W ** 2).sum()  # 加權平均加 L2
            loss.backward()  # 反向傳播
            return loss  # 回傳損失給 L-BFGS

        opt.step(closure)  # 執行 L-BFGS 直到收斂或達上限
        self.W, self.b = W.detach().numpy(), b.detach().numpy()  # 存成 numpy 參數
        return self  # 回傳模型本身

    def predict_proba(self, X):  # 預測類別機率
        return _softmax(((np.asarray(X, dtype=float) - self.mean) / self.std) @ self.W + self.b)  # 標準化後做線性轉換再 softmax

    def state_dict(self):  # 可寫入 JSON 的狀態
        return {"type": "softmax", "l2": self.l2, "mean": self.mean.tolist(), "std": self.std.tolist(),  # 模型類型、L2 與標準化統計量
                "W": self.W.tolist(), "b": self.b.tolist()}  # 權重與偏差


class MLPModel:  # 單隱藏層 MLP
    """One hidden ReLU layer, full-batch Adam in float64, fixed epochs, deterministic per seed. label_smoothing > 0
    trains on ordinal_targets' soft labels instead of the one-hot cross-entropy (the same loss when it is 0).

    Regularisation options (issue #41; the defaults leave the training loop bit-identical to before): l1 > 0 adds an L1
    penalty on the input layer's weights (sparse input selection); fit(..., val=(X_val, y_val)) evaluates the class-
    balanced cross-entropy on `val` every `eval_every` epochs, keeps the parameters of the best evaluation and stops after
    `patience` evaluations in a row without improvement (best_epoch records where it stopped; epochs is the cap).

    metric_kernel=True (architecture v2 section 4F) trains on metric_loss.kernel_targets, the official score's kernel normalised
    per class, in place of the one-hot or label-smoothed targets; finetune_expected_score is that section's second stage."""

    def __init__(self, hidden=64, l2=1e-4, epochs=300, lr=1e-2, seed=0, balanced=True, label_smoothing=0.0, l1=0.0, metric_kernel=False):  # 設定超參數
        self.hidden, self.l2, self.epochs, self.lr, self.seed, self.balanced = hidden, l2, epochs, lr, seed, balanced  # 基本超參數
        self.label_smoothing, self.l1, self.metric_kernel = float(label_smoothing), float(l1), bool(metric_kernel)  # 目標與正則化選項
        self.best_epoch = None  # 提早停止時保留參數的 epoch
        if not 0.0 <= self.label_smoothing < 1.0:  # 平滑量超出範圍
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")  # 拒絕
        if self.l1 < 0:  # L1 不可為負
            raise ValueError(f"l1 must be >= 0, got {l1}")  # 拒絕

    def _net(self, n_in):  # 建立網路
        torch.manual_seed(self.seed)  # 固定初始化，結果可重現
        return torch.nn.Sequential(torch.nn.Linear(n_in, self.hidden), torch.nn.ReLU(),  # 隱藏層與 ReLU
                                   torch.nn.Linear(self.hidden, N_CLASSES)).double()  # 輸出 10 類 logits，使用 float64

    def fit(self, X, y, w=None, val=None, eval_every=10, patience=5):  # 訓練
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)  # 特徵與標籤
        w = class_weights(y) if (w is None and self.balanced) else (np.ones(len(y)) if w is None else np.asarray(w, float))  # 未給權重時預設類別平衡
        self.mean, self.std = _standardise(X)  # 記下標準化統計量
        Xt = torch.tensor((X - self.mean) / self.std, dtype=torch.float64)  # 標準化後的特徵張量
        yt, wt = torch.tensor(y), torch.tensor(w, dtype=torch.float64)  # 標籤與權重張量
        self.net = self._net(X.shape[1])  # 依輸入維度建立網路
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)  # 全批次 Adam
        if self.metric_kernel:  # section 4F 的核目標
            soft = torch.tensor(kernel_targets(y), dtype=torch.float64)  # 官方分數核的軟目標
        else:  # 一般的目標
            soft = torch.tensor(ordinal_targets(y, self.label_smoothing), dtype=torch.float64) if self.label_smoothing > 0 else None  # 有平滑時用序數軟目標，否則用 one-hot
        if val is not None:  # 有驗證資料時準備提早停止
            Xv, yv = np.asarray(val[0], dtype=float), np.asarray(val[1], dtype=int)  # 驗證特徵與標籤
            Xvt, yvt = torch.tensor((Xv - self.mean) / self.std, dtype=torch.float64), torch.tensor(yv)  # 以訓練集統計量標準化
            wvt = torch.tensor(class_weights(yv) if self.balanced else np.ones(len(yv)), dtype=torch.float64)  # 驗證集的類別平衡權重
            best, best_state, bad = float("inf"), None, 0  # 最佳損失、最佳參數與連續未改善次數
        self.best_epoch = None  # 重設保留參數的 epoch
        for epoch in range(self.epochs):  # 最多 epochs 個 epoch
            opt.zero_grad()  # 清除梯度
            if soft is None:  # one-hot 目標
                ce = torch.nn.functional.cross_entropy(self.net(Xt), yt, reduction="none")  # 逐樣本交叉熵
            else:  # 軟目標
                ce = -(soft * torch.log_softmax(self.net(Xt), dim=1)).sum(dim=1)  # 對軟目標的交叉熵
            loss = (ce * wt).sum() / wt.sum() + self.l2 * sum((p ** 2).sum() for n, p in self.net.named_parameters() if "weight" in n)  # 加權平均加權重的 L2
            if self.l1 > 0:  # 啟用 L1
                loss = loss + self.l1 * self.net[0].weight.abs().sum()  # 只懲罰輸入層權重
            loss.backward()  # 反向傳播
            opt.step()  # 更新參數
            if val is not None and (epoch + 1) % eval_every == 0:  # 每 eval_every 個 epoch 評估一次
                with torch.no_grad():  # 評估不需梯度
                    cev = torch.nn.functional.cross_entropy(self.net(Xvt), yvt, reduction="none")  # 驗證集逐樣本交叉熵
                    score = float((cev * wvt).sum() / wvt.sum())  # 類別平衡的驗證交叉熵
                if score < best - 1e-12:  # 有改善
                    best, bad = score, 0  # 更新最佳值並重設計數
                    best_state, self.best_epoch = {k: v.detach().clone() for k, v in self.net.state_dict().items()}, epoch + 1  # 複製目前參數並記錄 epoch
                else:  # 沒有改善
                    bad += 1  # 累計未改善次數
                    if bad >= patience:  # 連續 patience 次沒改善
                        break  # 提早停止
        if val is not None and best_state is not None:  # 有評估過
            self.net.load_state_dict(best_state)  # 還原最佳參數
            self.val_ce = best  # 記錄最佳驗證交叉熵
        return self  # 回傳模型本身

    def finetune_expected_score(self, X, y, w=None, val=None, lr=1e-3, epochs=300, eval_every=10, patience=5):  # section 4F 第二階段的期望分微調
        """Section 4F's second stage: continue the fitted net on the negative expected official score
        (metric_loss.expected_score_loss_torch), class-balanced, same L2, full-batch Adam at `lr`. With val=(X_val, y_val) the
        class-balanced loss on val is evaluated before the first epoch and every `eval_every` epochs, the best parameters are
        kept (the start included, so the stage never leaves the net worse on val) and training stops after `patience`
        evaluations without improvement; ft_epoch records the epochs of the kept parameters (0: the start). Without val it
        runs exactly `epochs` epochs."""
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)  # 特徵與標籤
        w = class_weights(y) if (w is None and self.balanced) else (np.ones(len(y)) if w is None else np.asarray(w, float))  # 未給權重時預設類別平衡
        Xt = torch.tensor((X - self.mean) / self.std, dtype=torch.float64)  # 沿用 fit 的標準化統計量
        yt, wt = torch.tensor(y), torch.tensor(w, dtype=torch.float64)  # 標籤與權重張量
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)  # 第二階段另建 Adam，學習率用參數 lr（預設 1e-3）
        self.ft_epoch = epochs  # 沒有驗證時會跑滿 epochs
        if val is not None:  # 有驗證資料時準備提早停止
            Xv, yv = np.asarray(val[0], dtype=float), np.asarray(val[1], dtype=int)  # 驗證特徵與標籤
            Xvt, yvt = torch.tensor((Xv - self.mean) / self.std, dtype=torch.float64), torch.tensor(yv)  # 以訓練集統計量標準化
            wvt = torch.tensor(class_weights(yv) if self.balanced else np.ones(len(yv)), dtype=torch.float64)  # 驗證集的類別平衡權重

            def val_loss():  # 驗證集的類別平衡負期望分
                with torch.no_grad():  # 評估不需梯度
                    return float((expected_score_loss_torch(self.net(Xvt), yvt) * wvt).sum() / wvt.sum())  # 加權平均

            best, bad, self.ft_epoch = val_loss(), 0, 0  # 先評估起點，起點也可能被保留
            best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}  # 保存起點參數
        for epoch in range(epochs):  # 最多 epochs 個 epoch
            opt.zero_grad()  # 清除梯度
            es = expected_score_loss_torch(self.net(Xt), yt)  # 逐樣本負期望分
            loss = (es * wt).sum() / wt.sum() + self.l2 * sum((p ** 2).sum() for n, p in self.net.named_parameters() if "weight" in n)  # 加權平均加同樣的 L2
            loss.backward()  # 反向傳播
            opt.step()  # 更新參數
            if val is not None and (epoch + 1) % eval_every == 0:  # 每 eval_every 個 epoch 評估一次
                cur = val_loss()  # 目前的驗證損失
                if cur < best - 1e-12:  # 有改善
                    best, bad = cur, 0  # 更新最佳值並重設計數
                    best_state, self.ft_epoch = {k: v.detach().clone() for k, v in self.net.state_dict().items()}, epoch + 1  # 複製目前參數並記錄 epoch
                else:  # 沒有改善
                    bad += 1  # 累計未改善次數
                    if bad >= patience:  # 連續 patience 次沒改善
                        break  # 提早停止
        if val is not None:  # 有驗證資料
            self.net.load_state_dict(best_state)  # 還原最佳參數（可能是起點）
            self.val_es = best  # 記錄最佳驗證損失
        return self  # 回傳模型本身

    def predict_proba(self, X):  # 預測類別機率
        with torch.no_grad():  # 推論不需梯度
            z = self.net(torch.tensor((np.asarray(X, dtype=float) - self.mean) / self.std, dtype=torch.float64)).numpy()  # 標準化後算 logits
        return _softmax(z)  # 轉成機率

    def state_dict(self):  # 可寫入 JSON 的狀態
        return {"type": "mlp", "hidden": self.hidden, "l2": self.l2, "epochs": self.epochs, "lr": self.lr, "seed": self.seed,  # 模型類型與超參數
                "label_smoothing": self.label_smoothing, "l1": self.l1, "best_epoch": self.best_epoch, "mean": self.mean.tolist(), "std": self.std.tolist(),  # 訓練選項與標準化統計量
                "params": {k: v.tolist() for k, v in self.net.state_dict().items()}}  # 網路參數


class ThresholdModel:  # 單一特徵的門檻法
    """Monotone cut points on one feature column (the pilot's method), one-hot probabilities."""

    def __init__(self, feature, balanced=True):  # 指定特徵欄
        self.feature, self.balanced = feature, balanced  # 使用的欄與是否類別平衡

    def fit(self, X, y, w=None):  # 擬合切點
        y = np.asarray(y, dtype=int)  # 類別索引
        w = class_weights(y) if (w is None and self.balanced) else w  # 未給權重時預設類別平衡
        self.cuts = fit_thresholds(np.asarray(X, dtype=float)[:, self.feature], y, weights=w)  # 找出使加權官方分最高的單調切點
        return self  # 回傳模型本身

    def predict_proba(self, X):  # 預測（one-hot）
        return np.eye(N_CLASSES)[apply_thresholds(np.asarray(X, dtype=float)[:, self.feature], self.cuts)]  # 依切點分類後轉成 one-hot

    def state_dict(self):  # 可寫入 JSON 的狀態
        return {"type": "threshold", "feature": self.feature,  # 模型類型與特徵欄
                "cuts": [c if np.isfinite(c) else ("inf" if c > 0 else "-inf") for c in self.cuts]}  # 無窮大的切點轉成字串，JSON 才能保存


def from_state(d):  # 由 state_dict 還原模型
    if d["type"] == "softmax":  # 多項邏輯迴歸
        m = SoftmaxModel(l2=d["l2"])  # 建立空模型
        m.mean, m.std, m.W, m.b = (np.array(d[k], dtype=float) for k in ("mean", "std", "W", "b"))  # 還原標準化統計量與參數
        return m  # 還原好的模型
    if d["type"] == "mlp":  # MLP 模型
        m = MLPModel(hidden=d["hidden"], l2=d["l2"], epochs=d["epochs"], lr=d["lr"], seed=d["seed"], label_smoothing=d.get("label_smoothing", 0.0),  # 還原超參數（舊檔沒有平滑欄時取 0）
                     l1=d.get("l1", 0.0))  # 舊檔沒有 l1 時取 0
        m.best_epoch = d.get("best_epoch")  # 保留參數（最佳評估）時的 epoch；舊檔沒有時為 None
        m.mean, m.std = np.array(d["mean"], dtype=float), np.array(d["std"], dtype=float)  # 標準化統計量
        m.net = m._net(len(m.mean))  # 依輸入維度建立網路
        m.net.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in d["params"].items()})  # 載入網路參數
        return m  # 還原好的模型
    if d["type"] == "threshold":  # 門檻法
        m = ThresholdModel(feature=d["feature"])  # 指定特徵欄
        m.cuts = np.array([float(c) for c in d["cuts"]])  # 字串 inf 轉回浮點數
        return m  # 還原好的模型
    raise ValueError(f"unknown model type {d['type']!r}")  # 不認得的模型類型

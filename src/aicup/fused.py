"""The fused (Human SL + L1b) question model of the submission (issue #9) and its opponent extension (issue #12).

A question's fused features are qfeatures.question_features of its games' 18 Human SL sums (24) followed by
the same features of the games' 10 L1b-head sums (16) -- l1b_cv.feature_sets' "fused", column for column.
The fused_opp features append the same 16 features of the L1b sums over the *opponent's* moves of the same
games, with the opponent's move counts (l1b_score.py --opponent; l1b_opp_check.features' "fused_opp"): online
games pair players of similar ranks, so the opponent's moves are a second look at the target's rank. The
fused_opp_diff features (issue #25) append three more: over the question's games the mean of (opponent pmi -
own pmi), the mean and the std of the opponent pmi, pmi being the posterior-mean rank index of a game's ten
L1b sums -- the server pairs players by their labels, so the opponents' strength relative to the target's own
play carries label-side information the 56 features left on the table (experiments/error_analysis.py, +0.018).
The fused MLP (rankmodel.MLPModel(hidden=32), fitted by scripts/fit_fused_model.py exactly as
experiments/l1b_fold0_check.py / l1b_opp_fold0_check.py did) is stored as a rank_model.json-like document that
also names the L1b models whose scores produced its training features (heads file, state and trunk sha256,
training folds, the folds they gave features for) and says whether it takes the opponent features. At
prediction time every L1b model of the document scores the question's games (and their opponents), the MLP
runs once per model on that model's features, and the probabilities are averaged (the rule the fold-0
checks validated); pairing_order makes sure the shards' L1b models are exactly the document's.
"""
import json  # 讀取模型文件
from pathlib import Path  # 路徑處理

import numpy as np  # 數值陣列

from aicup.handicap import HD_FEATURE_NAMES  # 讓子題目特徵的欄名
from aicup.evidence import posterior_mean_index  # pmi：後驗平均等級索引
from aicup.humansl import PROFILES  # 18 個 Human SL 剖面
from aicup.ladder import environment, to_json  # 程式版本與執行環境的紀錄、嚴格 JSON 輸出
from aicup.metric import LEVELS  # 10 個等級
from aicup.qfeatures import feature_names, question_features, sum_cols  # 題目層特徵的定義
from aicup.rankmodel import MLPModel, from_state  # 題目層 MLP 與由 state 還原模型
from aicup.setmodel_io import file_sha256  # noqa: F401  重新匯出，供呼叫端與測試使用

NAME = "mlp_h32_fused"  # fused 模型名稱
NAME_OPP = "mlp_h32_fused_opp"  # 加上對手特徵的模型名稱
NAME_OPP_DIFF = "mlp_h32_fused_opp_diff"  # 再加差值特徵的模型名稱
NAME_OPP_DIFF_HD = "mlp_h32_fused_opp_diff_hd"  # 再加讓子特徵的模型名稱
FORMAT = 1  # 文件格式版本
FUSED_FEATURE_NAMES = feature_names(PROFILES) + [f"l1b_{n}" for n in feature_names(LEVELS)]  # 40 欄：Human SL 24 欄加 L1b 16 欄
OPP_FEATURE_NAMES = [f"opp_{n}" for n in feature_names(LEVELS)]  # 對手落子的 L1b 16 欄
FUSED_OPP_FEATURE_NAMES = FUSED_FEATURE_NAMES + OPP_FEATURE_NAMES  # 56 欄
DIFF_FEATURE_NAMES = ["opp_minus_own_pmi_mean", "opp_pmi_mean", "opp_pmi_std"]  # 3 欄：對手減本人 pmi 的平均、對手 pmi 的平均與標準差
FUSED_OPP_DIFF_FEATURE_NAMES = FUSED_OPP_FEATURE_NAMES + DIFF_FEATURE_NAMES  # 59 欄
FUSED_OPP_DIFF_HD_FEATURE_NAMES = FUSED_OPP_DIFF_FEATURE_NAMES + HD_FEATURE_NAMES  # 再加讓子題目特徵（issue #47），共 64 欄
FEATURE_SETS = {"fused": FUSED_FEATURE_NAMES, "fused_opp": FUSED_OPP_FEATURE_NAMES, "fused_opp_diff": FUSED_OPP_DIFF_FEATURE_NAMES,  # 特徵組名稱 -> 欄名
                "fused_opp_diff_hd": FUSED_OPP_DIFF_HD_FEATURE_NAMES}  # 含讓子特徵的 64 欄組
NAMES = {"fused": NAME, "fused_opp": NAME_OPP, "fused_opp_diff": NAME_OPP_DIFF, "fused_opp_diff_hd": NAME_OPP_DIFF_HD}  # 特徵組 -> 模型名稱
DIFF_SETS = ("fused_opp_diff", "fused_opp_diff_hd")  # 含差值特徵的特徵組
MODEL_KEYS = ("name", "heads_file", "state_sha256", "trunk_file", "trunk_sha256", "train_folds", "feature_folds")  # 每筆 L1b 模型紀錄必備的欄位


def fused_features(hsl_sums, l1b_sums, n_moves):  # 組出 40 欄的 fused 特徵
    """[40]: question_features of the games' Human SL sums [m, 18], then of their L1b sums [m, 10], both with n_moves [m]."""
    return np.concatenate([question_features(hsl_sums, n_moves), question_features(l1b_sums, n_moves)])  # 先 Human SL 24 欄，再 L1b 16 欄


def fused_opp_features(hsl_sums, l1b_sums, n_moves, opp_sums, opp_n_moves):  # 組出 56 欄的 fused_opp 特徵
    """[56]: fused_features, then question_features of the L1b sums over the opponent's moves [m, 10] with the
    opponent's move counts [m] (l1b_opp_check.features' "fused_opp", column for column)."""
    return np.concatenate([fused_features(hsl_sums, l1b_sums, n_moves), question_features(opp_sums, opp_n_moves)])  # 再接對手的 16 欄


def diff_features(l1b_sums, opp_sums):  # 3 個 pmi 特徵：對手減本人的平均、對手的平均與標準差
    """[3]: over a question's games, mean(opponent pmi - own pmi), mean(opponent pmi), std(opponent pmi); pmi is
    posterior_mean_index of a game's ten L1b sums [m, 10] (experiments/error_analysis.py's mlp_diff features, column for column)."""
    own = posterior_mean_index(np.ascontiguousarray(l1b_sums, dtype=float))  # 本人每局的 pmi
    opp = posterior_mean_index(np.ascontiguousarray(opp_sums, dtype=float))  # 對手每局的 pmi
    return np.array([(opp - own).mean(), opp.mean(), opp.std()])  # 依 DIFF_FEATURE_NAMES 的順序排出 3 欄


def fused_opp_diff_features(hsl_sums, l1b_sums, n_moves, opp_sums, opp_n_moves):  # 組出 59 欄的特徵
    """[59]: fused_opp_features, then diff_features of the target's and the opponent's L1b sums."""
    return np.concatenate([fused_opp_features(hsl_sums, l1b_sums, n_moves, opp_sums, opp_n_moves), diff_features(l1b_sums, opp_sums)])  # 56 欄後接 3 個 pmi 特徵


def question_diff_features(l1b, opp, questions):  # 一次算出所有題目的差值特徵
    """[Q, 3]: diff_features of every question from l1b_score.py tables of the target's (l1b) and the opponent's (opp) sums."""
    cols = sum_cols(LEVELS)  # 10 個等級的總和欄名
    li, oi = l1b.set_index("game_key"), opp.set_index("game_key")  # 兩張表都以局鍵為索引
    L, O = li[cols].to_numpy(dtype=float), oi[cols].to_numpy(dtype=float)  # 本人與對手的總和矩陣
    row_l, row_o = {k: i for i, k in enumerate(li.index)}, {k: i for i, k in enumerate(oi.index)}  # 局鍵 -> 列號
    rows = [diff_features(L[[row_l[k] for k in ks]], O[[row_o[k] for k in ks]]) for ks in questions["game_keys"]]  # 逐題計算
    return np.stack(rows) if rows else np.zeros((0, len(DIFF_FEATURE_NAMES)))  # 沒有題目時回傳空矩陣


def feature_set_of(opponent=False, feature_set=None):  # 決定特徵組名稱
    """The feature set name: `feature_set` when given, else fused_opp / fused from the older opponent flag."""
    if feature_set is None:  # 沒有指定特徵組
        feature_set = "fused_opp" if opponent else "fused"  # 相容舊的旗標
    if feature_set not in FEATURE_SETS:  # 未知的名稱
        raise ValueError(f"unknown feature set {feature_set!r}; one of {sorted(FEATURE_SETS)}")  # 列出可用的特徵組
    return feature_set  # 回傳名稱


def feature_names_for(feature_set):  # 取得特徵組的欄名
    """The feature names of a set given by name, or by the older opponent flag (True: fused_opp, False: fused)."""
    if isinstance(feature_set, (bool, np.bool_)):  # 傳入的是舊的布林旗標
        feature_set = "fused_opp" if feature_set else "fused"  # 轉成名稱
    return list(FEATURE_SETS[feature_set_of(feature_set=feature_set)])  # 回傳欄名清單的副本


def l1b_model_record(name, heads_file, sidecar, feature_folds):  # 文件中一個 L1b 模型的紀錄
    """The document's record of one L1b model: its heads file name, the sidecar's sha256s and training folds
    (aicup.l1b.save_heads), and the folds whose features it supplied."""
    return {"name": name, "heads_file": Path(heads_file).name, "state_sha256": sidecar["state_sha256"],  # 名稱、檔名與等級頭的 sha256
            "trunk_file": sidecar.get("trunk_file"), "trunk_sha256": sidecar.get("trunk_sha256"),  # 微調後的主幹（若有）
            "train_folds": sidecar.get("train_folds"), "feature_folds": [int(f) for f in feature_folds]}  # 訓練折與提供特徵的折


def fused_document(mlp, l1b_models, train_folds, per_player, n_train, questions_sha256, extra=None, opponent=False, feature_set=None):  # 組出融合模型文件
    """The rank_model_fused[_opp[_diff]].json document: name, features (40 / 56 / 59), feature_set, the opponent flag
    (True for every set but "fused"), profiles, levels, training provenance, l1b_models, state."""
    for m in l1b_models:  # 檢查每筆 L1b 模型紀錄
        missing = [k for k in MODEL_KEYS if k not in m]  # 缺少的欄位
        if missing:  # 有欄位缺少
            raise ValueError(f"l1b model record {m.get('name')!r} lacks {missing}")  # 拒絕不完整的紀錄
    feature_set = feature_set_of(opponent, feature_set)  # 決定特徵組
    opponent = feature_set != "fused"  # 除了 fused 以外都使用對手特徵
    what = ("fused Human SL + L1b question features" + (" + the L1b features of the opponent's moves" if opponent else "")  # 用途說明的開頭
            + (" + the opponent-minus-own pmi differences" if feature_set in DIFF_SETS else "")  # 有差值特徵時補上
            + (" + the handicap (HD) question features" if feature_set == "fused_opp_diff_hd" else ""))  # 有讓子特徵時補上
    return {"name": NAMES[feature_set], "format": FORMAT, "features": feature_names_for(feature_set), "opponent": bool(opponent), "feature_set": feature_set,  # 模型名稱、格式與特徵
            "profiles": list(PROFILES), "levels": list(LEVELS),  # 剖面與等級，載入時核對
            "train_folds": [int(f) for f in train_folds], "per_player": int(per_player), "train_questions": int(n_train),  # 訓練折、每人題數與訓練題數
            "questions_sha256": questions_sha256, "l1b_models": [dict(m) for m in l1b_models],  # 訓練題的雜湊與 L1b 模型清單
            "role": f"the submission's decision model: MLP on {what}, averaged over the L1b models",  # 文件的用途說明
            **(extra or {}), **environment("cpu"), "state": mlp.state_dict()}  # 額外欄位、程式版本與執行環境、MLP 權重


def save_fused_model(doc, path):  # 寫出融合模型文件
    path = Path(path)  # 統一成 Path
    path.parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    path.write_text(to_json(doc, indent=1) + "\n")  # 寫成 JSON
    return path  # 回傳路徑


def load_fused_model(path):  # 讀入並核對融合模型文件
    """(document with file_sha256, feature_set and opponent added, MLPModel). The features decide the kind: the 40 fused
    names (opponent False), the 56 fused_opp names or the 59 fused_opp_diff names (opponent True); ValueError when they
    are neither, when the document's own opponent flag or feature_set disagrees, when profiles / levels are not this
    code's, or when it names no L1b model."""
    path = Path(path)  # 統一成 Path
    doc = json.loads(path.read_text())  # 讀入文件
    feats = doc.get("features")  # 文件記錄的欄名
    feature_set = next((k for k, names in FEATURE_SETS.items() if feats == names), None)  # 找出欄名完全相同的特徵組
    if feature_set is None:  # 沒有相符的特徵組
        sets = [f"{k} ({len(v)})" for k, v in FEATURE_SETS.items()]  # 各特徵組的名稱與欄數，供錯誤訊息使用
        raise ValueError(f"{path}: features are neither this code's {', '.join(sets[:-1])} nor {sets[-1]} features ({len(feats or [])} given)")  # 拒絕未知的特徵
    opponent = feature_set != "fused"  # 由特徵推得的對手旗標
    if doc.get("opponent", opponent) != opponent:  # 與文件記錄的旗標不符
        raise ValueError(f"{path}: opponent flag {doc.get('opponent')} disagrees with its {len(feats)} features")  # 拒絕矛盾的文件
    if doc.get("feature_set", feature_set) != feature_set:  # 與文件記錄的特徵組不符
        raise ValueError(f"{path}: feature_set {doc.get('feature_set')!r} disagrees with its {len(feats)} features ({feature_set})")  # 拒絕矛盾的文件
    if doc.get("profiles") != list(PROFILES) or doc.get("levels") != list(LEVELS):  # 剖面或等級的定義不同
        raise ValueError(f"{path}: profiles or levels differ from this code's")  # 拒絕
    if not doc.get("l1b_models"):  # 沒有列出 L1b 模型
        raise ValueError(f"{path}: l1b_models is empty; the fused model must name the L1b models its features come from")  # 拒絕
    model = from_state(doc["state"])  # 還原 MLP
    if not isinstance(model, MLPModel):  # 權重不是 MLP
        raise ValueError(f"{path}: state is not an MLP")  # 拒絕
    return {**doc, "opponent": opponent, "feature_set": feature_set, "file_sha256": file_sha256(path)}, model  # 回傳補上欄位的文件與模型


def pairing_order(doc, shard_models):  # 把文件的 L1b 模型對應到分片的模型
    """For each L1b model of the fused document, the index of the shards' model with the same state and trunk sha256.
    ValueError when a document model is missing from the shards, a shard model is not in the document, or the shards
    carry no L1b scores at all."""
    if not shard_models:  # 分片沒有 L1b 分數
        raise ValueError("the shards carry no L1b scores (the shard step needs --l1b-heads for a fused decision)")  # 提示分片步驟要加 --l1b-heads
    key = lambda m: (m.get("state_sha256"), m.get("trunk_sha256"))  # noqa: E731  以等級頭與主幹的 sha256 作為配對鍵
    where = {}  # 配對鍵 -> 分片模型的索引
    for i, m in enumerate(shard_models):  # 建立查詢表
        where.setdefault(key(m), []).append(i)  # 同一個鍵可能出現多次
    order = []  # 文件中各模型對應的分片索引
    for m in doc["l1b_models"]:  # 依文件的順序配對
        hits = where.get(key(m), [])  # 相符的分片模型
        if not hits:  # 找不到
            same_heads = [s for s in shard_models if s.get("state_sha256") == m.get("state_sha256")]  # 看看是否只有主幹不同
            why = "its trunk sha256 differs" if same_heads else "it is not among the shards' L1b models"  # 錯誤原因
            raise ValueError(f"fused model's L1b model {m['name']} ({m.get('heads_file')}): {why}")  # 拒絕
        if len(hits) > 1:  # 在分片中重複出現
            raise ValueError(f"fused model's L1b model {m['name']} appears {len(hits)} times among the shards' models")  # 拒絕
        order.append(hits[0])  # 記下對應的索引
    extra = [s.get("name") for i, s in enumerate(shard_models) if i not in order]  # 分片多出來的模型
    if extra:  # 分片的模型須與文件完全一致
        raise ValueError(f"shard L1b models {extra} are not in the fused model; the shards must be scored with exactly its models")  # 拒絕
    return order  # 回傳配對順序


def fused_predict(mlp, hsl_sums, l1b_sums, n_moves, order, opp_sums=None, opp_n_moves=None, feature_set=None, hd_feats=None):  # 一題的融合預測
    """[10]: the mean over the document's models (shard columns `order`) of the MLP's probabilities on the features of
    `feature_set` (default: fused_opp when the opponent's L1b sums [m, K, 10] and move counts [m] are given, else
    fused); fused_opp_diff adds the opponent-minus-own differences, fused_opp_diff_hd also the question's handicap
    features hd_feats [5] (aicup.handicap.question_hd_features_from_games), the same for every model. ValueError when
    the set needs opponent sums or handicap features and none are given."""
    feature_set = feature_set_of(opp_sums is not None, feature_set)  # 決定特徵組
    if feature_set == "fused":  # 不使用對手特徵
        feats = [fused_features(hsl_sums, l1b_sums[:, k, :], n_moves) for k in order]  # 每個 L1b 模型各一組特徵
    else:  # 需要對手特徵
        if opp_sums is None:  # 缺少對手的總和
            raise ValueError(f"the {feature_set} features need the opponent's L1b sums")  # 拒絕
        fn = fused_opp_diff_features if feature_set in DIFF_SETS else fused_opp_features  # 依特徵組選擇函式
        feats = [fn(hsl_sums, l1b_sums[:, k, :], n_moves, opp_sums[:, k, :], opp_n_moves) for k in order]  # 每個 L1b 模型各一組特徵
        if feature_set == "fused_opp_diff_hd":  # 還需要讓子特徵
            if hd_feats is None:  # 缺少讓子特徵
                raise ValueError("the fused_opp_diff_hd features need the question's handicap features (hd_feats)")  # 拒絕
            hd = np.asarray(hd_feats, dtype=float).reshape(-1)  # 攤平成一維
            if hd.shape != (len(HD_FEATURE_NAMES),):  # 必須剛好 5 欄
                raise ValueError(f"hd_feats must hold the {len(HD_FEATURE_NAMES)} handicap features, got shape {hd.shape}")  # 拒絕
            feats = [np.concatenate([x, hd]) for x in feats]  # 每組特徵後面接同一份讓子特徵
    return np.mean([mlp.predict_proba(x[None, :])[0] for x in feats], axis=0)  # 各模型的機率取平均

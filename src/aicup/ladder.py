"""Cross-validation of the L3 aggregator ladder on folds 1-4, its pooled paired summary and the gates.

Protocol (plan 2026-10-19-l3-aggregator-ladder-ctm, Global Constraints):
- Leave one fold out over the CV folds (1-4). Fold 0 is refused everywhere here: it is evaluated
  once, for the chosen rung only, by experiments/ladder_eval_fold0.py.
- Validation questions of split v: pilot.pilot_questions of fold v's scoring.fold_players selection
  (nested 1x20 / 2x10 / 4x5 carvings).
- Training questions of split v: rng = RandomState(42 + v) made once, then qfeatures.sample_questions
  per training fold in ascending order -- exactly fit_rank_model.train_set, so r2_mlp24 at
  per_player=8 reproduces results/rank_model_eval.json. Every rung and seed of v gets the same ones.
- Static rungs (ladder_data.STATIC_RUNGS: kind, rep, make_model) fit make_model() on the stacked
  training features. Set rungs (SET_RUNGS: rep, make(d_in), TrainCfg overrides) train a correction Δ
  to cross-fitted base logits (settrain.crossfit_base_logits with the r2_mlp24 model on q24
  features); the validation base is the r2_mlp24 static rung of the split, which split_base checks.
  Set rungs stop early on inner validation players held out of the training questions
  (settrain.inner_players: 10% of each rank's players, the same for every rung and seed of a split);
  the validation fold only scores.
- Scores are taken at the tick of the rung's TrainCfg.pick: the final tick for every rung but
  r5_ctm_minmax_pc, which trains the original CTM's two-tick loss and, as the original evaluates, selects
  (inner early stopping) and predicts at each question's most certain tick. EXTRAS (more ticks at
  inference, another tick pick) are extra columns of the same trained model and never enter the gates.
- The 2026-10-24 follow-up (issue #2, item 3): the probability-mean ensemble ens_top3 of the three best
  set rungs (build_ensemble, experiments/ladder_ensemble.py), r5_ctm_minmax_pc (the r5_ctm_minmax model and
  loss selected at the tick the loss trains; the recorded r5_ctm_minmax stays as it was) and the U-Net
  synapse r5_ctm_unet are each compared with r5_ctm and with r3_pma (FOLLOWUP_COMPARISONS),
  pre-specified, reported and never gated.

Files: run_rung_cv returns one frame per (rung, seed) -- question_id, fold, player_id, y, n_games,
p0..p9, score, plus p0@X..p9@X, score@X for each extra X -- with its settings, provenance (code
version, torch, device, TrainCfg, parameter count) and per-split diagnostics in DataFrame.attrs; a
non-finite set-model loss or probability raises instead. save_cv writes it as
<cv_dir>/cv_filename(...) (fixed-width unicode strings, attrs as a JSON string, loadable with
allow_pickle=False); split_base caches the base logits as <cv_dir>/base_v{v}[_pp{N}][_nocounts].npz.

summarize: per label (rung[@ppN][@nocounts][@extra]) the per-question score is the mean over seeds
first, then every metric is the class-balanced mean per question size averaged over sizes 5/10/20,
per validation fold and on the folds pooled. Comparisons are paired on the pooled questions:
qeval.paired_diff (player bootstrap, n=5000), per-fold and per-seed point differences and the
Nadeau-Bengio corrected interval (variance x 2.33 = 1 + J n_test / n_train for J = 4 folds) as a
conservative reference. Labels whose seeds were trained by different code (code_sha256) are listed
in mixed_versions, a warning only. gates: every pre-specified comparison of a gate must meet all of
CONDITIONS, on a summary bootstrapped with at least N_BOOT replicates.
"""
import hashlib  # 題目與程式檔的雜湊
import json  # 讀寫摘要與 attrs
import math  # 開根號與有限值判斷
import os  # 原子取代檔案
import re  # 比對 CV 結果檔名
import subprocess  # 呼叫 git 取得版本
import time  # 計時
from dataclasses import asdict, replace  # TrainCfg 的轉換與覆寫
from datetime import datetime  # 摘要的建立時間
from pathlib import Path  # 路徑處理

import numpy as np  # 數值運算
import pandas as pd  # 題目與結果資料表
import torch  # 裝置名稱與版本

from aicup.aggregators import CTM, DeepSets, PooledDelta, QglobDelta, SetTransformer, XAttnGRU, n_params  # 集合梯級的聚合器
from aicup.ladder_data import STATIC_RUNGS, fit_clip_k, pack_questions, static_features  # 題目打包與靜態梯級
from aicup.metric import LEVELS, RANK_TO_IDX  # 等級定義
from aicup.pilot import pilot_questions  # 驗證題目的切法
from aicup.qeval import SIZES, balanced, balanced_mean_over_n, evaluate, nll_brier, paired_diff  # 題目層評估
from aicup.qfeatures import sample_questions  # 訓練題目的抽樣
from aicup.rankmodel import N_CLASSES, _softmax, class_weights  # 類別數、softmax 與類別權重
from aicup.scoring import TRAIN_DIR, fold_players, player_games  # 訓練資料與各折玩家
from aicup.settrain import BASE_CLIP, TrainCfg, crossfit_base_logits, fit_set_model, predict_set_model  # 集合模型的訓練與預測

ROOT = Path(__file__).resolve().parents[2]  # 儲存庫根目錄
FOLDS_CSV = ROOT / "splits" / "player_folds.csv"  # 玩家分折表（最終流程也讀這個路徑）
CV_FOLDS = (1, 2, 3, 4)  # CV 使用的折，第 0 折除外
DEFAULT_PER_PLAYER = 32  # 每位訓練玩家抽的題數
TRAIN_SEED = 42  # 切分 v 的訓練題目：RandomState(TRAIN_SEED + v)，與 fit_rank_model 相同
CHUNK = 8192  # 靜態特徵每次打包的題數
BASE_RUNG = "r2_mlp24"  # 當時的提交模型：每個集合梯級的基準，也是每個比較的參考
BASE_TOL = 1e-12  # 驗證基準與靜態梯級機率的容許差
KEY_COLS = ["question_id", "fold", "player_id", "y", "n_games"]  # 結果資料表的鍵欄
# 集合梯級內層提早停止的逐切分 attrs（run_rung_cv）
INNER_KEYS = ("best_epoch", "inner_ce_init", "inner_ce_best", "stopped_early", "n_inner_players")  # 內層提早停止記錄的欄位

# 梯級 -> (向量表示, make(d_in), TrainCfg 覆寫值)；預設大小（Tasks 4-6）：XAttnGRU d_hidden 256
# 在 d_in 301 時有 458,250 個參數，CTM 有 437,514 個（比值 1.047），所以不需要調整大小。
# r3_pma 與 r4_xattn_t1 是只報告的歸因對照（REPORT_ONLY，永不作為門檻）：r3_settf
# 拿掉 SAB（對編碼後的各局做 PMA 池化，局與局之間沒有交互；258,058 個參數），
# 以及訓練與預測都只用一個 tick 的 r4_xattn（458,250 個參數）。
# r5_ctm_minmax 以原始 CTM 的 min-loss / max-certainty 損失訓練，卻在最後一個 tick 選模（內層 CE）與評分，
# 而該損失並不訓練最後一個 tick：每次記錄的執行都保留第 0 次評估（Δ ≡ 0），即
# METHOD_NOTES §2.7 的已知限制；它維持記錄時的樣子。r5_ctm_minmax_pc（2026-10-24）是同一個
# 模型與損失，改在最確定的 tick 上選模與評分（pick "certain"），與原始實作的評估方式相同。
# r5_ctm_unet 是把 synapse 換成原始實作預設的 depth-4 U-Net 的 r5_ctm（synapse_depth=4；467,898
# 個參數）。
SET_RUNGS = {  # 集合梯級的定義
    "r2_pooled_delta": ("staged", lambda d: PooledDelta(d), {}),  # 標準化 token 的題內平均 → Δ，沒有每局編碼器
    "r3_null": ("staged", lambda d: QglobDelta(d), {}),  # Δ 只看題目層全域輸入（純再校準）
    "r3_deepsets": ("staged", lambda d: DeepSets(d), {}),  # (3) 每局編碼器 → mean + max
    "r3_settf": ("staged", lambda d: SetTransformer(d), {}),  # (3) 每局編碼器 → 2 層 SAB → PMA
    "r3_deepsets_whole": ("whole", lambda d: DeepSets(d), {}),  # DeepSets 改用整局 token
    "r4_xattn": ("staged", lambda d: XAttnGRU(d, d_hidden=256), {}),  # (4) cross-attention + GRU，跑 8 個 tick
    "r5_ctm": ("staged", lambda d: CTM(d), {}),  # (5) CTM，跑 8 個 tick
    "r5_ctm_minmax": ("staged", lambda d: CTM(d), {"loss": "ctm_minmax"}),  # CTM 改用原始的雙 tick 損失
    "r3_pma": ("staged", lambda d: SetTransformer(d, layers=0), {}),  # 歸因對照：拿掉 SAB，只留 PMA
    "r4_xattn_t1": ("staged", lambda d: XAttnGRU(d, d_hidden=256, ticks=1), {}),  # 歸因對照：只跑 1 tick
    "r5_ctm_unet": ("staged", lambda d: CTM(d, synapse_depth=4), {}),  # synapse 換成 depth-4 U-Net
    "r5_ctm_minmax_pc": ("staged", lambda d: CTM(d), {"loss": "ctm_minmax", "pick": "certain"}),  # 雙 tick 損失，在最確定的 tick 選模與預測
}  # 結束集合梯級定義
RUNGS = STATIC_RUNGS | SET_RUNGS  # 所有可訓練的梯級
# 同一個訓練好的模型只報告的預測：名稱 -> predict_set_model 的關鍵字參數
EXTRAS = {  # 額外預測的定義
    "r4_xattn": {"T16": {"ticks": 16}},  # 推論時跑 16 ticks
    "r5_ctm": {"T16": {"ticks": 16}},  # 推論時跑 16 ticks
    "r5_ctm_minmax": {"T16": {"ticks": 16}, "certain": {"pick": "certain"}},  # 16 ticks，以及改取最確定的 tick
    "r5_ctm_unet": {"T16": {"ticks": 16}},  # 推論時跑 16 ticks
    "r5_ctm_minmax_pc": {"T16": {"ticks": 16}, "final": {"pick": "final"}},  # 16 ticks，以及改取最後一個 tick
}  # 結束額外預測定義
# 2026-10-24 的追加項（issue #2，待辦清單第 3 項）：ens_top3（ENSEMBLE_MEMBERS 的機率平均集成，
# 由 build_ensemble 從它們的 CV 檔建立；experiments/ladder_ensemble.py；不是可訓練的梯級）、
# r5_ctm_minmax_pc 與 r5_ctm_unet，各自對 r5_ctm 與 r3_pma——比較組在執行前就已固定；
# 列在 REPORT_ONLY，永不作為門檻
ENSEMBLE_RUNG = "ens_top3"  # 集成的標籤
ENSEMBLE_MEMBERS = ("r3_pma", "r4_xattn_t1", "r5_ctm")  # CV 前三名的集合梯級
FOLLOWUP_COMPARISONS = [(a, b) for a in (ENSEMBLE_RUNG, "r5_ctm_minmax_pc", "r5_ctm_unet") for b in ("r5_ctm", "r3_pma")]  # 追加項的六組比較

# 門檻：事先指定的 (a, b) 比較，即「a 優於 b」；每一組都必須符合所有條件
GATES = {  # 四個門檻
    "G_rep": [("r2_staged", "r2_whole"), ("r3_deepsets", "r3_deepsets_whole")],  # 分段表示是否優於整局
    "G_set": [("r3_settf", "r3_deepsets"), ("r3_settf", "r2_pooled_delta")],  # 局與局的交互是否有益
    "G_iter": [("r4_xattn", "r3_settf"), ("r5_ctm", "r3_settf")],  # 反覆讀取是否有益
    "G_sync": [("r5_ctm", "r4_xattn")],  # CTM 的同步機制是否有益
}  # 結束門檻定義
# 只報告、永不作為門檻：每局的非線性編碼、訓練框架、再校準、最確定 tick 的挑法、
# 每位玩家 8 -> 32 題訓練題，以及 G_set / G_iter 的歸因對照：SAB 的局與局交互、
# 注意力池化 vs mean / max、反覆讀取（8 ticks vs 1）
REPORT_ONLY = [("r3_deepsets", "r2_pooled_delta"), ("r2_staged", "r2_pooled_delta"), ("r3_null", BASE_RUNG),  # 每局編碼、訓練框架、再校準
               ("r5_ctm_minmax@certain", "r5_ctm"), (f"{BASE_RUNG}@pp8", BASE_RUNG),  # 最確定的 tick、訓練題數
               ("r3_settf", "r3_pma"), ("r3_pma", "r3_deepsets"), ("r4_xattn", "r4_xattn_t1"), *FOLLOWUP_COMPARISONS]  # 歸因對照與追加項
N_BOOT = 5000  # 門檻判定所需的 bootstrap 重抽次數
NB_VARIANCE = 2.33  # Nadeau-Bengio 校正的變異數倍數
MIN_DIFF = 0.005  # 條件 (4) 的最小點差
N_SEEDS = 3  # 集合梯級所需的種子數
MIN_POSITIVE_FOLDS = 3  # 條件 (3)：4 個驗證折中至少幾折點差為正
CONDITIONS = ("ci_lo_gt_0", "every_seed_gt_0", "folds_gt_0_3of4", "diff_ge_0.005")  # 四個條件的名稱
# 負責訓練與評分梯級的模組，依此順序雜湊成 code_sha256；ladder.py、qeval.py 與
# ladder_fold0.py（摘要與評估程式）刻意不納入，這樣修改它們時，已完成的訓練
# 不會被標成由其他程式產生
CODE_FILES = ("gamevec.py", "ladder_data.py", "aggregators.py", "settrain.py", "rankmodel.py", "qfeatures.py", "metric.py")  # 納入 code_sha256 的模組
# 每次 CV 執行的 attrs 中的來源紀錄鍵；不適用時為 None（靜態梯級的 train_cfg 與 n_params），
# 在開始記錄之前寫出的檔案經 load_cv 讀入時也是 None
PROVENANCE = ("git_head", "git_dirty", "code_sha256", "code_files", "torch_version", "device_name", "train_cfg", "n_params",  # 來源紀錄的鍵
              "wall_seconds")  # 以及執行時間


# ---------------------------------------------------------------- 來源紀錄（provenance）

def _git(*args):  # 執行 git 並取得輸出
    """stdout of `git <args>` in the repository, None when git cannot tell (no git, no .git, an error)."""
    try:  # git 可能不存在
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30)  # 在儲存庫根目錄執行
    except (OSError, subprocess.SubprocessError):  # 沒有 git 或逾時
        return None  # 無法判斷
    return r.stdout if r.returncode == 0 else None  # 失敗時也視為無法判斷


def read_code_version():  # 讀取目前的程式版本
    """git_head (short sha of HEAD), git_dirty (tracked files differ from HEAD: `git status --porcelain
    --untracked-files=no` is not empty) -- both None when git cannot tell, e.g. a tree synced without
    .git -- code_sha256, the sha256 of the bytes of the CODE_FILES that exist, concatenated in order
    (`cat` them | sha256sum), and code_files, those files repo-relative."""
    head = _git("rev-parse", "--short", "HEAD")  # HEAD 的短 sha
    head = (head.strip() or None) if head is not None else None  # 空字串也視為無法判斷
    status = _git("status", "--porcelain", "--untracked-files=no") if head is not None else None  # 受版控檔案的變更清單
    h, files = hashlib.sha256(), []  # 程式雜湊與納入的檔案
    for name in CODE_FILES:  # 依固定順序
        path = Path(__file__).resolve().parent / name  # 與本模組同一個資料夾
        if path.is_file():  # 只納入存在的檔案
            h.update(path.read_bytes())  # 串接位元組
            files.append(path.relative_to(ROOT).as_posix())  # 記下相對路徑
    return {"git_head": head, "git_dirty": None if status is None else bool(status.strip()), "code_sha256": h.hexdigest(),  # 版本、是否有未提交的修改、程式雜湊
            "code_files": files}  # 納入雜湊的檔案


# 在本模組被匯入時只讀一次（訓練模組也同時被匯入），所以即使長時間工作執行中檔案被同步更新，
# 一次執行記錄的仍是這個行程實際執行的程式
_CODE_VERSION = read_code_version()  # 匯入時的程式版本


def code_version():  # 目前行程的程式版本
    """The provenance of the code this process runs (read_code_version at import), as a new dict."""
    return dict(_CODE_VERSION)  # 複製一份，避免呼叫端改到快取


def device_name(device):  # 裝置名稱
    """torch.cuda.get_device_name of a cuda device, else the device type ("cpu")."""
    d = torch.device(device)  # 解析裝置字串
    return torch.cuda.get_device_name(d) if d.type == "cuda" else d.type  # GPU 型號或裝置類型


def environment(device="cpu"):  # 每個結果都記錄的執行環境（最終流程與競賽一也使用）
    """code_version() with torch_version and device_name(device): what every result records."""
    return {**code_version(), "torch_version": torch.__version__, "device_name": device_name(device)}  # 程式版本、torch 版本與裝置


# ---------------------------------------------------------------- 輸入

def check_folds(folds):  # 驗證 CV 折的設定
    """Sorted tuple of distinct CV folds; fold 0 is refused."""
    folds = [int(f) for f in folds]  # 轉成整數
    if 0 in folds:  # 第 0 折不可進入 CV
        raise ValueError("fold 0 is evaluated once (experiments/ladder_eval_fold0.py) and never enters the CV")  # 拒絕
    if len(set(folds)) != len(folds) or len(folds) < 2:  # 有重複或少於 2 折
        raise ValueError(f"need at least 2 distinct CV folds, got {folds}")  # 拒絕
    return tuple(sorted(folds))  # 排序後的折


def load_players(folds, train_dir=TRAIN_DIR, folds_csv=FOLDS_CSV):  # 讀取各 CV 折的玩家
    """{fold: scoring.fold_players selection} of the CV folds (reads the training CSVs, ~30 s)."""
    folds = check_folds(folds)  # 先確認沒有第 0 折
    games, folds_df = player_games(train_dir), pd.read_csv(folds_csv)  # 訓練棋局清單與分折表
    return {f: fold_players(games, folds_df, f) for f in folds}  # 每折選出的玩家


def load_vectors(vectors_dir, folds):  # 讀取各折的每局向量
    """The gamevec vectors of the folds merged into one dict (each fold{F}.npz read once, by name)
    and the fold of every game."""
    parts = []  # 各折的向量
    for f in folds:  # 逐折
        path = Path(vectors_dir) / f"fold{f}.npz"  # 該折的向量檔
        if not path.exists():  # 還沒建立
            raise FileNotFoundError(f"{path} not built (experiments/ladder_build_vectors.py)")  # 提示建立向量的腳本
        with np.load(path) as z:  # 開啟 npz
            parts.append(dict(z))  # 讀入所有陣列
    names = sorted(parts[0])  # 第一折的鍵
    if any(sorted(p) != names for p in parts):  # 各折的鍵不一致
        raise ValueError(f"vector files of folds {list(folds)} have different keys")  # 拒絕
    vec = {k: np.concatenate([p[k] for p in parts]) for k in names}  # 依鍵串接各折
    fold_of = np.concatenate([np.full(len(p["game_key"]), f, dtype=np.int64) for f, p in zip(folds, parts)])  # 每局所屬的折
    if len(np.unique(vec["game_key"])) != len(fold_of):  # 同一局出現在多個折
        raise ValueError(f"game keys repeat across the vectors of folds {list(folds)}")  # 拒絕
    return vec, fold_of  # 合併的向量與折標記


def _check_players(players, vec, fold_of, folds):  # 核對向量與玩家選擇
    """The vectors of each fold must hold exactly the games its players contribute."""
    for f in folds:  # 逐折
        if f not in players:  # 缺少該折的玩家
            raise KeyError(f"no players for fold {f}")  # 拒絕
        want = {k for ks in players[f]["game_keys"] for k in ks}  # 該折玩家選出的棋局
        have = set(vec["game_key"][fold_of == f].tolist())  # 向量檔中該折的棋局
        if want != have:  # 兩者不一致
            raise ValueError(f"fold {f}: {len(have)} games in the vectors but {len(want)} selected "  # 回報兩邊的數量
                             f"(missing {len(want - have)}, extra {len(have - want)})")  # 缺少與多出的局數


def training_questions(players, v, folds, per_player):  # 切分 v 的訓練題目
    """Training questions of split v (fit_rank_model.train_set): one RandomState(42 + v), then
    sample_questions per training fold in ascending order; a `fold` column records the fold."""
    rng = np.random.RandomState(TRAIN_SEED + v)  # 整個切分共用一個亂數器
    return pd.concat([sample_questions(players[f], per_player, rng).assign(fold=f) for f in sorted(folds) if f != v],  # 依折號由小到大抽樣，略過驗證折
                     ignore_index=True)  # 重新編號


def validation_questions(players, v):  # 切分 v 的驗證題目
    return pilot_questions(players[v]).assign(fold=v)  # 驗證折玩家的巢狀題目


def _labels(qs):  # 題目的類別索引
    y = qs["rank"].map(RANK_TO_IDX)  # 等級轉索引
    if y.isna().any():  # 有不認得的等級
        raise ValueError(f"{int(y.isna().sum())} questions without a known rank")  # 拒絕
    return y.to_numpy(dtype=np.int64)  # 整數陣列


def _chunks(qs, vec, rep, counts):  # 分批打包題目
    for i in range(0, len(qs), CHUNK):  # 每批 CHUNK 題
        yield pack_questions(qs.iloc[i:i + CHUNK], vec, rep, counts)  # 打包這一批


def _features(qs, vec, kind, rep, counts, clip_k=None):  # 分批計算靜態特徵
    """static_features of the questions, packed CHUNK at a time to bound memory; every kind is a
    function of each question alone, so the rows equal those of one packing of all questions."""
    return np.vstack([static_features(p, vec, kind, counts, clip_k) for p in _chunks(qs, vec, rep, counts)])  # 逐批計算後堆疊


def _game_rows(qs, vec, rep, counts):  # fit_clip_k 需要的打包結果
    """{"idx": per question its vec rows}: the part of a packing that fit_clip_k reads."""
    return {"idx": [ix for p in _chunks(qs, vec, rep, counts) for ix in p["idx"]]}  # 只保留各批的 idx


def _fit_predict(make, X_tr, y_tr, X_val):  # 擬合靜態模型並預測
    return make().fit(X_tr, y_tr).predict_proba(X_val)  # 新建模型、擬合、預測驗證題


def _suffix(per_player, counts):  # 檔名後綴
    return ("" if per_player == DEFAULT_PER_PLAYER else f"_pp{per_player}") + ("" if counts else "_nocounts")  # 非預設題數加 _ppN，局長消融加 _nocounts


def cv_filename(rung, seed, per_player=DEFAULT_PER_PLAYER, counts=True):  # CV 結果的檔名
    """{rung}[_pp{N}][_nocounts]_s{seed}.npz"""
    return f"{rung}{_suffix(per_player, counts)}_s{seed}.npz"  # 組成檔名


# ---------------------------------------------------------------- 基準 logits

BASE_INFO = ("ce_train", "ce_val", "max_dev", "max_dev_unclipped", "n_clipped_rows", "n_train", "n_val")  # 基準快取記錄的診斷欄位


def _questions_sha(*frames):  # 題目內容的雜湊
    h = hashlib.sha256()  # 雜湊器
    for qs in frames:  # 逐個題目表
        for qid, keys in zip(qs["question_id"], qs["game_keys"]):  # 逐題
            h.update(f"{qid}:{';'.join(keys)}\n".encode())  # 題號與棋局鍵
        h.update(b"--\n")  # 題目表之間的分隔
    return h.hexdigest()  # 用來辨認快取是否對應同一批題目


def _weighted_ce(logits, y):  # 類別平衡交叉熵
    """Class-balanced CE of softmax(logits): sum w CE / sum w with rankmodel.class_weights."""
    z = logits - logits.max(axis=1, keepdims=True)  # 減去每列最大值，避免溢位
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))  # 取對數 softmax
    w = class_weights(y)  # 類別平衡權重
    return float((w * -logp[np.arange(len(y)), y]).sum() / w.sum())  # 加權平均的負對數機率


def split_base(v, vec, train_q, val_q, counts=True, per_player=DEFAULT_PER_PLAYER, cache=None):  # 切分 v 的基準 logits
    """Base logits of split v for the set rungs: (train [N_train, 10], val [N_val, 10], info).

    Training rows: cross-fitted (each training fold from r2_mlp24 fitted on the other training folds);
    validation rows: r2_mlp24 fitted on all training folds, stacked ascending. The validation base must
    equal the r2_mlp24 static rung of the split: max |softmax(base) - p_static| <= 1e-12 on the rows
    whose static probabilities are all >= 1e-9 (below that the log clip of the base moves them); the
    max deviation over every row is recorded. info: class-balanced CE of the base on the training
    (cross-fitted) and validation questions, the deviations and the counts. With `cache` (a directory)
    the result is stored as base_v{v}[_pp{N}][_nocounts].npz and reused; a file made for other
    questions is refused.
    """
    sha = _questions_sha(train_q, val_q)  # 這批題目的雜湊
    path = None if cache is None else Path(cache) / f"base_v{v}{_suffix(per_player, counts)}.npz"  # 快取檔路徑
    if path is not None and path.exists():  # 已有快取
        with np.load(path, allow_pickle=False) as z:  # 讀取快取
            d = dict(z)  # 所有陣列
        if str(d["questions_sha256"]) != sha:  # 快取對應的是別批題目
            raise ValueError(f"{path} is stale: it was made for other questions; delete it")  # 要求刪除過期快取
        return d["train"], d["val"], {k: d[k].item() for k in BASE_INFO}  # 直接沿用快取

    kind, rep, make = STATIC_RUNGS[BASE_RUNG]  # r2_mlp24 的特徵與模型
    X, X_val = _features(train_q, vec, kind, rep, counts), _features(val_q, vec, kind, rep, counts)  # 訓練與驗證的 q24 特徵
    y, y_val = _labels(train_q), _labels(val_q)  # 訓練與驗證的標籤
    fold = train_q["fold"].to_numpy()  # 每個訓練題所屬的折
    if np.any(np.diff(fold) < 0):  # 折沒有依序排列
        raise ValueError("training questions must be in ascending fold order")  # 交叉擬合需要依折排序
    tr_folds = sorted(set(fold.tolist()))  # 訓練折
    by_fold, base_val = crossfit_base_logits(make, {f: X[fold == f] for f in tr_folds}, {f: y[fold == f] for f in tr_folds},  # 每個訓練折的交叉擬合 logits
                                             X_val)  # 以及驗證題的 logits
    base = np.concatenate([by_fold[f] for f in tr_folds])  # 依折序串接訓練列
    p_static = _fit_predict(make, X, y, X_val)  # 這個切分的靜態梯級
    dev = np.abs(_softmax(base_val) - p_static)  # 驗證基準與靜態梯級機率的差
    unclipped = p_static.min(axis=1) >= BASE_CLIP  # 沒有受 log 截斷影響的列
    info = {"ce_train": _weighted_ce(base, y), "ce_val": _weighted_ce(base_val, y_val), "max_dev": float(dev.max()),  # 基準的 CE 與最大偏差
            "max_dev_unclipped": float(dev[unclipped].max()) if unclipped.any() else 0.0,  # 未截斷列的最大偏差
            "n_clipped_rows": int((~unclipped).sum()), "n_train": len(y), "n_val": len(y_val)}  # 截斷列數與題數
    if info["max_dev_unclipped"] > BASE_TOL:  # 驗證基準與靜態梯級不一致
        raise AssertionError(f"split {v}: validation base differs from the {BASE_RUNG} static proba by "  # 停止：基準必須就是 r2_mlp24
                             f"{info['max_dev_unclipped']:.3g} > {BASE_TOL}")  # 訊息續行
    if path is not None:  # 需要寫快取
        path.parent.mkdir(parents=True, exist_ok=True)  # 確保資料夾存在
        tmp = path.with_name(path.stem + ".tmp.npz")  # 先寫暫存檔
        np.savez_compressed(tmp, train=base, val=base_val, y_train=y, y_val=y_val, questions_sha256=np.asarray(sha),  # 基準、標籤與題目雜湊
                            **{k: np.asarray(val) for k, val in info.items()})  # 診斷資訊
        os.replace(tmp, path)  # 原子地換成正式檔名
    return base, base_val, info  # 訓練基準、驗證基準與診斷


# ---------------------------------------------------------------- CV（交叉驗證）

def _static_split(rung, vec, train_q, val_q, counts):  # 一個切分的靜態梯級
    kind, rep, make = STATIC_RUNGS[rung]  # 特徵種類、向量表示與模型工廠
    info, clip_k = {}, None  # 診斷資訊與截斷寬度
    if kind == "clip":  # 截斷寬度只由訓練題目決定
        clip_k = info["clip_k"] = fit_clip_k(_game_rows(train_q, vec, rep, counts), vec)  # 由訓練題求寬度並記錄
    X_tr, X_val = _features(train_q, vec, kind, rep, counts, clip_k), _features(val_q, vec, kind, rep, counts, clip_k)  # 訓練與驗證特徵
    return {"": _fit_predict(make, X_tr, _labels(train_q), X_val)}, info  # 主預測與診斷


def train_cfg(rung, seed, device):  # 集合梯級的訓練設定
    """The TrainCfg of a set rung: the defaults with its seed, device and SET_RUNGS overrides."""
    return replace(TrainCfg(seed=seed, device=device), **SET_RUNGS[rung][2])  # 套用該梯級的覆寫


def _set_split(rung, v, vec, train_q, val_q, counts, per_player, seed, device, cache, extras=True):  # 一個切分的集合梯級
    """({"": the probabilities at the rung's TrainCfg.pick tick, X: each EXTRAS[rung] prediction X unless
    extras=False}, info).

    A non-finite training loss or probability raises FloatingPointError naming rung, seed and split
    (evaluate would score an all-NaN row as a guess of D); nothing is returned, so nothing is written
    and a rerun does not skip the run."""
    rep, make, _ = SET_RUNGS[rung]  # 向量表示與模型工廠
    base, base_val, base_info = split_base(v, vec, train_q, val_q, counts, per_player, cache)  # 訓練列的交叉擬合基準、驗證基準與診斷
    data = pack_questions(train_q, vec, rep, counts)  # 打包訓練題目
    where = f"{rung} seed {seed} split {v}"  # 錯誤訊息用的位置描述
    try:  # 訓練集合模型
        fitted = fit_set_model(make, data, base, train_cfg(rung, seed, device))  # 學習對基準的修正量 Δ
    except FloatingPointError as e:  # 內層驗證 CE 不是有限值，與下面的檢查一樣註明位置
        raise FloatingPointError(f"{where}: {e}") from e  # 加上梯級、種子與切分後再引發
    del data  # 打包驗證題目之前先釋放訓練 token（數 GB）
    if not np.isfinite(fitted.losses).all():  # 訓練損失出現非有限值
        raise FloatingPointError(f"{where}: non-finite training loss (per-epoch means {fitted.losses})")  # 停止，不寫出結果
    val = pack_questions(val_q, vec, rep, counts)  # 打包驗證題目
    out = {}  # 各個預測的機率
    for name, kw in [("", {}), *(EXTRAS.get(rung, {}).items() if extras else [])]:  # 主預測與額外預測
        P = predict_set_model(fitted, val, base_val, **kw)  # 驗證題的機率
        bad = ~np.isfinite(P).all(axis=1)  # 含非有限值的列
        if bad.any():  # 有壞列
            raise FloatingPointError(f"{where}: non-finite probabilities for {int(bad.sum())} of {len(P)} validation "  # 停止，不寫出結果
                                     f"questions ({name or 'main'} prediction)")  # 註明是哪個預測
        out[name] = P  # 保存機率
    inner = fitted.inner_ce  # inner_frac = 0 時為 []：沒有選檢查點，內層欄位都是 None
    return out, {"base": base_info, "train_loss": fitted.losses, "n_params": n_params(fitted.model),  # 基準診斷、訓練損失與參數數量
                 "best_epoch": fitted.best_epoch, "inner_ce_init": inner[0] if inner else None,  # 保留權重的 epoch 與 Δ ≡ 0 的內層 CE
                 "inner_ce_best": inner[fitted.best_eval] if inner else None, "stopped_early": fitted.stopped_early,  # 最佳內層 CE 與是否提早停止
                 "n_inner_players": fitted.n_inner_players}  # 內層驗證玩家數


def _frame(val_q, v, probas):  # 整理驗證結果資料表
    """Validation frame: KEY_COLS, then p0..p9 and score per prediction ("" first, then each extra X as @X)."""
    y = _labels(val_q)  # 驗證題的標籤
    cols = {"question_id": val_q["question_id"].to_numpy(dtype=str), "fold": np.full(len(val_q), v, dtype=np.int64),  # 題號與折
            "player_id": val_q["player_id"].to_numpy(dtype=str), "y": y, "n_games": val_q["n_games"].to_numpy(dtype=np.int64)}  # 玩家、標籤與局數
    for name, P in probas.items():  # 逐個預測
        sfx = f"@{name}" if name else ""  # 額外預測的欄名後綴
        _, s = evaluate(P, y, val_q)  # 期望得分決策後的逐題官方分數
        cols.update({f"p{k}{sfx}": P[:, k] for k in range(N_CLASSES)})  # 10 個機率欄
        cols[f"score{sfx}"] = s  # 逐題分數欄
    return pd.DataFrame(cols)  # 驗證結果資料表


def run_rung_cv(rung, vectors_dir, folds=CV_FOLDS, seed=0, per_player=DEFAULT_PER_PLAYER, counts=True, device="cuda",  # 一個梯級的留一折 CV
                base_cache=None, players=None):  # 基準快取與玩家
    """Leave-one-fold-out CV of one rung (seed: set models' initialisation and shuffling; static rungs
    are deterministic). Returns the validation questions of every fold with KEY_COLS, p0..p9, score
    (and @X columns for EXTRAS). attrs: rung, kind, seed, per_player, counts, folds; the provenance
    of environment() -- git_head, git_dirty, code_sha256, code_files, torch_version, device_name
    ("cpu" for a static rung) -- train_cfg (the TrainCfg used) and n_params (None for a static rung);
    per split n_train and wall_seconds and, by rung, clip_k (r1_clip) or, for set rungs, base,
    train_loss (mean per epoch begun) and the inner early stopping: best_epoch (epochs completed at
    the kept weights, 0.0 for Δ ≡ 0), inner_ce_init (the inner CE of Δ ≡ 0), inner_ce_best,
    stopped_early and n_inner_players -- all None when the TrainCfg has inner_frac = 0.

    base_cache: directory for split_base's files (None: no caching). players: {fold: fold_players
    frame}, loaded with load_players when None.
    """
    folds = check_folds(folds)  # 確認折設定，拒絕第 0 折
    is_set = rung in SET_RUNGS  # 是否為集合梯級
    if not is_set and rung not in STATIC_RUNGS:  # 不認得的梯級
        raise KeyError(f"unknown rung {rung!r}; known: {list(STATIC_RUNGS) + list(SET_RUNGS)}")  # 拒絕
    env = environment(device if is_set else "cpu")  # 靜態梯級不論 device 為何都在 CPU 上擬合
    vec, fold_of = load_vectors(vectors_dir, folds)  # 讀取各折向量
    players = load_players(folds) if players is None else players  # 讀取各折玩家
    _check_players(players, vec, fold_of, folds)  # 核對向量與玩家
    frames, per_split = [], {}  # 各切分的結果與診斷
    for v in folds:  # 每個折輪流當驗證折
        t0 = time.perf_counter()  # 開始計時
        train_q, val_q = training_questions(players, v, folds, per_player), validation_questions(players, v)  # 該切分的訓練與驗證題目
        if is_set:  # 集合梯級
            probas, info = _set_split(rung, v, vec, train_q, val_q, counts, per_player, seed, device, base_cache)  # 訓練修正量並預測
        else:  # 靜態梯級
            probas, info = _static_split(rung, vec, train_q, val_q, counts)  # 擬合並預測
        frames.append(_frame(val_q, v, probas))  # 該驗證折的結果
        per_split[str(v)] = {**info, "n_train": len(train_q), "wall_seconds": time.perf_counter() - t0}  # 該切分的診斷與耗時
    df = pd.concat(frames, ignore_index=True)  # 合併各驗證折
    df.attrs = {"rung": rung, "kind": "set" if is_set else "static", "seed": int(seed), "per_player": int(per_player),  # 執行設定
                "counts": bool(counts), "folds": list(folds), **env}  # 以及程式版本與環境
    # 集合梯級：每個切分都相同；靜態梯級兩者都沒有
    df.attrs["train_cfg"] = asdict(train_cfg(rung, seed, device)) if is_set else None  # 所用的 TrainCfg
    df.attrs["n_params"] = per_split[str(folds[0])]["n_params"] if is_set else None  # 參數數量
    for key in ("n_train", "wall_seconds", "clip_k", "base", "train_loss", *INNER_KEYS):  # 逐切分的診斷欄位
        if key in per_split[str(folds[0])]:  # 這種梯級有這個欄位
            df.attrs[key] = {v: d[key] for v, d in per_split.items()}  # 以切分為鍵記錄
    return df  # 帶 attrs 的結果資料表


# ---------------------------------------------------------------- 檔案

def _plain(x):  # 轉成可寫入 JSON 的純 Python 物件
    """JSON-able copy: numpy scalars and arrays to Python, dict keys to str, non-finite floats to None."""
    if isinstance(x, dict):  # 字典
        return {str(k): _plain(v) for k, v in x.items()}  # 鍵轉字串，值遞迴處理
    if isinstance(x, (list, tuple)):  # 清單或 tuple
        return [_plain(v) for v in x]  # 逐項遞迴處理
    if isinstance(x, np.ndarray):  # numpy 陣列
        return _plain(x.tolist())  # 轉成清單再處理
    if isinstance(x, (bool, np.bool_)):  # 布林值（要在整數之前判斷）
        return bool(x)  # 轉成 Python bool
    if isinstance(x, (int, np.integer)):  # 整數
        return int(x)  # 轉成 Python int
    if isinstance(x, (float, np.floating)):  # 浮點數
        return float(x) if math.isfinite(x) else None  # NaN 與 inf 轉成 None
    return x  # 其他型別原樣保留


def to_json(obj, **kw):  # 嚴格 JSON 輸出（最終流程與競賽一的腳本也使用）
    """Strict JSON (allow_nan=False) of obj after _plain, so NaN / inf become null."""
    return json.dumps(_plain(obj), allow_nan=False, **kw)  # 不允許寫出 NaN，確保輸出是標準 JSON


def save_cv(df, path):  # 寫出一次 CV 的結果
    """Write a run_rung_cv frame and its attrs (as a JSON string) to an npz, atomically. String columns
    are fixed-width unicode, so np.load(path, allow_pickle=False) reads it back (see load_cv)."""
    arrays = {c: df[c].to_numpy() if pd.api.types.is_numeric_dtype(df[c]) else df[c].to_numpy(dtype=str)  # 數值欄原樣，其他欄轉成固定寬度字串
              for c in df.columns}  # 逐欄處理
    arrays["__meta__"] = np.asarray(to_json({**df.attrs, "columns": list(df.columns)}))  # attrs 與欄位順序存成 JSON 字串
    path = Path(path)  # 轉成 Path
    path.parent.mkdir(parents=True, exist_ok=True)  # 確保資料夾存在
    tmp = path.with_name(path.stem + ".tmp.npz")  # 先寫暫存檔
    np.savez_compressed(tmp, **arrays)  # 壓縮寫出
    os.replace(tmp, path)  # 原子地換成正式檔名


def load_cv(path):  # 讀回 save_cv 的檔案
    """(frame, meta) of a save_cv file; meta is also the frame's attrs. A file written before the
    provenance was recorded loads with every PROVENANCE field None."""
    with np.load(path, allow_pickle=False) as z:  # 不允許 pickle
        meta = json.loads(str(z["__meta__"]))  # 解析 attrs
        columns = meta.pop("columns")  # 欄位順序
        df = pd.DataFrame({c: z[c] for c in columns})  # 依原順序重建資料表
    for key in PROVENANCE:  # 舊檔可能缺少來源紀錄
        meta.setdefault(key, None)  # 缺的欄位補 None
    df.attrs = dict(meta)  # 放回 attrs
    return df, meta  # 資料表與中繼資料


# ---------------------------------------------------------------- 集成

def build_ensemble(cv_dir, members=ENSEMBLE_MEMBERS, rung=ENSEMBLE_RUNG, seeds=(0, 1, 2), per_player=DEFAULT_PER_PLAYER,  # 建立成員的機率平均集成
                   counts=True, overwrite=False):  # 計數設定與是否覆蓋
    """Write the probability-mean ensemble of the members' CV files: for each seed, the members' files
    cv_filename(member, seed, per_player, counts) in cv_dir (their main prediction; extras are ignored),
    checked for identical validation questions (_same_keys), their p0..p9 averaged and the score of every
    question recomputed from the mean probabilities (qeval.evaluate: the expected-score decision), saved
    with save_cv as cv_dir / cv_filename(rung, seed, ...). attrs: rung, kind "ensemble" (not "static", so
    check_comparison asks it for N_SEEDS seeds like a set rung; one seed per member seed), seed, per_player,
    counts, folds, members, member_code_sha256 and member_git_head (each member's
    provenance), the environment("cpu") of the code that built it (its code_sha256 is that of the training
    modules now, not the members'), train_cfg / n_params / wall_seconds None. Returns the paths written.

    `rung` must not be a trainable rung (RUNGS); a member file missing raises FileNotFoundError, an
    existing output FileExistsError unless overwrite."""
    cv_dir, members = Path(cv_dir), list(members)  # 轉成 Path 與清單
    if rung in RUNGS:  # 標籤與可訓練的梯級撞名
        raise ValueError(f"{rung!r} is a trainable rung; an ensemble needs its own label")  # 集成要用自己的標籤
    if len(set(members)) < 2 or len(set(members)) != len(members):  # 成員少於 2 個或有重複
        raise ValueError(f"an ensemble needs at least 2 distinct members, got {members}")  # 拒絕
    pcols = [f"p{k}" for k in range(N_CLASSES)]  # 機率欄名
    written = []  # 寫出的檔案
    for seed in seeds:  # 每個種子一個集成檔
        out = cv_dir / cv_filename(rung, seed, per_player, counts)  # 輸出路徑
        if out.exists() and not overwrite:  # 已存在且未允許覆蓋
            raise FileExistsError(f"{out} exists (pass overwrite=True / --overwrite to rebuild it)")  # 拒絕
        frames, metas = [], []  # 各成員的資料表與中繼資料
        for m in members:  # 逐個成員
            path = cv_dir / cv_filename(m, seed, per_player, counts)  # 成員在該種子的 CV 檔
            if not path.exists():  # 還沒跑
                raise FileNotFoundError(f"{path} is missing: member {m} seed {seed} of {rung} has no CV results yet")  # 拒絕
            df, meta = load_cv(path)  # 讀取成員結果
            if meta["rung"] != m or int(meta["seed"]) != seed:  # 檔案內容與檔名不符
                raise ValueError(f"{path}: holds {meta['rung']} seed {meta['seed']}, not {m} seed {seed}")  # 拒絕
            frames.append(df)  # 保存資料表
            metas.append(meta)  # 保存中繼資料
        for m, df in zip(members[1:], frames[1:]):  # 其他成員逐一比對
            _same_keys(frames[0], df, f"{rung} seed {seed}: {members[0]} vs {m}")  # 驗證題目必須完全相同
        P = np.mean([df[pcols].to_numpy(dtype=float) for df in frames], axis=0)  # 成員機率平均
        ens = frames[0][KEY_COLS].reset_index(drop=True).copy()  # 以第一個成員的鍵欄為底
        _, s = evaluate(P, ens["y"].to_numpy(dtype=np.int64), ens)  # 以平均機率重新決策並評分
        for k in range(N_CLASSES):  # 逐個類別
            ens[f"p{k}"] = P[:, k]  # 寫入平均機率
        ens["score"] = s  # 寫入逐題分數
        ens.attrs = {"rung": rung, "kind": "ensemble", "seed": int(seed), "per_player": int(per_player), "counts": bool(counts),  # 集成的設定
                     "folds": sorted(int(f) for f in ens["fold"].unique()), "members": members,  # 折與成員
                     "member_code_sha256": {m: meta["code_sha256"] for m, meta in zip(members, metas)},  # 各成員的程式雜湊
                     "member_git_head": {m: meta["git_head"] for m, meta in zip(members, metas)},  # 各成員的 git 版本
                     **environment("cpu"), "train_cfg": None, "n_params": None, "wall_seconds": None}  # 建立集成的環境，不適用的欄位為 None
        save_cv(ens, out)  # 寫出集成檔
        written.append(out)  # 記下路徑
    return written  # 寫出的路徑清單


def across_seeds(cv_dir, rung, per_player=DEFAULT_PER_PLAYER, counts=True, n_boot=N_BOOT):  # 比較提交規則與 CV 統計量
    """Report only (item 2): how the submission's decision rule -- one decision on the probabilities averaged
    over a rung's seeds -- scores against the CV statistic of the same files, which decides per seed and
    averages the scores. {"label", "seeds", "score_seedmean_proba", "score_cv", "diff", "ci95", "by_n",
    "n_boot"}: diff = seedmean_proba − cv with its player-bootstrap CI (qeval.paired_diff)."""
    label = _label(rung, per_player, counts)  # 該設定的標籤
    runs = _read_runs(cv_dir)  # 讀取所有 CV 檔
    if label not in runs:  # 沒有這個標籤的結果
        raise KeyError(f"no CV files of {label} in {cv_dir}")  # 拒絕
    run = runs[label]  # 該標籤的各種子結果
    if len(run["seeds"]) < 2:  # 種子太少
        raise ValueError(f"{label} has {len(run['seeds'])} seed(s); averaging over seeds needs at least 2")  # 至少要 2 個種子
    y, n, _ = _arrays(run["keys"])  # 標籤與局數
    P = np.mean([run["proba"][s] for s in run["seeds"]], axis=0)  # 種子間平均機率
    _, s_mean = evaluate(P, y, run["keys"])  # 平均機率一次決策後的逐題分數
    d = paired_diff(s_mean, run["mean"], y, run["keys"]["player_id"].to_numpy(dtype=str), n, n=n_boot)  # 對 CV 統計量的成對差
    return {"label": label, "seeds": run["seeds"], "score_seedmean_proba": balanced_mean_over_n(s_mean, y, n),  # 提交規則的分數
            "score_cv": balanced_mean_over_n(run["mean"], y, n), **d, "n_boot": n_boot}  # CV 統計量與差值


# ---------------------------------------------------------------- 摘要

_RUN_FILE = re.compile(r".+_s\d+\.npz")  # CV 結果檔名的樣式


def _label(rung, per_player, counts, extra=None):  # 摘要中的標籤
    return (rung + ("" if per_player == DEFAULT_PER_PLAYER else f"@pp{per_player}") + ("" if counts else "@nocounts")  # 梯級加上非預設題數與局長消融的後綴
            + (f"@{extra}" if extra else ""))  # 額外預測的後綴


def _same_keys(a, b, what):  # 確認兩個結果的題目相同
    for c in KEY_COLS:  # 逐個鍵欄
        if not np.array_equal(np.asarray(a[c]).astype(str), np.asarray(b[c]).astype(str)):  # 內容或順序不同
            raise ValueError(f"{what}: the validation questions differ (column {c}); paired results need identical questions")  # 成對比較需要相同的題目


def _read_runs(cv_dir):  # 讀取資料夾中所有 CV 結果
    """{label: {"rung", "kind", "per_player", "counts", "extra", "keys", "scores": {seed: [Q]}, "proba": {seed: [Q, 10]},
    "meta": {seed: meta}}} from the save_cv files of cv_dir; each extra column set is its own label."""
    runs = {}  # 標籤到結果
    for path in sorted(Path(cv_dir).glob("*.npz")):  # 逐個 npz
        if not _RUN_FILE.fullmatch(path.name) or path.name.startswith("base_"):  # 不是 CV 結果檔或是基準快取
            continue  # 略過
        df, meta = load_cv(path)  # 讀取結果
        rung, seed, pp, counts = meta["rung"], int(meta["seed"]), int(meta["per_player"]), bool(meta["counts"])  # 檔案記錄的設定
        if path.name != cv_filename(rung, seed, pp, counts):  # 檔名與內容不符
            raise ValueError(f"{path.name}: the name does not match its contents ({cv_filename(rung, seed, pp, counts)})")  # 拒絕
        extras = [None] + [c[len("score@"):] for c in df.columns if c.startswith("score@")]  # 主預測與各個額外預測
        for extra in extras:  # 每個預測各成一個標籤
            label, sfx = _label(rung, pp, counts, extra), f"@{extra}" if extra else ""  # 標籤與欄名後綴
            run = runs.setdefault(label, {"rung": rung, "kind": meta["kind"], "per_player": pp, "counts": counts, "extra": extra,  # 第一次遇到時建立
                                          "keys": df[KEY_COLS], "scores": {}, "proba": {}, "meta": {}})  # 鍵欄與各種子的容器
            _same_keys(run["keys"], df, f"{label} seed {seed}")  # 各種子的題目必須相同
            run["scores"][seed] = df[f"score{sfx}"].to_numpy(dtype=float)  # 該種子的逐題分數
            run["proba"][seed] = df[[f"p{k}{sfx}" for k in range(N_CLASSES)]].to_numpy(dtype=float)  # 該種子的機率
            run["meta"][seed] = meta  # 該種子的中繼資料
    for run in runs.values():  # 每個標籤
        run["seeds"] = sorted(run["scores"])  # 有結果的種子
        run["mean"] = np.mean([run["scores"][s] for s in run["seeds"]], axis=0)  # 每題分數的種子平均
    return runs  # 所有標籤的結果


def _arrays(keys):  # 取出標籤、局數與折
    return keys["y"].to_numpy(dtype=np.int64), keys["n_games"].to_numpy(dtype=np.int64), keys["fold"].to_numpy(dtype=np.int64)  # 三個整數陣列


def _per_fold(s, y, n, fold):  # 各驗證折的標題分數
    return {int(f): balanced_mean_over_n(s[fold == f], y[fold == f], n[fold == f]) for f in np.unique(fold)}  # 每折分別計算


def _rung_summary(run):  # 一個標籤的摘要
    y, n, fold = _arrays(run["keys"])  # 標籤、局數與折
    s = run["mean"]  # 逐題分數的種子平均
    per_seed = {seed: balanced_mean_over_n(run["scores"][seed], y, n) for seed in run["seeds"]}  # 各種子的標題分數
    # 每題先對種子平均 qeval.nll_brier（與 evaluate 的定義相同），再做類別平衡
    nll, brier = (np.mean(v, axis=0) for v in zip(*(nll_brier(run["proba"][seed], y) for seed in run["seeds"])))  # 逐題 NLL 與 Brier 的種子平均
    out = {"rung": run["rung"], "kind": run["kind"], "per_player": run["per_player"], "counts": run["counts"],  # 設定
           "extra": run["extra"], "report_only": run["extra"] is not None, "seeds": run["seeds"], "n_questions": len(y),  # 額外預測只報告
           "balanced_mean_over_n": balanced_mean_over_n(s, y, n),  # 合併的標題分數
           "balanced_by_n": {m: balanced_mean_over_n(s, y, n, sizes=(m,)) if (n == m).any() else None for m in SIZES},  # 各題目大小的分數
           "per_fold": _per_fold(s, y, n, fold),  # 各驗證折的分數
           "per_class": {LEVELS[k]: float(s[y == k].mean()) if (y == k).any() else None for k in range(N_CLASSES)},  # 各等級的平均分數
           "nll_balanced": balanced(nll, y), "brier": balanced(brier, y),  # 類別平衡 NLL 與 Brier
           "per_seed": per_seed, "seed_sd": float(np.std(list(per_seed.values()), ddof=1)) if len(per_seed) > 1 else None}  # 各種子分數與標準差
    # 各種子訓練時所用的程式版本（None：開始記錄來源紀錄之前的檔案）；
    # 超過一個時會醒目地報告（summarize、experiments/ladder_cv.py），但不會讓任何門檻失敗
    versions = {run["meta"][seed]["code_sha256"] for seed in run["seeds"]}  # 各種子的程式雜湊
    out["code_sha256"] = sorted(versions, key=lambda x: (x is None, x or ""))  # 排序，None 放最後
    out["mixed_versions"] = len(versions) > 1  # 是否混用不同程式
    first = run["meta"][run["seeds"][0]]  # 第一個種子的中繼資料
    if "clip_k" in first:  # r1_clip 的截斷寬度
        out["clip_k"] = first["clip_k"]  # 照抄到摘要
    if "train_loss" in first:  # 每個種子與切分最後一個 epoch 的訓練損失
        out["train_loss_last"] = {seed: {v: losses[-1] if losses else None  # 取最後一個 epoch
                                         for v, losses in run["meta"][seed]["train_loss"].items()}  # 逐切分
                                  for seed in run["seeds"]}  # 逐種子
    return out  # 該標籤的摘要


def _seed_pairs(A, B):  # 比較雙方的種子組合
    """Per-seed pairs (key, seed of a, seed of b). Only a static side (deterministic, seed 0) is reused
    against every seed of the other side; two set rungs pair the seeds both have, so a set rung that
    has run fewer seeds yields fewer pairs (and check_comparison counts each side's seeds)."""
    sa, sb = A["seeds"], B["seeds"]  # 雙方的種子
    if A["kind"] == "static" and B["kind"] == "static":  # 兩個都是靜態梯級
        return [(sa[0], sa[0], sb[0])]  # 只有一個差值
    if B["kind"] == "static":  # 只有 b 是靜態
        return [(s, s, sb[0]) for s in sa]  # a 的每個種子對 b 的唯一結果
    if A["kind"] == "static":  # 只有 a 是靜態
        return [(s, sa[0], s) for s in sb]  # b 的每個種子對 a 的唯一結果
    return [(s, s, s) for s in sorted(set(sa) & set(sb))]  # 兩個集合梯級：雙方共有的種子


def _compare(a, b, A, B, n_boot):  # 一組成對比較
    """Paired comparison a - b on the pooled questions: the seed-mean point difference with its player
    bootstrap CI (qeval.paired_diff), the Nadeau-Bengio interval (the CI's distances from the point
    difference scaled by sqrt(2.33), i.e. variance x 2.33), per-fold and per-seed point differences."""
    _same_keys(A["keys"], B["keys"], f"{a} vs {b}")  # 兩者必須是同一批題目
    y, n, fold = _arrays(A["keys"])  # 標籤、局數與折
    pid = A["keys"]["player_id"].to_numpy(dtype=str)  # 玩家 id（bootstrap 的單位）
    d = paired_diff(A["mean"], B["mean"], y, pid, n, n=n_boot)  # 種子平均分數的成對差與信賴區間
    diff, (lo, hi) = d["diff"], d["ci95"]  # 點差與區間
    k = math.sqrt(NB_VARIANCE)  # 區間寬度的放大倍數
    fa, fb = _per_fold(A["mean"], y, n, fold), _per_fold(B["mean"], y, n, fold)  # 雙方各折的分數
    per_seed = {key: balanced_mean_over_n(A["scores"][s1], y, n) - balanced_mean_over_n(B["scores"][s2], y, n)  # 各種子組合的點差
                for key, s1, s2 in _seed_pairs(A, B)}  # 依 _seed_pairs 的組合
    return {"a": a, "b": b, "diff": diff, "ci95": [lo, hi], "by_n": d["by_n"],  # 點差、信賴區間與各題目大小的差
            "nb_ci95": [diff - k * (diff - lo), diff + k * (hi - diff)],  # Nadeau-Bengio 保守區間
            "per_fold": {f: fa[f] - fb[f] for f in fa}, "per_seed": per_seed,  # 各折與各種子的點差
            "kind_a": A["kind"], "kind_b": B["kind"], "seeds_a": A["seeds"], "seeds_b": B["seeds"], "n_boot": n_boot}  # 判定條件需要的資訊


def comparison_pairs(labels):  # 列出要做的比較
    """Pre-specified comparisons among the labels present: every label vs r2_mlp24, the gate pairs, the
    report-only pairs, and each @nocounts label vs its counts form (the game-length ablation)."""
    labels = list(labels)  # 轉成清單
    pairs = [(x, BASE_RUNG) for x in labels if x != BASE_RUNG]  # 每個標籤對基準
    pairs += [p for gate in GATES.values() for p in gate] + REPORT_ONLY  # 門檻與只報告的比較
    pairs += [(x, x.replace("@nocounts", "")) for x in labels if "@nocounts" in x]  # 局長消融對原版
    present = set(labels)  # 有結果的標籤
    return list(dict.fromkeys((a, b) for a, b in pairs if a in present and b in present))  # 只留雙方都有結果的，去重並保持順序


def summarize(cv_dir, n_boot=N_BOOT):  # 產生 CV 摘要
    """Pooled CV summary of every save_cv file in cv_dir (see the module docstring), with the
    split_base diagnostics of its base files. Returned in its JSON form (string keys, non-finite
    values as None), so gates() reads it and cv_summary.json alike. mixed_versions lists the labels
    whose seeds carry different code_sha256 (each label's rung entry has code_sha256 and
    mixed_versions); it is a warning, and the gates do not read it."""
    cv_dir = Path(cv_dir)  # 轉成 Path
    runs = _read_runs(cv_dir)  # 讀取所有 CV 結果
    rungs = {label: _rung_summary(run) for label, run in runs.items()}  # 每個標籤的摘要
    comparisons = {f"{a} vs {b}": _compare(a, b, runs[a], runs[b], n_boot) for a, b in comparison_pairs(runs)}  # 所有事先指定的比較
    base = {}  # 基準快取的診斷
    for path in sorted(cv_dir.glob("base_v*.npz")):  # 逐個基準快取檔
        if not path.name.endswith(".tmp.npz"):  # 略過未完成的暫存檔
            with np.load(path, allow_pickle=False) as z:  # 讀取快取
                base[path.stem] = {k: z[k].item() for k in BASE_INFO}  # 取出診斷欄位
    folds = sorted({int(f) for run in runs.values() for f in run["keys"]["fold"].unique()})  # 出現過的驗證折
    return _plain({"created": datetime.now().isoformat(timespec="seconds"), "cv_dir": str(cv_dir),  # 建立時間與資料夾
            "protocol": {"folds": folds, "sizes": list(SIZES), "train_questions": f"RandomState({TRAIN_SEED} + v)",  # 協定：折、題目大小與訓練題抽樣
                         "n_boot": n_boot, "nb_variance_factor": NB_VARIANCE, "min_diff": MIN_DIFF, "n_seeds": N_SEEDS,  # bootstrap 與門檻參數
                         "min_positive_folds": f"{MIN_POSITIVE_FOLDS} of {len(CV_FOLDS)}"},  # 至少幾折為正
            "mixed_versions": [label for label, r in rungs.items() if r["mixed_versions"]],  # 混用程式版本的標籤
            "rungs": rungs, "comparisons": comparisons, "diagnostics": {"base": base}})  # 各標籤摘要、比較與診斷


# ---------------------------------------------------------------- 門檻

def _gt(x, t):  # 缺值視為不通過的大於比較
    return x is not None and x > t  # None 時為 False


def check_comparison(c):  # 判定一組比較的四個條件
    """The four conditions of one comparison: (1) the seed-mean paired CI lower bound > 0, (2) every
    per-seed point difference > 0, where each set-rung side must have run >= 3 seeds (a static side is
    deterministic, seed 0, and is compared with each seed of the other side; two static rungs give
    one difference) and two set rungs need >= 3 seeds in common, (3) >= 3 of the 4 validation folds'
    point differences > 0, (4) the point difference >= 0.005. The comparison passes only with all
    four and a CI from >= N_BOOT bootstrap replicates (a summary made with fewer cannot decide)."""
    per_seed, per_fold = list(c["per_seed"].values()), list(c["per_fold"].values())  # 各種子與各折的點差
    sides = [(side, c[f"kind_{side}"], len(c[f"seeds_{side}"])) for side in ("a", "b")]  # 雙方的種類與種子數
    short = [f"{side} is a set rung with {k} seed{'s' if k != 1 else ''}" for side, kind, k in sides  # 種子不足的集合梯級
             if kind != "static" and k < N_SEEDS]  # 靜態梯級不受種子數限制
    need = N_SEEDS if any(kind != "static" for _, kind, _ in sides) else 1  # 需要的種子差值數
    cond = {"ci_lo_gt_0": _gt(c["ci95"][0], 0.0),  # (1) 信賴區間下界 > 0
            "every_seed_gt_0": not short and len(per_seed) >= need and all(_gt(d, 0.0) for d in per_seed),  # (2) 每個種子的點差 > 0
            "folds_gt_0_3of4": len(per_fold) == len(CV_FOLDS) and sum(_gt(d, 0.0) for d in per_fold) >= MIN_POSITIVE_FOLDS,  # (3) 至少 3/4 折為正
            "diff_ge_0.005": c["diff"] is not None and c["diff"] >= MIN_DIFF}  # (4) 點差 >= 0.005
    boot_ok = (c.get("n_boot") or 0) >= N_BOOT  # bootstrap 次數足夠
    reasons = []  # 沒過的原因
    if not cond["ci_lo_gt_0"]:  # 條件 (1) 不成立
        reasons.append(f"CI lower bound {c['ci95'][0]} <= 0")  # 記錄原因
    if not cond["every_seed_gt_0"]:  # 條件 (2) 不成立
        reasons.append(f"per-seed differences {per_seed} (need {need} seeds per set rung, all > 0"  # 記錄各種子點差
                       + (f"; {', '.join(short)}" if short else "") + ")")  # 以及種子不足的一方
    if not cond["folds_gt_0_3of4"]:  # 條件 (3) 不成立
        reasons.append(f"per-fold differences {per_fold} (need {MIN_POSITIVE_FOLDS} of {len(CV_FOLDS)} > 0)")  # 記錄各折點差
    if not cond["diff_ge_0.005"]:  # 條件 (4) 不成立
        reasons.append(f"difference {c['diff']} < {MIN_DIFF}")  # 記錄點差
    if not boot_ok:  # 重抽次數不足
        reasons.append(f"CI from n_boot {c.get('n_boot')} < {N_BOOT} replicates: this summary cannot decide gates")  # 這份摘要不能判定門檻
    return {"pass": all(cond.values()) and boot_ok, "missing": False, "conditions": cond, "n_boot_ok": boot_ok,  # 判定結果與各條件
            "reasons": reasons, "diff": c["diff"], "ci95": c["ci95"]}  # 原因、點差與區間


def gates(summary):  # 判定所有門檻
    """{gate: {"pass", "comparisons": {"a vs b": check_comparison result}}} for GATES; a comparison
    absent from the summary fails as missing."""
    out = {}  # 各門檻的結果
    for name, pairs in GATES.items():  # 逐個門檻
        comps = {}  # 該門檻各比較的結果
        for a, b in pairs:  # 逐組比較
            key = f"{a} vs {b}"  # 摘要中的鍵
            c = summary.get("comparisons", {}).get(key)  # 該比較的摘要
            comps[key] = check_comparison(c) if c is not None else {  # 有結果時判定四個條件
                "pass": False, "missing": True, "conditions": {k: False for k in CONDITIONS}, "n_boot_ok": False,  # 沒有結果時視為不通過
                "reasons": [f"no CV results for {key}"]}  # 原因：缺少 CV 結果
        out[name] = {"pass": all(c["pass"] for c in comps.values()), "comparisons": comps}  # 每組比較都通過才算門檻成立
    return out  # 各門檻的判定


def g_set_ready(summary):  # 是否可以執行 r4／r5
    """(ok, reason): the iterative rungs (4)(5) run only when the r3 rungs and their controls are in
    the summary and G_set holds."""
    need = list(dict.fromkeys(x for pair in GATES["G_set"] for x in pair))  # G_set 需要的標籤（去重並保持順序）
    missing = [x for x in need if x not in summary.get("rungs", {})]  # 摘要中缺少的標籤
    if missing:  # 還不能檢查 G_set
        return False, (f"G_set cannot be checked: no CV results for {', '.join(missing)}; "  # 說明缺少哪些結果
                       f"run them first, or pass --force to run r4/r5 anyway")  # 提示先跑或用 --force
    g = gates(summary)["G_set"]  # 判定 G_set
    if not g["pass"]:  # G_set 不成立
        why = "; ".join(f"{k}: {', '.join(c['reasons'])}" for k, c in g["comparisons"].items() if not c["pass"])  # 彙整沒過的原因
        return False, f"G_set does not hold ({why}); pass --force to run r4/r5 anyway"  # 依協定不跑 r4／r5，除非 --force
    return True, "G_set holds"  # 可以執行 (4)(5)

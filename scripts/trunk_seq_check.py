#!/usr/bin/env python
"""Does an in-game temporal layer add to trunk vectors under a whole-question loss? (architecture v2 section 4A)

Three arms on one trainer, same questions, same seeds:
  R0  the 64 R features (fused_opp_diff_hd on the _hd scores) only;
  C   R + per-ply residual blocks (kernel 1) over trunk vectors and per-move log p, pooled per game, then per question;
  T   the same with five dilated kernel-3 blocks (63 plies).
Everything from the loss back to the blocks trains together; the trunk vectors are fixed (scripts/trunk_extract.py).
C's hidden size is chosen so its blocks hold about as many parameters as T's. With --inner-frac the stopping point is chosen on
held-out training players only and the validation fold is scored once, at that point. Fold 0 is refused.

  python -I scripts/trunk_seq_check.py build --out results/trunk_seq
  python -I scripts/trunk_seq_check.py train --arm T --val-fold 1 --seed 0 --out results/trunk_seq
  python -I scripts/trunk_seq_check.py rfeat --val-fold 2
  python -I scripts/trunk_seq_check.py compare --val-folds 2 3 4 --out results/trunk_seq/es
  --loss kernel | es-ft  (section 4F) cross-entropy against the official score's kernel (tag -k), or the expected-score
                         fine-tune after the cross-entropy stage, stopped on the inner players (tag -es)
  python -I scripts/trunk_seq_check.py train --arm C --val-fold all --seed 0 --inner-frac 0.1   # the submission's models (2026-11-04):
      every fold's players train, the 10% inner hold-out picks the stopping point, the weights and both standardisations go to
      results/models/ct/all_<tag>_s<seed>.pt (+ .json sidecar); --save-dir also keeps a validation-fold model's weights.
      scripts/trunk_predict.py reads them (load_checkpoint).
  Section 4M (2026-11-04) data-axis arms, same trainer: --train-folds 2 (tag -f2: two of the other folds train, fold (v mod 4) + 1
      dropped), --train-games 10 --train-sizes 5 10 (tag -g10s5-10: each training player keeps 10 scored games, questions of 5-10
      games; -s5-10 alone is its control), --vec-dir results/trunk_vec_ft (tag -ft: another trunk's vectors; `build --vec-dir` first
      writes their score arrays scores_ft_fold<f>.npy). compare --sizes 5 10 evaluates only those question sizes.
  Section 4M M4: --pooling pma (tag -pma) replaces both mean/std poolings by pooling-by-multihead-attention (Lee et al. 2019): four
      learned seeds attend over a game's plies, two over a question's games; the linear layers after them keep their shapes.
  Section 4M M14: --opp-hsl (tag -oh) fills the opponent plies' Human SL token block (scores columns 11-29, zeros before) with the
      centred log p of the OPPONENT's moves under the same 18 profiles (results/scores_opp_hd, _x40; score_folds.py --opponent);
      `build --opp-hsl` writes the scores_oh_* arrays. --opp-hsl-r (tag -ohr) also appends the 39 opponent-HSL question features
      (aicup.opphsl) to R (64 -> 103); the R cache of -oh is the plain one.
  Section 4M M8: --extra-games (tag -x40) adds every training player's games 21-40 (scored into the *_x40 directories, EXTRA) to the
      games its questions draw from; the validation questions stay the 20-game sample's. --with-fold0 (tag -f0, --val-fold all only)
      adds fold 0's players to the submission's training set (the user's decision). `build --build-folds 0` and
      `build --vec-dir results/trunk_vec_x40 --extra-games` write the matching score arrays.
  Section 4M M30 (2026-11-06): --token-enc dual (tag -dual) replaces the first Linear(162, d) by a board tower (trunk g, l: 128 ->
      48) and a score tower (L1b + Human SL: 30 -> 24), each Linear-ReLU-Linear-LayerNorm, then Linear(48 + 24 + 4 flags, d);
      --token-enc clip (tag -clip) adds CLIP-style alignment: both towers' outputs are projected to 32 dims and L2-normalised, and a
      symmetric InfoNCE over up to 2,048 real plies of the batch (the same ply's two views are the positive pair, learnable temperature)
      joins the loss with weight --clip-weight (0.1). The heads only shape training; evaluation never computes the InfoNCE.
  Section 4M M33 (2026-11-06): --player-contrast W (tag -pc, -pc<W> when W != 0.1) builds every batch from pairs of training
      questions of the same player (each player's questions paired at random each epoch) and adds W x a symmetric InfoNCE between the
      two questions' embeddings (the question layer's output before dropout, through a training-only Linear(32, 32), L2-normalised,
      temperature 0.1): other players in the batch are the negatives. The checkpoint holds the model only (the head is not saved).
      Diagnostics: --pair-batches (tag -pb) builds the same paired batches without the InfoNCE; --tag-extra X appends -X to the arm
      tag (a same-recipe re-run beside the original files).
  Section 4M M39 (2026-11-07): --pc-temp T (tag suffix t<T> when not 0.1) and --pc-views K (K same-player questions a group, tag
      suffix v<K>; K > 2 uses the supervised-contrastive loss, every other question of the player a positive); K = 2 is M33's loss.
  Section 4M M40: --r-extra-npz PATTERN (tag -rh) appends the per-question columns of PATTERN.format(v=val_fold) (X_tr / X_va, in the
      R cache's question order; experiments/nano4/opphist_r_extra.py) to R.
  Section 4M M36 (2026-11-06): --r-extra DIR (tag -ra) appends to R the question mean and std of the cross-fitted spatial-attention
      heads' per-game log-softmax (aicup.attnfeat; DIR = experiments/nano4/e2e_pilot.py score parts, --r-extra-games the games CSV): 64 -> 84.
"""
import argparse  # 解析命令列參數
import json  # 讀寫快取的題目表、附屬檔與結果紀錄
import math  # 對數與餘弦排程
import os  # 環境變數、行程編號與原子替換檔案
import sys  # 匯入路徑與結束碼
import time  # 計時
from pathlib import Path  # 路徑操作

import numpy as np  # 陣列運算
import pandas as pd  # 棋局表與題目表
import torch  # 網路與 GPU 張量

ROOT = Path(__file__).resolve().parents[1]  # 儲存庫根目錄
for d in (ROOT / "src", ROOT / "scripts"):  # aicup 套件與同目錄的腳本
    if str(d) not in sys.path:  # 尚未在匯入路徑中
        sys.path.insert(0, str(d))  # 加到最前面，下面的匯入才找得到
from aicup.handicap import read_hd_table  # noqa: E402  讀取讓子（HD）表
from aicup.ladder import FOLDS_CSV  # noqa: E402  玩家分折表的路徑
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402  官方分數評估與成對比較
from aicup.qfeatures import sample_questions  # noqa: E402  從玩家的棋局抽訓練題
from aicup.metric_loss import expected_score_loss_torch, kernel_ce_torch, kernel_targets, score_matrix  # noqa: E402  section 4F 的核目標與期望分損失
from aicup.rankmodel import class_weights  # noqa: E402  類別平衡權重
from aicup.opphsl import opp_hsl_features  # noqa: E402  M14 的 39 個對手 HSL 題目特徵
from aicup.scoring import _part_moves, completed_parts, part_paths, read_scored_games  # noqa: E402  讀取評分輸出的各分段檔
import hd_check  # noqa: E402  64 個 R 特徵的計算
import l1b_cv  # noqa: E402  讀各折資料、玩家表與驗證題

FOLDS = (1, 2, 3, 4)  # 交叉驗證的四個折；第 0 折不當驗證折，只在 --with-fold0 時加入訓練
VEC, HSL = ROOT / "results/trunk_vec", ROOT / "results/scores_hd"  # 預設的主幹向量與 Human SL 評分目錄
L1B, OPP = {1: "scores3_B_hd", 2: "scores3_B_hd", 3: "scores3_hd", 4: "scores3_hd"}, {1: "scores3_B_opp_hd", 2: "scores3_B_opp_hd", 3: "scores3_opp_hd", 4: "scores3_opp_hd"}  # 第 3 組各折的 L1b 評分目錄（本人側、對手側）：第 1、2 折用 heads3_B 的分數，第 3、4 折用 heads3_A 的
L1B[0], OPP[0] = "scores3_hd", "scores3_opp_hd"  # 第 0 折：兩個 L1b 模型都是折外；它的 token 與 R 用 A 的分數（section 4M M8）
L1B_SETS = {  # section 4M M9：哪一組 L1b 等級頭的逐手 log p 供應 token 與 R；第 4 組 = heads4_A（以第 1、3 折訓練）/ heads4_B（以第 2、4 折訓練）
    3: {"l1b": L1B, "opp": OPP, "heads": ["heads3_A", "heads3_B"]},  # 第 3 組：正式流程的 heads3_A / heads3_B
    4: {"l1b": {0: "scores4_hd", 2: "scores4_hd", 4: "scores4_hd", 1: "scores4_B_hd", 3: "scores4_B_hd"},  # 第 4 組本人側：第 0、2、4 折用 heads4_A 的分數，第 1、3 折用 heads4_B 的
        "opp": {0: "scores4_opp_hd", 2: "scores4_opp_hd", 4: "scores4_opp_hd", 1: "scores4_B_opp_hd", 3: "scores4_B_opp_hd"}, "heads": ["heads4_A", "heads4_B"]}}  # 第 4 組對手側目錄與等級頭名稱
EXTRA_MAX_GAMES = 40  # section 4M M8：fold_players 排列中的第 21-40 局，評分到下列目錄（scripts/m_series/x40_keys.py）
EXTRAS = {k: {"hsl": ROOT / f"results/scores_hd_x{k}", "vec": ROOT / f"results/trunk_vec_x{k}", "suffix": f"_x{k}"} for k in (40, 80)}  # 額外棋局的 Human SL 與主幹向量目錄：第 21-40 局 / 第 41-80 局
EXTRA = {**EXTRAS[40], "l1b": {f: v + "_x40" for f, v in L1B.items()}, "opp": {f: v + "_x40" for f, v in OPP.items()}}  # x40 這一組（section 4M M8）
for _k, _d in L1B_SETS.items():  # 每組 L1b 的額外棋局目錄：<dir>_x40、<dir>_x80
    _d["l1b_x"], _d["opp_x"] = {f: v + "_x40" for f, v in _d["l1b"].items()}, {f: v + "_x40" for f, v in _d["opp"].items()}  # 第 21-40 局的本人側與對手側目錄
    _d["l1b_x80"], _d["opp_x80"] = {f: v + "_x80" for f, v in _d["l1b"].items()}, {f: v + "_x80" for f, v in _d["opp"].items()}  # 第 41-80 局的本人側與對手側目錄
OPP_HSL = ROOT / "results/scores_opp_hd"  # section 4M M14：對手的著手在同樣 18 個剖面下的評分（scripts/score_folds.py --opponent）
OPP_HSL_X = {40: ROOT / "results/scores_opp_hd_x40"}  # x40 棋局的對手評分；第 41-80 局沒有這樣評分
HSL_TAG = os.environ.get("AICUP_HSL", "")  # section 4M M27：例如 "jp" -> results/scores_hd_jp、scores_hd_x40_jp、scores_opp_hd_jp、scores_opp_hd_x40_jp
if HSL_TAG:  # 改用另一種重播設定的 Human SL 表
    HSL = ROOT / f"results/scores_hd_{HSL_TAG}"  # 20 局樣本的 Human SL 目錄
    EXTRAS[40]["hsl"] = ROOT / f"results/scores_hd_x40_{HSL_TAG}"  # 第 21-40 局的 Human SL 目錄
    OPP_HSL = ROOT / f"results/scores_opp_hd_{HSL_TAG}"  # 對手側的 Human SL 目錄
    OPP_HSL_X = {40: ROOT / f"results/scores_opp_hd_x40_{HSL_TAG}"}  # x40 棋局對手側的 Human SL 目錄
    EXTRA["hsl"] = EXTRAS[40]["hsl"]  # x40 這一組也指向新目錄


def opp_hsl_level(a):  # 這次執行使用對手 HSL 的程度
    """0: no opponent HSL; 1 (--opp-hsl, tag -oh): the opponent plies' Human SL token block (scores columns 11-29) holds the centred
    log p of the opponent's moves instead of zeros; 2 (--opp-hsl-r, tag -ohr): and the 39 opponent-HSL question features join R."""
    return 2 if getattr(a, "opp_hsl_r", False) else (1 if getattr(a, "opp_hsl", False) else 0)  # -ohr 優先；getattr 讓沒有這些屬性的參數物件也能用


def opp_hsl_dir(level_k):  # 額外棋局等級對應的對手 HSL 目錄
    """The opponent-HSL table directory of extra level k (0: the 20-game sample); SystemExit when those games were not scored."""
    if not level_k:  # 0：20 局樣本
        return OPP_HSL  # 基礎目錄
    if level_k not in OPP_HSL_X:  # 這一級的棋局沒有對手 HSL 評分（例如 x80）
        raise SystemExit(f"--opp-hsl: the opponent's moves of the x{level_k} games were not scored under the 18 profiles (x{sorted(OPP_HSL_X)} were)")  # 停止並說明哪些有評分
    return OPP_HSL_X[level_k]  # 該級的目錄


def opp_hsl_table(directory, f, hsl):  # 第 f 折的對手 HSL 表，只留本人側表中的棋局
    """The opponent-HSL scored games of fold f under `directory`, restricted to the games of the target-side table `hsl` (which they must all cover)."""
    t = read_scored_games(directory / f"fold{f}")  # 該折全部已完成分段檔的評分表
    missing = set(hsl["game_key"]) - set(t["game_key"])  # 本人側有、對手側沒有的棋局
    if missing:  # 對手側沒有全部涵蓋
        raise ValueError(f"{directory}/fold{f}: {len(missing)} games of the target-side HSL table have no opponent scores, e.g. {sorted(missing)[:3]}")  # 停止並列出範例
    return t[t["game_key"].isin(set(hsl["game_key"]))].reset_index(drop=True)  # 只留本人側表中的棋局


def extra_level(a):  # 這次執行的額外棋局等級
    """0 (the 20-game sample only), 40 (+ games 21-40) or 80 (+ games 21-80): --extra-games."""
    k = int(getattr(a, "extra_games", 0) or 0)  # 沒給或為 None 時當成 0
    if k not in (0, 40, 80):  # 只支援這三種
        raise SystemExit(f"--extra-games {k}: 40 or 80")  # 其他值直接停止
    if HSL_TAG and k == 80:  # section 4M M27：第 41-80 局沒有在 AICUP_RULES 下評分
        raise SystemExit(f"AICUP_HSL={HSL_TAG}: --extra-games 80 has no {HSL_TAG} Human SL tables")  # 沒有對應的 Human SL 表，停止
    return k  # 回傳等級


def extra_sets(level):  # 要納入的各級額外棋局
    """The extra-game levels included up to `level`: [] / [40] / [40, 80]."""
    return [k for k in (40, 80) if k <= level]  # 不超過 level 的各級


def set_dirs_for(dirs, k):  # 某組 L1b 在額外等級 k 的目錄
    """(l1b map, opp map) of L1b set `dirs` for extra level k (40 or 80)."""
    return (dirs["l1b_x"], dirs["opp_x"]) if k == 40 else (dirs["l1b_x80"], dirs["opp_x80"])  # 40 用 _x40 目錄，80 用 _x80 目錄


def l1b_set_of(a):  # 這次執行使用的 L1b 組
    """The L1b set a run uses (3 unless --l1b-set says otherwise)."""
    k = int(getattr(a, "l1b_set", 3) or 3)  # 預設第 3 組
    if k not in L1B_SETS:  # 不認得的組別
        raise SystemExit(f"--l1b-set {k}: known sets {sorted(L1B_SETS)}")  # 停止並列出已知的組別
    return k  # 回傳組別
N_SCORES = 30  # 每手分數欄數：L1b 為水準＋10 個置中值；Human SL 為水準＋18 個置中值，只在本人手上有值（其他為零；-oh 的陣列對手手也填）
# section 4M M74（2026-11-10）：AICUP_AI=1 時附加 8 個來自非人類強網的逐手「狀態」欄（experiments/ai_state_extract.py）：
# 輪到下的一方的勝率與目數領先、這手棋的代價（目數、勝率），以及強網 policy 對這手棋的看法。
AI_TAG = "ai" if os.environ.get("AICUP_AI") else ""  # 啟用時 tag 加上 ai
N_AI = 8 if AI_TAG else 0  # 狀態欄數（未啟用時為 0）
AI_DIRS = {"base": ROOT / "results/ai_state/base", 40: ROOT / "results/ai_state/x40"}  # 20 局樣本與 x40 棋局的狀態檔目錄
D_TOKEN = 64 + 64 + N_SCORES + N_AI + 4  # 每手 token 寬度：主幹 g、l、分數欄、狀態欄，再加 is_target、mover、t / T、pass


def ai_columns(ai, mover, game_row):  # 強網狀態檔轉成 8 個 token 欄
    """[plies, 8] from ai_state_extract's [plies, 6] (side to move): win, slog(lead), slog(point loss), win loss, log p played,
    log p played - best, entropy, agreement. The loss of a move = the mover's lead before it minus the mover's lead before the next
    ply of the same game (0 on a game's last ply); slog(x) = sign(x) log1p(|x|)."""
    ai = ai.astype(np.float32); win, lead = ai[:, 4], ai[:, 5]  # 轉成 float32；第 4、5 欄是輪到下的一方的勝率與領先
    nxt = np.zeros(len(win), bool); nxt[:-1] = game_row[1:] == game_row[:-1]  # 下一手是否屬於同一局
    same = np.zeros(len(win), bool); same[:-1] = mover[1:] == mover[:-1]  # 下一手是否由同一方下
    lead_n, win_n = np.roll(lead, -1), np.roll(win, -1)  # 下一手之前的領先與勝率
    pl = np.where(nxt, lead - np.where(same, lead_n, -lead_n), 0.0); wl = np.where(nxt, win - np.where(same, win_n, 1.0 - win_n), 0.0)  # 換成落子方視角後的目數損失與勝率損失；每局最後一手為 0
    slog = lambda x: np.sign(x) * np.log1p(np.abs(x))  # noqa: E731  帶正負號的對數壓縮
    return np.stack([win, slog(lead), slog(pl), wl, ai[:, 0], ai[:, 3], ai[:, 2], ai[:, 1]], axis=1)  # 依說明的順序排成 8 欄
HIDDEN = {"C": 110, "T": 64, "S": 192}  # 各臂的隱藏寬度；S（section 4M M21，2026-11-05）：對一局各手的 4 層 pre-norm transformer，全注意力
GAME_DIM, Q_DIM = 64, 32  # 每局向量與每題向量的維度
ARMS = ("R0", "C", "T", "S")  # 所有的臂（compare 的預設）
S_LAYERS, S_HEADS, S_MAX_PLIES = 4, 4, 512  # S 臂的層數、注意力頭數與位置嵌入的手數上限


def centred(logp):  # log p 轉成水準加置中值
    level = logp.mean(axis=1, keepdims=True)  # 每手在各欄上的平均（水準）
    return np.hstack([level, logp - level])  # 第一欄為水準，其後為各欄減去水準


def build(a):  # build 子命令：寫出每折的逐手分數陣列
    """One [plies, 30] fp16 array per fold, in the order of the trunk-vector parts, checked ply by ply."""
    out = Path(a.cache)  # 分數陣列寫到快取目錄
    out.mkdir(parents=True, exist_ok=True)  # 確保目錄存在
    vec_dir = Path(getattr(a, "vec_dir", VEC))  # 分數陣列依這個主幹向量目錄的分段檔順序排列
    extra = extra_level(a)  # 額外棋局等級；非 0 時 vec_dir 是額外棋局的向量目錄
    dirs = L1B_SETS[l1b_set_of(a)]  # 使用的 L1b 組
    oh = opp_hsl_level(a) > 0  # 是否在對手手填入對手的 Human SL
    if extra:  # 只處理「一個」額外向量目錄的分數陣列：等級由目錄名稱決定（trunk_vec_x40 / trunk_vec_x80）
        k = 80 if str(vec_dir).endswith("_x80") else 40  # 由目錄名稱判斷等級
        l1b_dirs, opp_dirs = set_dirs_for(dirs, k)  # 該級的 L1b 本人側與對手側目錄
        hsl_dir = EXTRAS[k]["hsl"]  # 該級的 Human SL 目錄
        oh_dir = opp_hsl_dir(k) if oh else None  # 該級的對手 HSL 目錄
    else:  # 20 局樣本
        l1b_dirs, opp_dirs, hsl_dir = dirs["l1b"], dirs["opp"], HSL  # 基礎目錄
        oh_dir = OPP_HSL if oh else None  # 基礎的對手 HSL 目錄
    k_ai = (80 if str(vec_dir).endswith("_x80") else 40) if extra else "base"  # 強網狀態檔的目錄鍵
    for f in (getattr(a, "build_folds", None) or FOLDS):  # 逐折處理（預設第 1-4 折）
        fold_dir = vec_dir / f"fold{f}"  # 該折的主幹向量目錄
        tgt, opp, hsl = (_part_moves(ROOT / "results/l1b" / l1b_dirs[f] / f"fold{f}"), _part_moves(ROOT / "results/l1b" / opp_dirs[f] / f"fold{f}"), _part_moves(hsl_dir / f"fold{f}"))  # 本人側 L1b、對手側 L1b、Human SL：{局鍵: (表列, log p, 手數編號)}
        ohsl = _part_moves(oh_dir / f"fold{f}") if oh else None  # section 4M M14：對手手在 18 個剖面下的分數
        blocks, bad = [], 0  # 各分段檔的分數區塊與對不上的局數
        for s, k in completed_parts(fold_dir):  # 依主幹向量分段檔的順序
            csv_path, npz_path = part_paths(fold_dir, s, k)  # 這個分段檔的棋局表與向量檔
            games = pd.read_csv(csv_path, usecols=["game_key"])["game_key"].tolist()  # 分段檔內各局的鍵，順序即 game_row
            with np.load(npz_path) as z:  # 開啟向量檔
                row, t, target = z["game_row"], z["t"], z["target"]  # 每手所屬的局、手數編號、是否為本人手
            sc = np.zeros((len(t), N_SCORES + N_AI), np.float32)  # 這個分段檔的分數，預設為零
            if AI_TAG:  # section 4M M74：強網狀態欄，同一個分段檔、同樣的手順序
                with np.load(npz_path) as z:  # 再開啟一次向量檔，取落子方欄
                    mover_ = z["mover"]  # 每手的落子方
                ai = np.load(AI_DIRS[k_ai] / f"fold{f}" / f"ai_s{s}_p{k:03d}.npy")  # 對應這個分段檔的狀態檔
                if len(ai) != len(t):  # 手數必須相同
                    raise ValueError(f"{npz_path}: {len(t)} plies, the state file holds {len(ai)}")  # 對不上就停止
                sc[:, N_SCORES:] = ai_columns(ai, mover_, row)  # 填入 8 個狀態欄
            for g, key in enumerate(games):  # 逐局核對並填入分數
                idx = np.flatnonzero(row == g)  # 這一局在分段檔內的位置
                plies_t, plies_o = t[idx][target[idx]], t[idx][~target[idx]]  # 本人手與對手手的手數編號
                if (not (np.array_equal(plies_t, tgt[key][2]) and np.array_equal(plies_o, opp[key][2]) and np.array_equal(plies_t, hsl[key][2]))  # 三份評分的手數編號都要與向量檔一致
                        or (oh and not (key in ohsl and np.array_equal(plies_o, ohsl[key][2])))):  # 用對手 HSL 時，這一局也要有對手評分且手數一致
                    bad += 1  # 記一局對不上
                    continue  # 跳過，最後整折報錯
                sc[idx[target[idx]], :11] = centred(tgt[key][1])  # 本人手：本人側 L1b 的水準＋10 個置中值
                sc[idx[~target[idx]], :11] = centred(opp[key][1])  # 對手手：對手側 L1b 的分數
                sc[idx[target[idx]], 11:N_SCORES] = centred(hsl[key][1])  # 本人手：Human SL 的水準＋18 個置中值
                if oh:  # 對手手的 Human SL 區塊（一般的陣列裡為零）
                    sc[idx[~target[idx]], 11:N_SCORES] = centred(ohsl[key][1])  # 填入對手的 Human SL 分數
            blocks.append(sc.astype(np.float16))  # 以 fp16 保存
        if bad:  # 有任何一局對不上
            raise ValueError(f"fold {f}: {bad} games whose plies do not line up with the stored scores")  # 整折停止，不寫出不完整的陣列
        arr = np.concatenate(blocks)  # 依分段檔順序接起來
        path = scores_path(out, f, vec_dir, l1b_set_of(a), oh)  # 依設定決定檔名
        np.save(path, arr)  # 寫出分數陣列
        print(f"fold {f}: {len(arr)} plies, every game lined up ({path.name})", flush=True)  # 回報手數與檔名


_QUANT8 = {}  # int8 主幹向量的統計快取：{折目錄: (mu, sd)}


def part_gl(z, fold_dir):  # 從已開啟的分段檔取出主幹向量 g、l
    """(g, l) [plies, 64] of an opened trunk-vector part: stored as fp16, or (section 4M M135, trunk_extract.py --quant8) as int8 in
    units of sd / 20 around mu (keys gq, lq; the statistics in <fold dir>/quant8.npz), restored here to fp16."""
    if "g" in z.files:  # fp16 儲存的分段檔
        return z["g"], z["l"]  # 直接回傳
    key = str(fold_dir)  # 快取鍵
    if key not in _QUANT8:  # 這個目錄的統計還沒讀過
        st = np.load(Path(fold_dir) / "quant8.npz")  # 讀量化統計
        _QUANT8[key] = (st["mu"].astype(np.float32), st["sd"].astype(np.float32))  # 存進快取
    mu, sd = _QUANT8[key]  # 平均與標準差（前段屬於 g，後段屬於 l）
    n = z["gq"].shape[1]  # g 的欄數
    return (z["gq"].astype(np.float32) / 20.0 * sd[:n] + mu[:n]).astype(np.float16), (z["lq"].astype(np.float32) / 20.0 * sd[n:] + mu[n:]).astype(np.float16)  # 以 sd / 20 為單位還原 g 與 l


EXT_FOLD = 9  # section 4M M135：外部玩家集合的假折號


def ext_dirs(root):  # 外部玩家集合的檔案位置
    """Section 4M M135: the files of an external-players set, the competition's layout under one root with pseudo-fold EXT_FOLD
    (experiments/nano4/m135_ext_prep.py writes the game tables; the scoring scripts run with AICUP_TRAIN_DIR / AICUP_FOLDS_CSV)."""
    root = Path(root)  # 根目錄
    f = f"fold{EXT_FOLD}"  # 假折的目錄名稱
    return {"vec": root / "trunk_vec" / f, "hsl": root / "scores_hd" / f, "ohsl": root / "scores_opp_hd" / f, "l1b": root / "l1b" / "scores3_hd" / f,  # 主幹向量、Human SL 兩側與本人側 L1b 的目錄
            "opp": root / "l1b" / "scores3_opp_hd" / f, "scores": root / f"scores_oh_{f}.npy", "index": root / f"scores_index_{f}.npz", "hd": root / "hd_table.csv",  # 對手側 L1b、分數陣列、索引與讓子表
            "folds": root / "player_folds.csv", "games": {k: root / f"games_{k}.csv" for k in ("hsl", "l1b", "opp")}}  # 分折表與 R 要讀的三張棋局表


def ext_rank_order(r):  # 外部段位名稱的排序鍵
    """Sort key of an external rank name (F18k ... F1k, F1d ... F9d), weakest first."""
    r = r[1:] if r[:1] == "F" else r  # 去掉段位名稱開頭的 F
    return -int(r[:-1]) if r.endswith("k") else int(r[:-1]) - 1  # 級位取負（18k 最小），段位從 0 起算


def ext_build(a):  # build --ext：外部集合的分數陣列、索引與棋局表
    """`build --ext ROOT`: the external set's [plies, 30] fp16 score array (the -oh layout: L1b level + 10 centred, Human SL level + 18
    centred, on the target's and on the opponent's plies), game after game in sorted key order, with its index (key, start, n,
    and every ply's is-target flag) and the three game tables R reads (games_hsl / l1b / opp.csv). It does not need the trunk
    vectors: a game's plies are the union of its scored target and opponent moves, which must be 0 .. n - 1 exactly, and the
    Human SL and L1b passes must have scored the same moves. ext_tokens checks the trunk parts against the index when it loads."""
    e = ext_dirs(a.ext)  # 檔案位置
    tgt, opp, hsl, ohsl = (_part_moves(e[k]) for k in ("l1b", "opp", "hsl", "ohsl"))  # 四份評分的 {局鍵: (表列, log p, 手數編號)}
    if not (set(tgt) == set(opp) == set(hsl) == set(ohsl)):  # 四份評分必須是同一批棋局
        raise ValueError(f"external set {a.ext}: the four scored directories hold different games ({len(tgt)} / {len(opp)} / {len(hsl)} / {len(ohsl)})")  # 回報各份的局數
    keys = sorted(hsl)  # 依局鍵排序，決定陣列中的棋局順序
    blocks, flags, lens, bad = [], [], [], []  # 分數區塊、本人手旗標、各局手數、對不上的局
    for key in keys:  # 逐局處理
        tt, to = tgt[key][2].astype(np.int64), opp[key][2].astype(np.int64)  # 本人手與對手手的手數編號
        n = len(tt) + len(to)  # 這一局的總手數
        if not (np.array_equal(tt, hsl[key][2]) and np.array_equal(to, ohsl[key][2]) and np.array_equal(np.sort(np.r_[tt, to]), np.arange(n))):  # Human SL 與 L1b 評到同樣的手，且兩側合起來剛好是 0 .. n - 1
            bad.append(key)  # 記下這一局
            continue  # 跳過
        sc = np.zeros((n, N_SCORES), np.float32)  # 這一局的分數
        sc[tt, :11], sc[to, :11] = centred(tgt[key][1]), centred(opp[key][1])  # L1b：本人手放本人側、對手手放對手側
        sc[tt, 11:N_SCORES], sc[to, 11:N_SCORES] = centred(hsl[key][1]), centred(ohsl[key][1])  # Human SL：兩側都填（-oh 的配置）
        fl = np.zeros(n, bool)  # 每手是否為本人手
        fl[tt] = True  # 標記本人手
        blocks.append(sc.astype(np.float16)); flags.append(fl); lens.append(n)  # 收集分數、旗標與手數
    if bad:  # 有對不上的局
        raise ValueError(f"external set {a.ext}: {len(bad)} of {len(keys)} games whose scored moves do not line up, e.g. {bad[:3]}")  # 停止並列出範例
    arr, lens = np.concatenate(blocks), np.asarray(lens, dtype=np.int64)  # 整份分數陣列與各局手數
    if not np.isfinite(arr).all():  # 檢查 NaN 與無限大
        raise ValueError(f"external set {a.ext}: non-finite scores")  # 有就停止
    np.save(e["scores"], arr)  # 寫出分數陣列
    np.savez(e["index"], key=np.array(keys), start=np.cumsum(lens) - lens, n=lens, target=np.concatenate(flags))  # 寫出索引：局鍵、起點、手數與每手的本人手旗標
    for name, src in (("hsl", "hsl"), ("l1b", "l1b"), ("opp", "opp")):  # R 要讀的三張棋局表
        read_scored_games(e[src]).to_csv(e["games"][name], index=False)  # 合併各評分分段檔的棋局表後寫成 CSV
    print(f"external set: {len(keys)} games, {len(arr)} plies, every game lined up ({e['scores']}, index and game tables written)", flush=True)  # 回報局數與手數
    return 0  # 成功


def ext_tokens(root, device, stats, drop=None):  # 載入外部集合的 token
    """Section 4M M135: the external set's tokens (standardised with `stats`, fp16, in device memory) and its game table. The trunk
    parts come in their own order; each game's scores are gathered from the score array through its index, and its ply count and
    is-target flags must agree with the index (ValueError otherwise). Flags as raw_tokens."""
    e = ext_dirs(root)  # 檔案位置
    ix = np.load(e["index"])  # ext_build 寫出的索引
    sc, tflag = np.load(e["scores"]), ix["target"]  # 分數陣列與每手的本人手旗標
    where = {k: (int(s), int(n)) for k, s, n in zip(ix["key"].astype(str), ix["start"], ix["n"])}  # 局鍵 -> (起點, 手數)
    out, offsets, start = [], {}, 0  # token 區塊、局表與目前的起點
    for s, k in completed_parts(e["vec"]):  # 依主幹向量分段檔自己的順序
        csv_path, npz_path = part_paths(e["vec"], s, k)  # 分段檔的棋局表與向量檔
        tab = pd.read_csv(csv_path, usecols=["game_key", "n_plies"], dtype={"game_key": str}, keep_default_na=False)  # 局鍵（保留為字串）與手數
        with np.load(npz_path) as z:  # 開啟向量檔
            (g, l), pos, mover, target, t = part_gl(z, e["vec"]), z["pos"], z["mover"], z["target"], z["t"]  # 主幹向量、落點、落子方、本人手旗標、手數編號
        lens = tab["n_plies"].to_numpy()  # 各局手數
        missing = [key for key in tab["game_key"] if key not in where]  # 分數陣列中沒有的局
        if missing:  # 有主幹向量卻沒有分數
            raise ValueError(f"{npz_path}: {len(missing)} games have no scores in the external score array, e.g. {missing[:3]}")  # 停止並列出範例
        rows = np.concatenate([np.arange(where[key][0], where[key][0] + where[key][1]) for key in tab["game_key"]])  # 依分段檔的局順序取出分數陣列的列
        if len(rows) != len(t) or any(where[key][1] != n for key, n in zip(tab["game_key"], lens)) or not np.array_equal(tflag[rows], target):  # 手數與本人手旗標都要與索引一致
            raise ValueError(f"{npz_path}: the plies of its games do not line up with the external score array")  # 對不上就停止
        T = np.repeat(lens, lens).astype(np.float32)  # 每手所屬局的總手數
        fl = np.stack([target.astype(np.float32), mover.astype(np.float32), t / T, (pos == 361).astype(np.float32)], axis=1)  # 四個旗標：本人手、落子方、t / T、pass（落點 361）
        out.append(np.hstack([g, l, sc[rows], fl.astype(np.float16)]))  # 這個分段檔的 token
        for key, n in zip(tab["game_key"], lens):  # 記錄每局在 token 陣列中的位置
            offsets[key] = (start, int(n))  # (起點, 手數)
            start += int(n)  # 下一局的起點
    if not out:  # 沒有任何完成的分段檔
        raise FileNotFoundError(f"no finished trunk-vector parts in {e['vec']}")  # 停止
    return torch.as_tensor(standardise_tokens(np.concatenate(out), stats, drop), device=device), offsets  # 以傳入的統計標準化後放上裝置


def ext_questions(root, pool=0, per_unit=32):  # 外部集合的訓練題與 R 特徵（有快取）
    """Section 4M M135: the external set's training questions (per unit `per_unit` subsets of 5 .. min(20, its games) games, as the
    competition's training questions), their 64 R features (the same function; the handicap block is that of even games), the
    rank index (weakest first) and the rank names; cached in ROOT/r_ext_p<pool>.npz + .json."""
    root = Path(root)  # 外部集合的根目錄
    e = ext_dirs(root)  # 檔案位置
    path, meta_path = root / f"r_ext_p{pool}.npz", root / f"r_ext_p{pool}.json"  # 特徵與題目表的快取檔
    if path.exists() and meta_path.exists():  # 已有快取
        z, m = np.load(path), json.loads(meta_path.read_text())  # 讀回特徵與題目表
        return z["X"], z["y"], pd.DataFrame(m["questions"]), m["ranks"]  # 直接回傳
    str_cols = {"game_key": str, "player_id": str, "rank": str, "color": str}  # 這些欄一律讀成字串
    hsl, l1b, opp = (pd.read_csv(e["games"][k], dtype=str_cols, keep_default_na=False) for k in ("hsl", "l1b", "opp"))  # 由 ext_build 寫出的三張棋局表
    if not (set(hsl["game_key"]) == set(l1b["game_key"]) == set(opp["game_key"])):  # 三張表必須是同一批棋局
        raise ValueError(f"external set {root}: the game tables hold different games")  # 停止
    hd = read_hd_table(e["hd"])  # 外部集合的讓子表
    hd = hd[hd["game_key"].isin(set(hsl["game_key"]))]  # 只留評分過的棋局
    players = hsl.groupby(["rank", "player_id"], sort=True)["game_key"].apply(list).reset_index().rename(columns={"game_key": "game_keys"})  # 每個單位（段位, 帳號）的棋局清單
    q = sample_questions(players, per_unit, np.random.RandomState(l1b_cv.TRAIN_SEED + 9000 + 1000 * pool), sizes=range(5, 21))  # 與比賽訓練題同法抽題，種子另外錯開
    X, _, _, _ = hd_check.features(hsl, l1b, opp, hd, q)  # 用同一個函式算 R 特徵
    ranks = sorted(set(q["rank"]), key=ext_rank_order)  # 出現的段位，最弱的在前
    y = q["rank"].map({r: i for i, r in enumerate(ranks)}).to_numpy(dtype=np.int64)  # 段位索引作為標籤
    cols = ["player_id", "rank", "n_games", "game_keys"]  # 存進快取的題目欄位
    tmp = root / f"r_ext_p{pool}.tmp{os.getpid()}.npz"  # 先寫暫存檔（帶行程編號，避免同時執行時互相覆蓋）
    np.savez(tmp, X=X["fused_opp_diff_hd"], y=y)  # 寫入 64 個 R 特徵與標籤
    os.replace(tmp, path)  # 原子替換成正式快取
    tmpj = root / f"r_ext_p{pool}.tmp{os.getpid()}.json"  # 題目表的暫存檔
    tmpj.write_text(json.dumps({"questions": q[cols].to_dict("list"), "ranks": ranks}))  # 寫入題目表與段位名稱
    os.replace(tmpj, meta_path)  # 原子替換
    return X["fused_opp_diff_hd"], y, q[cols], ranks  # 回傳特徵、標籤、題目表與段位名稱


def raw_tokens(parts):  # 讀出未標準化的 token 與局表
    """Unstandardised tokens (X [plies, D_TOKEN] float16: trunk g, l, the 30 scores, is_target, mover, t / T, pass) and the game
    table {game_key: (start, length)} of trunk-vector directories with their score arrays. parts: [(vec_dir, scores.npy)], the
    score array in the ply order of completed_parts(vec_dir) (what build and scripts/trunk_predict.py write)."""
    out, offsets, start = [], {}, 0  # token 區塊、局表與目前的起點
    for fold_dir, sc_path in parts:  # 逐一處理（向量目錄, 分數陣列）
        sc = np.load(sc_path, mmap_mode="r")  # 以記憶體映射讀分數陣列，不整份載入
        o = 0  # 分數陣列中目前的位置
        for s, k in completed_parts(fold_dir):  # 依分段檔的順序
            csv_path, npz_path = part_paths(fold_dir, s, k)  # 分段檔的棋局表與向量檔
            tab = pd.read_csv(csv_path, usecols=["game_key", "n_plies"], dtype={"game_key": str}, keep_default_na=False)  # 局鍵（保留為字串）與手數
            with np.load(npz_path) as z:  # 開啟向量檔
                (g, l), pos, mover, target, t = part_gl(z, fold_dir), z["pos"], z["mover"], z["target"], z["t"]  # 主幹向量、落點、落子方、本人手旗標、手數編號
            lens = tab["n_plies"].to_numpy()  # 各局手數
            T = np.repeat(lens, lens).astype(np.float32)  # 每手所屬局的總手數
            flags = np.stack([target.astype(np.float32), mover.astype(np.float32), t / T, (pos == 361).astype(np.float32)], axis=1)  # 四個旗標：本人手、落子方、t / T、pass（落點 361）
            out.append(np.hstack([g, l, np.asarray(sc[o:o + len(t)]), flags.astype(np.float16)]))  # 接上這個分段檔對應的分數列
            for key, n in zip(tab["game_key"], lens):  # 記錄每局的位置
                offsets[key] = (start, int(n))  # (起點, 手數)
                start += int(n)  # 下一局的起點
            o += len(t)  # 分數陣列往後移
        if o != len(sc):  # 分數陣列的手數必須剛好用完
            raise ValueError(f"{sc_path}: {len(sc)} plies, the parts of {fold_dir} hold {o}")  # 對不上代表陣列與向量目錄不配套
    return np.concatenate(out), offsets  # 回傳整份 token 與局表


def token_stats(X):  # 訓練端的 token 標準化統計
    """The training-side standardisation of the first D_TOKEN - 4 columns: (mean, std) of about 2M plies spread over X."""
    head = X[:: max(1, len(X) // 2_000_000), :D_TOKEN - 4].astype(np.float32)  # 等間隔抽約 200 萬手；最後四個旗標欄不標準化
    return head.mean(axis=0), head.std(axis=0) + 1e-6  # 平均與標準差（加小數避免除以零）


def standardise_tokens(X, stats, drop=None):  # 就地標準化 token
    """In place: the first D_TOKEN - 4 columns standardised with `stats` (fp16 storage); an ablation zeroes the trunk or score columns."""
    X[:, :D_TOKEN - 4] = ((X[:, :D_TOKEN - 4].astype(np.float32) - stats[0]) / stats[1]).astype(np.float16)  # 以 float32 計算，存回 fp16
    if drop == "trunk":  # 消融：欄位保留，所以每個臂的參數量都相同
        X[:, :128] = 0  # 主幹向量 g、l 歸零
    elif drop == "scores":  # 消融：拿掉逐手分數
        X[:, 128:128 + N_SCORES] = 0  # 分數欄歸零
    return X  # 回傳同一個陣列


def load_token_dirs(parts, device, stats=None, drop=None):  # 讀 token、標準化並放上裝置
    """raw_tokens of `parts`, standardised (with `stats`, else with their own token_stats) in device memory: (X, offsets, stats)."""
    X, offsets = raw_tokens(parts)  # 未標準化的 token 與局表
    if stats is None:  # 沒給統計（訓練端）
        stats = token_stats(X)  # 用這些資料自己算
    return torch.as_tensor(standardise_tokens(X, stats, drop), device=device), offsets, stats  # 標準化後放上裝置，並回傳所用的統計


def load_tokens(folds, cache, device, stats=None, drop=None, vec_dir=VEC, extra=0, l1b_set=3, opp_hsl=False):  # 載入指定各折的 token
    """Tokens of `folds` in device memory (fp16), the game table {key: (start, length)}, and the training-side standardisation;
    extra (section 4M M8: 40, x80: 80) also loads the folds' games 21-40 (and 41-80) from EXTRAS; l1b_set and opp_hsl pick the score arrays."""
    parts = [(Path(vec_dir) / f"fold{f}", scores_path(cache, f, vec_dir, l1b_set, opp_hsl)) for f in folds]  # 各折的向量目錄與分數陣列
    for k in extra_sets(int(extra or 0)):  # 需要額外棋局時
        parts += [(EXTRAS[k]["vec"] / f"fold{f}", scores_path(cache, f, EXTRAS[k]["vec"], l1b_set, opp_hsl)) for f in folds]  # 加上該級額外棋局的向量與分數
    return load_token_dirs(parts, device, stats, drop)  # 讀入並標準化


def fold_code(v):  # 抽樣種子用的折編號（all 為 5）
    """The integer that seeds a run's draws: the validation fold, or 5 for a run on all four folds ("all")."""
    return 5 if v == "all" else int(v)  # "all" 用 5，其餘為驗證折編號


def training_folds(v, n_train=0, with_fold0=False):  # 哪些折的玩家參與訓練
    """The folds whose players train for validation fold v: every other fold, or (section 4M M1) only n_train of them, dropping
    fold (v mod 4) + 1 first, so that over v = 1..4 every fold is dropped once. "all": every fold, plus fold 0 with with_fold0
    (section 4M M8, the submission's models only)."""
    if v == "all":  # 提交用的模型
        if n_train:  # all 不能只用部分的折
            raise SystemExit("--train-folds applies to a validation fold, not to 'all'")  # 停止
        return ([0] if with_fold0 else []) + list(FOLDS)  # 四折全訓，with_fold0 時再加上第 0 折
    others = [f for f in FOLDS if f != int(v)]  # 驗證折以外的各折
    if with_fold0:  # section 4M M20（2026-11-05）：正式配方的交叉驗證——第 0 折參與訓練，留出第 v 折（tag -f0）
        others = [0] + others  # 第 0 折也參與訓練
    if not n_train or n_train >= len(others):  # 沒有限制折數，或限制不小於可用的折數
        return others  # 全部使用
    if n_train < 1:  # 不合理的折數（負值）
        raise SystemExit("--train-folds takes 1-3")  # 停止
    drop = int(v) % 4 + 1  # 固定先丟掉第 (v mod 4) + 1 折：v = 1..4 時每一折各被丟一次
    out = [f for f in others if f != drop]  # 去掉該折
    return out[:n_train]  # 取前 n_train 折


def cap_games(players, n_games, seed=4242):  # 限制每位玩家可用的評分局數
    """Section 4M M2: every player keeps at most n_games of its scored games, chosen by a permutation seeded from the player id
    (independent of row order and of the other players, like scoring.fold_players); players with fewer games are unchanged."""
    if not n_games:  # 0 表示不限制
        return players  # 原樣回傳
    out = players.copy()  # 不改動呼叫端的表
    out["game_keys"] = [list(k) if len(k) <= n_games else [k[i] for i in np.random.RandomState((int(str(p)[:8], 16) + seed) % 2**32).permutation(len(k))[:n_games]]  # 超過上限時，以玩家編號為種子的排列取前 n_games 局
                        for p, k in zip(out["player_id"], out["game_keys"])]  # 逐位玩家
    return out  # 回傳限制後的表


def data_segments(a):  # 組出 tag 中描述訓練資料的各段
    """The training-data part of a run's tag, as segments (sections 4L.4 and 4M): f<train folds>, g<games per player>[s<min>-<max>],
    s<min>-<max> (question sizes), mix<pct>. Joined with "-" in the arm tag and with "_" in the R cache tag."""
    n_train, games, sizes = int(getattr(a, "train_folds", 0) or 0), int(getattr(a, "train_games", 0) or 0), getattr(a, "train_sizes", None)  # 訓練折數、每人局數上限、題目大小範圍
    mix = float(getattr(a, "mix", 0.0) or 0.0)  # 混人題比例
    seg = []  # 收集各段
    if getattr(a, "with_fold0", False):  # 第 0 折參與訓練
        seg.append("f0")  # tag 段 -f0
    if n_train:  # 只用部分的折訓練
        seg.append(f"f{n_train}")  # -f<折數>
    if int(getattr(a, "l1b_set", 3) or 3) != 3:  # 非預設的 L1b 組
        seg.append(f"h{int(a.l1b_set)}")  # -h<組>
    if games or sizes:  # 限制局數或題目大小
        seg.append((f"g{games}" if games else "") + (f"s{sizes[0]}-{sizes[1]}" if sizes else ""))  # -g<n>s<min>-<max>，或其中一段
    if int(getattr(a, "extra_games", 0) or 0):  # 用了額外棋局
        seg.append(f"x{int(a.extra_games)}")  # 額外棋局等級：-x40 / -x80
    if opp_hsl_level(a):  # section 4M M14：用了對手 HSL
        seg.append("oh" if opp_hsl_level(a) == 1 else "ohr")  # -oh 或 -ohr
    if mix > 0:  # 有混人題
        seg.append(f"mix{round(mix * 100)}")  # -mix<百分比>
    if HSL_TAG:  # section 4M M27：另一種重播設定的 Human SL 表
        seg.append(HSL_TAG)  # 例如 -jp
    if AI_TAG:  # section 4M M74：token 含強網的狀態欄
        seg.append(AI_TAG)  # 強網狀態欄：-ai
    return seg  # 回傳各段


def vec_tag(vec_dir):  # 主幹向量目錄對應的 tag
    """"" for the default trunk vectors, else "-<name>" with the trunk_vec_ prefix stripped (results/trunk_vec_ft -> "-ft")."""
    p = Path(vec_dir)  # 轉成路徑
    if p.resolve() == VEC.resolve():  # 預設的主幹向量
        return ""  # 不加 tag
    return "-" + (p.name[len("trunk_vec_"):] if p.name.startswith("trunk_vec_") else p.name)  # 去掉 trunk_vec_ 前綴後作為 tag


def scores_path(cache, f, vec_dir=VEC, l1b_set=3, opp_hsl=False):  # 分數陣列的檔名
    """The per-ply score array of fold f laid out in the part order of vec_dir (build writes it): scores[_<vec tag>][_h<set>][_oh]_fold<f>.npy
    (section 4M M14: _oh = the opponent plies' Human SL block filled)."""
    return Path(cache) / f"scores{vec_tag(vec_dir).replace('-', '_')}{'' if l1b_set == 3 else f'_h{l1b_set}'}{'_oh' if opp_hsl else ''}{'_' + HSL_TAG if HSL_TAG else ''}{'_' + AI_TAG if AI_TAG else ''}_fold{f}.npy"  # 依主幹向量、L1b 組、對手 HSL、Human SL 設定與強網狀態組出檔名


CKPT_FORMAT = 1  # 檢查點格式版本；讀取時必須相同


def save_checkpoint(path, model, token_stats_, r_stats, meta):  # 存檢查點與附屬檔
    """<path> (.pt): the model's state (cpu) with the token and R standardisation it was trained with; the sidecar <stem>.json
    records the sha256, the token layout constants and `meta` (the training facts). Returns the sidecar dict."""
    from aicup.l1b import atomic_save  # 原子寫入 .pt
    from aicup.ladder import environment, to_json  # 執行環境資訊與 JSON 輸出
    from aicup.setmodel_io import file_sha256, sidecar_path  # 檔案雜湊與附屬檔路徑
    path = Path(path)  # 轉成路徑
    path.parent.mkdir(parents=True, exist_ok=True)  # 確保目錄存在
    n_r = int(model.mlp[0].in_features - (0 if model.arm == "R0" else Q_DIM))  # R 特徵欄數 = MLP 輸入寬度減去題目向量
    obj = {"format": CKPT_FORMAT, "arm": model.arm, "n_r": n_r, "pool": model.pool, "token_enc": getattr(model, "token_enc", "linear"), "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},  # 格式、臂、R 欄數、池化、token 編碼器與移到 CPU 的權重
           "token_mean": None if token_stats_ is None else torch.as_tensor(np.asarray(token_stats_[0], dtype=np.float32)),  # token 標準化的平均（R0 沒有）
           "token_std": None if token_stats_ is None else torch.as_tensor(np.asarray(token_stats_[1], dtype=np.float32)),  # token 標準化的標準差
           "r_mean": torch.as_tensor(np.asarray(r_stats[0], dtype=np.float32)), "r_std": torch.as_tensor(np.asarray(r_stats[1], dtype=np.float32))}  # R 特徵標準化的平均與標準差
    atomic_save(obj, path)  # 原子寫入，不會留下寫一半的檔
    side = {"format": CKPT_FORMAT, "arm": model.arm, "n_r": n_r, "pool": model.pool, "token_enc": getattr(model, "token_enc", "linear"), "state_sha256": file_sha256(path), "d_token": D_TOKEN, "n_scores": N_SCORES,  # 附屬檔：架構、權重檔雜湊與 token 配置常數
            "hidden": HIDDEN.get(model.arm), "game_dim": GAME_DIM, "q_dim": Q_DIM, **meta,  # 寬度設定與訓練事實
            **environment("cuda" if torch.cuda.is_available() else "cpu")}  # 執行環境：程式版本、torch 版本與裝置名稱
    sidecar_path(path).write_text(to_json(side, indent=1) + "\n")  # 寫出 <stem>.json
    return side  # 回傳附屬檔內容


def load_checkpoint(path, device="cpu"):  # 讀回 save_checkpoint 的檔並核對
    """(Model in eval mode on `device`, token stats (mean, std) or None, (r_mean, r_std), sidecar) of a save_checkpoint file.
    ValueError when the sidecar is missing, its format is not this code's, the .pt sha256 differs, or the token layout differs."""
    from aicup.setmodel_io import file_sha256, sidecar_path  # 檔案雜湊與附屬檔路徑
    path = Path(path)  # 轉成路徑
    side_path = sidecar_path(path)  # 對應的 <stem>.json
    if not side_path.exists():  # 沒有附屬檔
        raise ValueError(f"{path}: sidecar {side_path.name} is missing")  # 無法核對，停止
    side = json.loads(side_path.read_text())  # 讀附屬檔
    if side.get("format") != CKPT_FORMAT:  # 格式版本不同
        raise ValueError(f"{path}: sidecar format {side.get('format')!r}, this code reads {CKPT_FORMAT}")  # 停止
    sha = file_sha256(path)  # 算 .pt 的 sha256
    if side.get("state_sha256") != sha:  # 與附屬檔記錄的不同
        raise ValueError(f"{path}: sha256 {sha[:12]} differs from the sidecar's {str(side.get('state_sha256'))[:12]}")  # 權重檔與附屬檔不配套，停止
    layout = (side.get("d_token"), side.get("n_scores"), side.get("game_dim"), side.get("q_dim"))  # 訓練時的 token 配置
    if layout != (D_TOKEN, N_SCORES, GAME_DIM, Q_DIM):  # 與目前程式的配置不同（例如 AICUP_AI 的設定不同）
        raise ValueError(f"{path}: token layout {layout} is not this code's {(D_TOKEN, N_SCORES, GAME_DIM, Q_DIM)}")  # 停止
    obj = torch.load(path, map_location="cpu", weights_only=True)  # 只載入張量與基本型別
    model = Model(obj["arm"], int(obj["n_r"]), pool=obj.get("pool", "meanstd"), token_enc=obj.get("token_enc", "linear"))  # 依檢查點重建同樣結構的模型（舊檔缺的欄位用預設值）
    model.load_state_dict(obj["state"])  # 載入權重
    model.to(device).eval()  # 移到裝置並切到評估模式
    stats = None if obj["token_mean"] is None else (obj["token_mean"].numpy(), obj["token_std"].numpy())  # token 標準化統計（R0 沒有）
    return model, stats, (obj["r_mean"].numpy(), obj["r_std"].numpy()), side  # 回傳模型、兩組統計與附屬檔


class Block(torch.nn.Module):  # C、T 臂的殘差區塊
    def __init__(self, d, kernel, dilation, dropout):  # 建立一個區塊
        super().__init__()  # 初始化 nn.Module
        self.norm = torch.nn.LayerNorm(d)  # 卷積前的正規化（pre-norm）
        self.conv = torch.nn.Conv1d(d, d, kernel, padding=dilation * (kernel - 1) // 2, dilation=dilation)  # 沿手的方向卷積；padding 讓長度不變
        self.drop = torch.nn.Dropout(dropout)  # 殘差分支的 dropout

    def forward(self, x, mask):  # 前向：x [G, L, d]，mask 標出真實的手
        h = self.conv(torch.relu(self.norm(x)).transpose(1, 2)).transpose(1, 2)  # 正規化、ReLU 後卷積（Conv1d 需要 [G, d, L]）
        return x + self.drop(h) * mask  # 殘差相加，填充的位置不更新


def masked_mean_std(x, w, dim):  # 帶權重（遮罩）的平均與標準差
    n = w.sum(dim=dim, keepdim=True).clamp(min=1.0)  # 有效個數，至少為 1 以免除以零
    m = (x * w).sum(dim=dim, keepdim=True) / n  # 加權平均
    v = (((x - m) ** 2) * w).sum(dim=dim, keepdim=True) / n  # 加權變異數
    return m.squeeze(dim), (v + 1e-6).sqrt().squeeze(dim)  # 平均與標準差（加小數讓開根號穩定）


class PMA(torch.nn.Module):  # 多頭注意力池化（M4）
    """Pooling by multihead attention (Lee et al. 2019, Set Transformer): k learned seed vectors attend over a masked set [B, N, d]
    (pad True where padding) and come back as one [B, k * d] vector (seed + attention output). Every row needs one unpadded element."""

    def __init__(self, d, k, heads, dropout):  # 建立 k 個種子向量的池化層
        super().__init__()  # 初始化 nn.Module
        self.seed = torch.nn.Parameter(torch.randn(1, k, d) / d ** 0.5)  # 學習的種子向量（作為查詢）
        self.norm = torch.nn.LayerNorm(d)  # 只正規化 key / value
        self.att = torch.nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)  # 多頭注意力

    def forward(self, x, pad):  # x [B, N, d]，pad 為 True 處是填充
        q = self.seed.expand(x.shape[0], -1, -1)  # 每一列共用同一組種子
        kv = self.norm(x)  # 正規化後作為 key 與 value
        h, _ = self.att(q, kv, kv, key_padding_mask=pad, need_weights=False)  # 種子對集合做注意力，忽略填充
        return (q + h).flatten(1)  # 種子加注意力輸出，攤平成 [B, k * d]


POOLINGS = ("meanstd", "pma", "sumll")  # 可選的池化；sumll（section 4M M134）：mean/std 路徑之外，再加上逐局對數概似相加的通道
TOKEN_ENCODERS = ("linear", "dual", "clip")  # section 4M M30：對 162 欄做一個 Linear，或盤面塔加分數塔（＋ CLIP 對齊）
DUAL_H = (48, 24)  # M30：盤面塔（主幹 g、l：128 欄）與分數塔（L1b ＋ Human SL：30 欄）的寬度
CLIP_DIM, CLIP_PLIES = 32, 2048  # M30 clip：對齊頭的投影寬度，以及每批次為 InfoNCE 抽樣的手數
PC_TEMP = 0.1  # section 4M M33：玩家對比 InfoNCE 的溫度


def paired_perm(pid, rng):  # 同一玩家兩兩相鄰的題目排列
    """Section 4M M33: a permutation of the question indices in which positions 2k and 2k + 1 hold two questions of the same player;
    each player's questions are shuffled and paired (an odd one out is dropped), then the pairs are shuffled."""
    pid = np.asarray(pid)  # 每題的玩家編號
    order = np.argsort(pid, kind="stable")  # 依玩家排序，同一玩家的題目相鄰
    bounds = np.flatnonzero(np.r_[True, pid[order][1:] != pid[order][:-1], True])  # 各玩家區段的邊界
    pairs = []  # 收集配對
    for lo, hi in zip(bounds[:-1], bounds[1:]):  # 逐位玩家
        idx = order[lo:hi][rng.permutation(hi - lo)]  # 打亂這位玩家的題目
        pairs += [idx[i:i + 2] for i in range(0, len(idx) - 1, 2)]  # 兩兩配對，落單的一題丟掉
    return np.concatenate([pairs[i] for i in rng.permutation(len(pairs))]) if pairs else np.zeros(0, int)  # 打亂配對的順序後接起來


def grouped_perm(pid, rng, k):  # 同一玩家每 k 題一組的題目排列
    """Section 4M M39: as paired_perm with groups of k same-player questions (a player's leftover < k questions are dropped)."""
    pid = np.asarray(pid)  # 每題的玩家編號
    order = np.argsort(pid, kind="stable")  # 依玩家排序
    bounds = np.flatnonzero(np.r_[True, pid[order][1:] != pid[order][:-1], True])  # 各玩家區段的邊界
    groups = []  # 收集各組
    for lo, hi in zip(bounds[:-1], bounds[1:]):  # 逐位玩家
        idx = order[lo:hi][rng.permutation(hi - lo)]  # 打亂這位玩家的題目
        groups += [idx[i:i + k] for i in range(0, len(idx) - k + 1, k)]  # 每 k 題一組，不足 k 題的丟掉
    return np.concatenate([groups[i] for i in rng.permutation(len(groups))]) if groups else np.zeros(0, int)  # 打亂各組的順序後接起來


def supcon_loss(z, head, k, temp):  # 監督式對比損失（M39）
    """Section 4M M39: supervised-contrastive loss over rows grouped by k (rows k*g .. k*g + k - 1 are one player's questions)."""
    h = torch.nn.functional.normalize(head(z), dim=1)  # 投影後 L2 正規化
    B = h.shape[0]  # 批次大小
    grp = torch.arange(B, device=z.device) // k  # 每列所屬的組（同一玩家）
    logits = h @ h.T / temp  # 兩兩相似度除以溫度
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)  # 自己對自己的位置
    logits = logits.masked_fill(self_mask, float("-inf"))  # 排除自己
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)  # 每列的 log softmax
    pos = (grp[:, None] == grp[None, :]) & ~self_mask  # 正例：同組的其他列
    return -(logp.masked_fill(~pos, 0.0).sum(dim=1) / pos.sum(dim=1)).mean()  # 每列對正例的平均 log 機率取負，再對批次平均


def player_contrast_loss(z, head, temp=PC_TEMP):  # 玩家對比損失（M33）
    """Section 4M M33: symmetric InfoNCE between rows 2k and 2k + 1 of z [2m, d] (two questions of one player), the other pairs as negatives."""
    h = torch.nn.functional.normalize(head(z), dim=1)  # 投影後 L2 正規化
    a, b = h[0::2], h[1::2]  # 每對的第一題與第二題
    logits = a @ b.T / temp  # 第一題對所有第二題的相似度
    target = torch.arange(a.shape[0], device=z.device)  # 正確的配對在對角線上
    return 0.5 * (torch.nn.functional.cross_entropy(logits, target) + torch.nn.functional.cross_entropy(logits.T, target))  # 兩個方向的交叉熵平均（對稱）


class Tower(torch.nn.Module):  # 雙塔編碼器中的一座塔（M30）
    """Section 4M M30: one input block's own encoder, Linear -> ReLU -> Linear -> LayerNorm (each block is scaled on its own before mixing)."""

    def __init__(self, n_in, h):  # n_in 欄輸入、寬度 h
        super().__init__()  # 初始化 nn.Module
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, h), torch.nn.ReLU(), torch.nn.Linear(h, h))  # 兩層線性夾一個 ReLU
        self.norm = torch.nn.LayerNorm(h)  # 每座塔的輸出各自正規化

    def forward(self, x):  # 前向
        return self.norm(self.net(x))  # 編碼後正規化


class Model(torch.nn.Module):  # R0、C、T、S 四個臂共用的網路；基礎集成的 24 個成員是其中的 C、T 臂
    def __init__(self, arm, n_r, dropout=0.1, pool="meanstd", token_enc="linear"):  # 依臂與選項建立網路
        super().__init__()  # 初始化 nn.Module
        if pool not in POOLINGS:  # 檢查池化方式
            raise ValueError(f"pooling {pool!r}, expected one of {POOLINGS}")  # 不支援就停止
        if token_enc not in TOKEN_ENCODERS:  # 檢查 token 編碼器
            raise ValueError(f"token encoder {token_enc!r}, expected one of {TOKEN_ENCODERS}")  # 不支援就停止
        self.arm, self.pool, self.token_enc = arm, pool, ("linear" if arm == "R0" else token_enc)  # R0 不用 token，編碼器固定記為 linear
        self.aux_loss = None  # M30 clip：上一次訓練前向的 InfoNCE（其他情況為 None）
        self.q_embed = None  # M33：上一次訓練前向的題目嵌入
        self.game_vecs = self.game_owner = None  # M134：上一次訓練前向的每局向量及其所屬題目（局層輔助損失用）
        if arm != "R0":  # 有 token 的臂
            d = HIDDEN[arm]  # 這個臂的隱藏寬度
            kernel = 3 if arm == "T" else 1  # T 用 kernel 3，C 用 kernel 1（逐手）
            if self.token_enc == "linear":  # 預設：一個線性投影
                self.proj = torch.nn.Linear(D_TOKEN, d)  # 每手 token -> d
            else:  # M30：盤面塔與分數塔，再把它們的輸出與 4 個旗標投影到 d
                hb, hs = DUAL_H  # 兩座塔的寬度
                self.board, self.score = Tower(128, hb), Tower(N_SCORES, hs)  # 盤面塔輸入主幹的 128 欄，分數塔輸入 30 個分數欄
                self.proj = torch.nn.Linear(hb + hs + 4, d)  # 兩座塔的輸出加 4 個旗標 -> d
                if self.token_enc == "clip":  # CLIP 式對齊頭（只用於訓練損失）與可學習的溫度
                    self.clip_board, self.clip_score = torch.nn.Linear(hb, CLIP_DIM), torch.nn.Linear(hs, CLIP_DIM)  # 兩座塔各自的對齊投影
                    self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07)))  # 可學習的溫度：存的是 log(1 / 溫度)，起始溫度 0.07（同 CLIP）
            if arm == "S":  # section 4M M21：對一局各手的全自注意力，加上學習的手位置嵌入
                self.pos = torch.nn.Embedding(S_MAX_PLIES, d)  # 手位置嵌入
                self.blocks = torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(d, S_HEADS, dim_feedforward=4 * d, dropout=dropout, batch_first=True, norm_first=True)  # pre-norm 的 transformer 編碼層
                                                   for _ in range(S_LAYERS)])  # 共 S_LAYERS 層
                self.final_norm = torch.nn.LayerNorm(d)  # 最後的正規化
            else:  # C 或 T 臂
                self.blocks = torch.nn.ModuleList([Block(d, kernel, 2 ** k if arm == "T" else 1, dropout) for k in range(5)])  # 五個殘差區塊；T 的擴張率為 1、2、4、8、16
            self.game = torch.nn.Linear(4 * d, GAME_DIM)  # 本人手與對手手各自的 mean/std（4d）-> 每局向量
            self.question = torch.nn.Linear(2 * GAME_DIM, Q_DIM)  # 各局向量的 mean/std -> 每題向量
            self.drop = torch.nn.Dropout(dropout)  # 題目向量的 dropout
            if pool == "pma":  # section 4M M4：4 個種子對一局的各手（-> 4d，與兩組 mean/std 同形），2 個種子對一題的各局（-> 2 * GAME_DIM）
                self.game_pool = PMA(d, 4, 2, dropout)  # 局內：4 個種子、2 個注意力頭
                self.q_pool = PMA(GAME_DIM, 2, 2, dropout)  # 題內：2 個種子、2 個注意力頭
            if pool == "sumll":  # section 4M M134：每局給出一個各等級的對數概似向量；一題加上 a * n^b * (它們的平均)：b = 1 是獨立證據相加，b = 0 是平均
                self.game_ll = torch.nn.Linear(GAME_DIM, 10)  # 每局向量 -> 10 級對數概似
                self.ll_log_a = torch.nn.Parameter(torch.zeros(()))  # log a，a 從 1 開始
                self.ll_b = torch.nn.Parameter(torch.zeros(()))  # b = sigmoid(ll_b)，起始為 0.5
        self.mlp = torch.nn.Sequential(torch.nn.Linear(n_r + (0 if arm == "R0" else Q_DIM), 32), torch.nn.ReLU(), torch.nn.Linear(32, 10))  # R 特徵（加上題目向量）-> 隱藏 32 -> 10 級 logits

    def embed(self, tokens):  # 每手的輸入投影
        """The per-ply input projection [G, L, d] and, for the two-tower encoders, the tower outputs (board, score)."""
        if self.token_enc == "linear":  # 單一線性投影
            return self.proj(tokens), None  # 沒有塔的輸出
        b, s = self.board(tokens[..., :128]), self.score(tokens[..., 128:128 + N_SCORES])  # 盤面塔與分數塔各自編碼
        return self.proj(torch.cat([b, s, tokens[..., D_TOKEN - 4:]], dim=-1)), (b, s)  # 兩塔輸出接上 4 個旗標後投影，並回傳兩塔輸出

    def clip_loss(self, views, mask):  # 兩座塔之間的 CLIP 式對比損失
        """M30: symmetric InfoNCE between the board and score views of up to CLIP_PLIES real plies (the same ply is the positive pair)."""
        b, s = views  # 盤面塔與分數塔的輸出
        idx = torch.nonzero(mask[..., 0].reshape(-1) > 0).squeeze(1)  # 真實手（非填充）的攤平索引
        if idx.numel() < 2:  # 少於兩手無法對比
            return None  # 這一批不加損失
        if idx.numel() > CLIP_PLIES:  # 手數太多
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:CLIP_PLIES]]  # 隨機抽 CLIP_PLIES 手
        zb = torch.nn.functional.normalize(self.clip_board(b.reshape(-1, b.shape[-1])[idx]), dim=1)  # 盤面視角：投影後 L2 正規化
        zs = torch.nn.functional.normalize(self.clip_score(s.reshape(-1, s.shape[-1])[idx]), dim=1)  # 分數視角：投影後 L2 正規化
        logits = self.logit_scale.clamp(max=math.log(100.0)).exp() * zb @ zs.T  # 相似度乘上可學的尺度（上限 100）
        target = torch.arange(idx.numel(), device=logits.device)  # 同一手在對角線上
        return 0.5 * (torch.nn.functional.cross_entropy(logits, target) + torch.nn.functional.cross_entropy(logits.T, target))  # 兩個方向的交叉熵平均（對稱）

    def encode_games(self, tokens, mask):  # 把每局的各手編碼成每局向量
        h, views = self.embed(tokens)  # 每手的投影（雙塔時另有兩塔輸出）
        h = h * mask  # 填充的位置歸零
        self.aux_loss = self.clip_loss(views, mask) if (self.token_enc == "clip" and self.training) else None  # 只在訓練時計算 CLIP 輔助損失
        if self.arm == "S":  # transformer 臂
            L = tokens.shape[1]  # 這一批的最大手數
            h = h + self.pos(torch.arange(L, device=tokens.device).clamp(max=S_MAX_PLIES - 1))[None]  # 加上手位置嵌入（超過上限的手共用最後一個位置）
            pad = ~mask[:, :, 0].bool()  # 注意力要忽略的填充位置
            for b in self.blocks:  # 逐層
                h = b(h, src_key_padding_mask=pad)  # 全注意力編碼
            h = self.final_norm(h) * mask  # 最後正規化並清掉填充
        else:  # C 或 T 臂
            for b in self.blocks:  # 逐個殘差區塊
                h = b(h, mask)  # 卷積區塊
        if self.pool == "pma":  # 注意力池化
            return torch.relu(self.game(self.game_pool(h, ~mask[:, :, 0].bool())))  # 4 個種子對各手做注意力，再轉成每局向量
        target = tokens[:, :, D_TOKEN - 4:D_TOKEN - 3] * mask  # 本人手的權重（is_target 旗標，未標準化）
        other = (1.0 - tokens[:, :, D_TOKEN - 4:D_TOKEN - 3]) * mask  # 對手手的權重
        tm, ts = masked_mean_std(h, target, 1)  # 本人手的平均與標準差
        om, os_ = masked_mean_std(h, other, 1)  # 對手手的平均與標準差
        return torch.relu(self.game(torch.cat([tm, ts, om, os_], dim=1)))  # 接起來轉成每局向量 [G, GAME_DIM]

    def pool_games(self, v, game_of, B):  # 把各局向量池化成每題的表示
        """[B, 2 * GAME_DIM]: the mean and std of each question's game vectors, or (pma) two seeds' attention over them."""
        owner = game_of[:, 0]  # 每局屬於批次中的哪一題
        if self.pool == "pma":  # 注意力池化
            counts = torch.bincount(owner, minlength=B)  # 每題的局數
            first = torch.cumsum(counts, 0) - counts  # 每題第一局的位置
            pos = torch.arange(owner.shape[0], device=v.device) - first[owner]  # 每局在題內的序號；一題的各局在 game_of 中是連續的
            padded = torch.zeros(B, int(counts.max()), v.shape[1], device=v.device, dtype=v.dtype)  # [B, 最多局數, GAME_DIM] 的填充張量
            padded[owner, pos] = v  # 放入各局向量
            pad = torch.arange(padded.shape[1], device=v.device)[None, :] >= counts[:, None]  # 超出該題局數的位置是填充
            return self.q_pool(padded, pad)  # 2 個種子對各局做注意力
        w = torch.zeros(B, owner.shape[0], device=v.device)  # 題目 × 局 的歸屬矩陣
        w[owner, torch.arange(owner.shape[0], device=v.device)] = 1.0  # 每局在所屬題目的位置為 1
        n = w.sum(dim=1, keepdim=True).clamp(min=1.0)  # 每題的局數
        m = (w @ v) / n  # 每題各局向量的平均
        s = ((w @ (v ** 2)) / n - m ** 2).clamp(min=0.0).add(1e-6).sqrt()  # 每題各局向量的標準差
        return torch.cat([m, s], dim=1)  # 平均與標準差接成 [B, 2 * GAME_DIM]

    def forward(self, r, tokens=None, mask=None, game_of=None):  # 前向：R 特徵加上（可選的）token
        if self.arm == "R0":  # 只用 R 特徵
            return self.mlp(r)  # 直接進 MLP
        v = self.encode_games(tokens, mask)  # 每局向量 [G, 64]
        self.game_vecs, self.game_owner = (v, game_of[:, 0]) if self.training else (None, None)  # section 4M M134：供局層輔助損失使用
        zq = torch.relu(self.question(self.pool_games(v if self.pool != "sumll" else v, game_of, r.shape[0])))  # 各局向量池化後轉成每題向量
        self.q_embed = zq if self.training else None  # section 4M M33：dropout 之前的題目嵌入，供玩家對比損失使用
        z = self.drop(zq)  # 題目向量的 dropout
        out = self.mlp(torch.cat([r, z], dim=1))  # R 特徵接上題目向量後進 MLP，得到 10 級 logits
        if self.pool == "sumll":  # M134 的對數概似相加通道
            owner = game_of[:, 0]; ll = self.game_ll(v)  # 每局所屬的題目；每局的 10 級對數概似
            n = torch.zeros(r.shape[0], device=v.device).index_add_(0, owner, torch.ones(owner.shape[0], device=v.device)).clamp(min=1.0)  # 每題的局數
            mean_ll = torch.zeros(r.shape[0], 10, device=v.device, dtype=ll.dtype).index_add_(0, owner, ll) / n[:, None]  # 每題各局對數概似的平均
            out = out + torch.exp(self.ll_log_a) * n[:, None] ** torch.sigmoid(self.ll_b) * mean_ll  # logits 加上 a × n^b × 平均
        return out  # 回傳 10 級 logits


def swa_average(eval_states, best_step, k):  # 以最佳評估點為中心平均 k 份權重
    """Uniform average of the k evaluation-time state dicts centred on the one recorded at best_step (clamped to the ends)."""
    steps = [s for s, _ in eval_states]  # 各評估點的步數
    i = steps.index(best_step) if best_step in steps else len(steps) - 1  # 最佳點的位置（找不到時用最後一個）
    lo = max(0, i - k // 2)  # 視窗起點
    window = eval_states[lo:lo + k] if lo + k <= len(eval_states) else eval_states[max(0, len(eval_states) - k):]  # 取 k 個狀態，超出尾端時往前靠
    avg = {}  # 平均後的狀態
    for key in window[0][1]:  # 逐個參數
        avg[key] = sum(st[key].float() for _, st in window) / len(window)  # 以 float 計算平均
        avg[key] = avg[key].to(window[0][1][key].dtype)  # 轉回原本的型別
    return avg  # 回傳平均後的 state dict


def batch_tokens(X, offsets, questions, device):  # 組出一批題目的填充 token
    """Padded tokens of the games of a batch of questions; game_of[:, 0] is each game's question (a game may repeat across questions)."""
    keys, owner = [], []  # 各局的鍵與所屬題目
    for i, ks in enumerate(questions):  # 逐題
        keys += list(ks)  # 這一題的各局
        owner += [i] * len(ks)  # 都屬於第 i 題
    starts = torch.tensor([offsets[k][0] for k in keys], device=device)  # 各局在 X 中的起點
    lens = torch.tensor([offsets[k][1] for k in keys], device=device)  # 各局的手數
    L = int(lens.max())  # 這一批的最大手數
    idx = starts[:, None] + torch.arange(L, device=device)[None]  # 每局每個位置在 X 中的列
    mask = (torch.arange(L, device=device)[None] < lens[:, None])  # 真實的手為 True
    tokens = X[idx.clamp(max=X.shape[0] - 1)].float() * mask[:, :, None]  # 取出 token（索引截在範圍內），填充處歸零
    return tokens, mask[:, :, None].float(), torch.tensor(owner, device=device)[:, None]  # token [G, L, D]、遮罩 [G, L, 1]、所屬題目 [G, 1]


def predict(model, R, questions, X, offsets, device, batch=64):  # 批次預測各題的 10 級機率
    model.eval()  # 評估模式（關閉 dropout）
    out = []  # 收集各批的機率
    with torch.no_grad():  # 不需要梯度
        for i in range(0, len(R), batch):  # 每 batch 題一批
            r = torch.as_tensor(R[i:i + batch], device=device)  # 這一批的 R 特徵
            if model.arm == "R0":  # 只用 R
                logits = model(r)  # 前向
            else:  # 有 token 的臂
                tokens, mask, game_of = batch_tokens(X, offsets, questions[i:i + batch], device)  # 這一批的 token
                logits = model(r, tokens, mask, game_of)  # 前向
            out.append(torch.softmax(logits, dim=1).double().cpu().numpy())  # 轉成機率（float64）
    return np.concatenate(out)  # 接成 [Q, 10] 的機率陣列


def mixed_questions(train_q, data, train_folds, v, mix, inner_frac):  # 把部分訓練題換成同級混人題
    """Section 4L.4: a seeded share `mix` of the training questions become same-rank mixed-player questions -- n distinct other players
    of the same rank (training folds only, never the inner held-out players train() will hold out), one random game each. player_id,
    rank and n_games stay the originating player's (train() still drops held-out players' questions and never scores a mixed question
    on the inner set); game_keys change and a `mixed` column marks the rows. A question whose rank has too few other players is left as is."""
    rng = np.random.RandomState(4242 + fold_code(v))  # 混人題的抽樣種子
    players = np.array(sorted(set(train_q["player_id"].to_numpy(dtype=str))))  # 訓練題的玩家
    held = set(np.random.RandomState(12345 + fold_code(v)).permutation(players)[:int(inner_frac * len(players))])  # 與 train() 相同的留出規則
    pool = {}  # 等級 -> [(玩家, 棋局清單)]
    for f in train_folds:  # 只從訓練折抽
        for r in data[f]["players"].itertuples():  # 逐位玩家
            if str(r.player_id) not in held:  # 排除 inner 留出玩家
                pool.setdefault(r.rank, []).append((str(r.player_id), list(r.game_keys)))  # 依等級收集
    q = train_q.reset_index(drop=True).copy()  # 複製題目表
    keys_col = [list(k) for k in q["game_keys"].tolist()]  # 各題的棋局清單（可改寫）
    mixed = np.zeros(len(q), dtype=bool)  # 哪些題目改成了混人題
    pick = np.flatnonzero(rng.rand(len(q)) < mix)  # 以比例 mix 抽出要改的題目
    pids, ranks = q["player_id"].to_numpy(dtype=str), q["rank"].to_numpy(dtype=str)  # 各題的玩家與等級
    for i in pick:  # 逐一改寫
        if pids[i] in held:  # 留出玩家的題目維持原樣：它們是 inner 集合，絕不當訓練資料
            continue  # 不改
        cands = [c for c in pool.get(ranks[i], []) if c[0] != pids[i]]  # 同等級的其他玩家
        n = len(keys_col[i])  # 這一題的局數
        if len(cands) < n:  # 其他玩家不夠
            continue  # 維持原樣
        chosen = rng.choice(len(cands), n, replace=False)  # 抽 n 位不同的玩家
        keys_col[i] = [cands[j][1][rng.randint(len(cands[j][1]))] for j in chosen]  # 每人隨機取一局
        mixed[i] = True  # 標記為混人題
    q["game_keys"] = keys_col  # 寫回棋局清單
    q["mixed"] = mixed  # 加上標記欄
    return q  # 回傳新的題目表


def extra_tables(f, base, folds_df, hd_all, dirs=None, level=40):  # 把額外棋局併入第 f 折的訓練表
    """Section 4M M8 / x80: fold f's training tables with the players' games 21-40 (and 41-80 at level 80) appended to the 20-game
    sample; the players' game_keys come back in fold_players' order over up to `level` games. ValueError when a game is scored
    twice or an extra directory is incomplete (players_from_scores checks every player's count against the selection)."""
    dirs = dirs or L1B_SETS[3]  # 預設第 3 組 L1b
    hsl, l1b, opp, ohsl = base["hsl"], base["l1b"], base["opp"], base.get("opp_hsl")  # 20 局樣本的各張表
    for k in extra_sets(level):  # 逐一加入各級額外棋局
        l1b_map, opp_map = set_dirs_for(dirs, k)  # 該級的 L1b 目錄
        hsl_x, l1b_x = l1b_cv.load_fold(EXTRAS[k]["hsl"], ROOT / "results/l1b" / l1b_map[f], f)  # 額外棋局的 Human SL 與本人側 L1b
        opp_x = read_scored_games(ROOT / "results/l1b" / opp_map[f] / f"fold{f}")  # 額外棋局的對手側 L1b
        hsl, l1b, opp = (pd.concat([t, x], ignore_index=True) for t, x in ((hsl, hsl_x), (l1b, l1b_x), (opp, opp_x)))  # 接到原表後面
        if ohsl is not None:  # 有對手 HSL 表時
            ohsl = pd.concat([ohsl, opp_hsl_table(opp_hsl_dir(k), f, hsl_x)], ignore_index=True)  # 也接上額外棋局的對手 HSL
    for name, t in (("hsl", hsl), ("l1b", l1b), ("opp", opp)):  # 檢查各表沒有重複的局
        if not t["game_key"].is_unique:  # 同一局在基礎與額外目錄都有
            raise ValueError(f"fold {f} {name}: {int(t['game_key'].duplicated().sum())} games appear in both the base and the extra directory")  # 停止
    if set(hsl["game_key"]) != set(opp["game_key"]):  # 本人側與對手側必須是同一批棋局
        raise ValueError(f"fold {f}: the extra target and opponent directories score different games")  # 停止
    players = l1b_cv.players_from_scores(hsl, folds_df, max_games=level)  # 依 fold_players 的順序還原每位玩家至多 level 局
    return {"hsl": hsl, "l1b": l1b, "opp": opp, "hd": hd_all[hd_all["game_key"].isin(set(hsl["game_key"]))], "players": players,  # 合併後的各張表、讓子資料與玩家表
            **({"opp_hsl": ohsl} if ohsl is not None else {})}  # 有對手 HSL 時一併回傳


def r_features(a, v, pool=0):  # 驗證折 v 的 64 個 R 特徵（有快取）
    """The 64 R features of the training and validation questions of validation fold v, cached, in the protocol of scripts/hd_check.py.
    v = "all" (the submission's models): the training questions of all four folds and no validation questions (empty arrays).
    a.mix > 0 (section 4L.4): that share of the training questions are same-rank mixed-player questions (mixed_questions), tag _mix<pct>."""
    mix = float(getattr(a, "mix", 0.0) or 0.0)  # 混人題比例
    games, sizes = int(getattr(a, "train_games", 0) or 0), getattr(a, "train_sizes", None)  # 每人局數上限與題目大小範圍
    extra = extra_level(a)  # 額外棋局等級
    train_folds = training_folds(v, int(getattr(a, "train_folds", 0) or 0), bool(getattr(a, "with_fold0", False)))  # 參與訓練的折
    tag = ("r_all" if v == "all" else f"r_val{v}") + (f"_p{pool}" if pool else "") + "".join("_" + x for x in data_segments(a) if x not in ("oh", "ai"))  # R 快取的 tag；pool k > 0：每位玩家 32 題訓練題的另一次抽樣；-oh 只改變 token
    path = Path(a.cache) / f"{tag}.npz"  # 特徵快取
    meta_path = Path(a.cache) / f"{tag}.json"  # 題目表快取
    if path.exists() and meta_path.exists():  # 快取已存在
        z = np.load(path)  # 讀特徵
        m = json.loads(meta_path.read_text())  # 讀題目表
        tr = pd.DataFrame(m["train"])  # 訓練題
        va = pd.DataFrame(m["val"])  # 驗證題
        return z["X_tr"], z["y_tr"], z["X_va"], z["y_va"], tr, va  # 直接回傳
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})  # 玩家分折表
    hd_all = read_hd_table(ROOT / "results/data/hd_table.csv")  # 全部棋局的讓子表
    data = {}  # 各折的資料表
    dirs = L1B_SETS[l1b_set_of(a)]  # 使用的 L1b 組
    for f in sorted(set(train_folds) | ({int(v)} if v != "all" else set())):  # 訓練折加上驗證折
        hsl, l1b = l1b_cv.load_fold(HSL, ROOT / "results/l1b" / dirs["l1b"][f], f)  # 20 局樣本的 Human SL 與本人側 L1b
        opp = read_scored_games(ROOT / "results/l1b" / dirs["opp"][f] / f"fold{f}")  # 對手側 L1b
        hd = hd_all[hd_all["game_key"].isin(set(hsl["game_key"]))]  # 這些棋局的讓子資料
        players = l1b_cv.players_from_scores(hsl, folds_df)  # 玩家與其棋局清單
        data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "hd": hd, "players": players, "val": l1b_cv.pilot_questions(players)}  # 各表與固定的驗證題
        if opp_hsl_level(a) == 2:  # section 4M M14：對手著手在 18 個剖面下的評分，棋局與 HSL 表完全相同
            data[f]["opp_hsl"] = opp_hsl_table(OPP_HSL, f, hsl)  # 讀對手 HSL 表
        data[f]["train"] = extra_tables(f, data[f], folds_df, hd_all, dirs, extra) if (extra and f in train_folds) else {k: data[f][k] for k in ("hsl", "l1b", "opp", "hd", "players", "opp_hsl") if k in data[f]}  # 訓練用的表：訓練折需要時併入額外棋局，否則沿用 20 局樣本
    rng = np.random.RandomState(l1b_cv.TRAIN_SEED + fold_code(v) + 1000 * pool)  # 抽訓練題的種子（每個 pool 一次不同的抽樣）
    size_range = range(int(sizes[0]), int(sizes[1]) + 1) if sizes else range(5, 21)  # 題目大小範圍（預設 5-20 局）
    train_q = pd.concat([sample_questions(cap_games(data[f]["train"]["players"], games), 32, rng, sizes=size_range) for f in train_folds], ignore_index=True)  # 每位訓練玩家抽 32 題
    if extra:  # 用了額外棋局
        print(f"extra games: training players draw from up to {extra} games ("  # 回報每個訓練折可抽的局數
              + ", ".join(f"fold {f}: {sum(len(k) for k in data[f]['train']['players']['game_keys'])} games" for f in train_folds) + ")", flush=True)  # 各折的總局數
    if mix > 0:  # 要做混人題
        train_q = mixed_questions(train_q, data, train_folds, v, mix, float(getattr(a, "inner_frac", 0.1)))  # 改寫部分訓練題
        print(f"mixed-player questions: {int(train_q['mixed'].sum())} of {len(train_q)} (share {mix})", flush=True)  # 回報混人題數
    cat = lambda key: pd.concat([data[f]["train"][key] for f in train_folds], ignore_index=True)  # noqa: E731  把各訓練折的同一張表接起來
    X_tr, y_tr, _, _ = hd_check.features(cat("hsl"), cat("l1b"), cat("opp"), cat("hd"), train_q)  # 訓練題的特徵與標籤
    Xt = X_tr["fused_opp_diff_hd"]  # 取 64 維的特徵組
    if opp_hsl_level(a) == 2:  # section 4M M14：再加 39 個對手 HSL 題目特徵（64 -> 103）
        Xt = np.hstack([Xt, opp_hsl_features(cat("hsl"), cat("opp_hsl"), train_q)])  # 接上對手 HSL 題目特徵
    cols = ["player_id", "rank", "n_games", "game_keys"]  # 題目表要保存的欄位
    tcols = cols + (["mixed"] if "mixed" in train_q.columns else [])  # 有混人題時多存 mixed 欄
    if v == "all":  # 提交用模型沒有驗證題
        Xv, y_va, val_q = np.zeros((0, Xt.shape[1]), dtype=Xt.dtype), np.zeros(0, dtype=y_tr.dtype), pd.DataFrame(columns=cols)  # 空陣列與空表
    else:  # 有驗證折
        d = data[v]  # 驗證折的資料
        X_va, y_va, _, _ = hd_check.features(d["hsl"], d["l1b"], d["opp"], d["hd"], d["val"])  # 驗證題的特徵與標籤
        Xv, val_q = X_va["fused_opp_diff_hd"], d["val"]  # 64 維特徵與驗證題表
        if opp_hsl_level(a) == 2:  # -ohr 時驗證題也要對手 HSL 題目特徵
            Xv = np.hstack([Xv, opp_hsl_features(d["hsl"], d["opp_hsl"], d["val"])])  # 驗證題也接上對手 HSL 題目特徵
    Path(a.cache).mkdir(parents=True, exist_ok=True)  # 確保快取目錄存在
    np.savez(path, X_tr=Xt, y_tr=y_tr, X_va=Xv, y_va=y_va)  # 寫出特徵快取
    meta_path.write_text(json.dumps({"train": train_q[tcols].to_dict("list"), "val": val_q[cols].to_dict("list")}))  # 寫出題目表快取
    return Xt, y_tr, Xv, y_va, train_q[tcols], val_q[cols]  # 回傳特徵、標籤與兩張題目表


def train(a):  # train 子命令：訓練並評估一個模型（一個臂、一個驗證折或 all、一個種子）
    sd = str(getattr(a, "save_dir", "") or "")  # 權重輸出目錄（可能沒給）
    if sd and (bool(HSL_TAG) != (f"_{HSL_TAG}" in sd if HSL_TAG else "_jp" in sd)):  # section 4M M27：jp 模型只能存進 *_jp，絕不存進 ct7 的目錄
        raise SystemExit(f"--save-dir {sd} does not match AICUP_HSL={HSL_TAG!r}")  # 目錄與 Human SL 設定不符就停止
    if a.val_fold == 0:  # 第 0 折不當驗證折
        raise SystemExit("fold 0 is not used here")  # 停止
    is_all = a.val_fold == "all"  # 提交用的模型：每一折的玩家都訓練，只靠 inner 留出決定停止點
    train_folds = training_folds(a.val_fold, a.train_folds, a.with_fold0)  # 參與訓練的折
    device = "cuda"  # 只在 GPU 上訓練
    torch.manual_seed(a.seed)  # 固定 torch 的種子
    np.random.seed(a.seed)  # 固定 numpy 的全域種子
    t0 = time.time()  # 計時起點
    X_tr, y_tr, X_va, y_va, q_tr, q_va = r_features(a, a.val_fold)  # 第 0 次抽樣的 R 特徵與題目表
    if a.r_extra:  # section 4M M36：加上交叉擬合的注意力頭的 20 個題目欄
        from aicup.attnfeat import load_scores, question_features  # 注意力頭的逐局分數與題目欄
        sc = load_scores(a.r_extra, a.r_extra_games)  # 讀逐局分數
        X_tr = np.hstack([X_tr, question_features(q_tr["game_keys"], sc)]).astype(X_tr.dtype)  # 訓練題接上 20 欄
        X_va = np.hstack([X_va, question_features(q_va["game_keys"], sc)]).astype(X_va.dtype) if len(q_va) else np.zeros((0, X_tr.shape[1]), X_tr.dtype)  # 驗證題也接上（沒有驗證題時給空陣列）
        print(f"r-extra: {len(sc)} scored games, R {X_tr.shape[1]} columns", flush=True)  # 回報欄數
    if a.r_extra_npz:  # section 4M M40：預先算好、依 R 快取題目順序的逐題欄位
        zx = np.load(a.r_extra_npz.format(v=a.val_fold))  # 這個驗證折的欄位檔
        if zx["X_tr"].shape[0] != X_tr.shape[0] or zx["X_va"].shape[0] != X_va.shape[0]:  # 列數必須與 R 相同
            raise SystemExit(f"--r-extra-npz rows {zx['X_tr'].shape[0]} / {zx['X_va'].shape[0]} do not match R {X_tr.shape[0]} / {X_va.shape[0]}")  # 不同就停止
        X_tr = np.hstack([X_tr, zx["X_tr"]]).astype(X_tr.dtype); X_va = np.hstack([X_va, zx["X_va"]]).astype(X_va.dtype)  # 訓練題與驗證題各自接上
        print(f"r-extra-npz: R {X_tr.shape[1]} columns", flush=True)  # 回報欄數
    y_all = y_tr.copy()  # 保留完整的訓練標籤（之後會切掉留出玩家）
    held = set()  # inner 留出的玩家
    mean, std = X_tr.mean(axis=0), X_tr.std(axis=0)  # R 的標準化統計（第 0 次抽樣的訓練題）
    std[std == 0] = 1.0  # 常數欄不縮放
    R_tr, R_va = ((X_tr - mean) / std).astype(np.float32), ((X_va - mean) / std).astype(np.float32)  # 標準化後的 R
    keys_tr, keys_va = q_tr["game_keys"].tolist(), q_va["game_keys"].tolist()  # 各題的棋局清單
    R_in, y_in, keys_in, q_in = None, None, None, None  # inner 評估集（沒有留出時為 None）
    if a.inner_frac > 0:  # 這個驗證折的每個臂、每個種子都用同一批留出玩家
        pid_tr = q_tr["player_id"].to_numpy(dtype=str)  # 訓練題的玩家
        players = np.array(sorted(set(pid_tr)))  # 訓練玩家（排序後與列的順序無關）
        held = set(np.random.RandomState(12345 + fold_code(a.val_fold)).permutation(players)[:int(a.inner_frac * len(players))])  # 只依驗證折決定的留出玩家
        is_in = np.array([x in held for x in pid_tr])  # 哪些題目屬於留出玩家
        mixed = q_tr["mixed"].to_numpy(dtype=bool) if "mixed" in q_tr.columns else np.zeros(len(q_tr), dtype=bool)  # 混人題含有其他訓練玩家的棋局，不放進 inner 評估集
        pick = np.random.RandomState(fold_code(a.val_fold)).permutation(np.flatnonzero(is_in & ~mixed))[:a.inner_questions]  # 從留出玩家的非混人題中抽 inner_questions 題
        R_in, y_in, keys_in, q_in = R_tr[pick], y_tr[pick], [keys_tr[i] for i in pick], q_tr.iloc[pick]  # inner 評估集
        keep = np.flatnonzero(~is_in)  # 非留出玩家的題目
        R_tr, y_tr, keys_tr = R_tr[keep], y_tr[keep], [keys_tr[i] for i in keep]  # 訓練集去掉留出玩家
        print(f"inner: {len(held)} of {len(players)} training players held out, {len(pick)} questions score the stopping point", flush=True)  # 回報留出人數與題數
    bagged = None  # bagging 選到的玩家（None 表示全部）
    if a.bag < 1:  # 依玩家 bagging：這個模型只看到以種子固定比例抽出的訓練玩家
        cand = np.array(sorted(set(q_tr["player_id"].astype(str)) - held))  # 非留出的訓練玩家
        bagged = set(np.random.RandomState(777 + a.seed + 100 * fold_code(a.val_fold)).permutation(cand)[:int(a.bag * len(cand))])  # 依種子與驗證折抽出比例 bag 的玩家

    pool_pids = []  # section 4M M33：每次抽樣中每一題的玩家，依該次抽樣的順序

    def prep(Xp, yp, qp):  # 整理一次抽樣的訓練題
        """One draw of training questions without the held-out players (and, with --bag, only the bagged ones), standardised as pool 0;
        weights are class-balanced, times each question's mean game uniqueness (1 / the number of questions using the game) with --uniqueness."""
        pid = qp["player_id"].to_numpy(dtype=str)  # 各題的玩家
        ok = np.array([(x not in held) and (bagged is None or x in bagged) for x in pid])  # 排除留出玩家；bagging 時只留選到的玩家
        Rp = ((Xp[ok] - mean) / std).astype(np.float32)  # 用第 0 次抽樣的統計標準化
        kp = [k for k, o in zip(qp["game_keys"].tolist(), ok) if o]  # 保留題目的棋局清單
        pool_pids.append(pid[ok])  # 記下各題的玩家（配對批次要用）
        w = class_weights(yp[ok])  # 類別平衡權重
        if a.uniqueness:  # 依棋局獨特度加權
            count = {}  # 每局被幾題用到
            for ks in kp:  # 逐題
                for g in ks:  # 逐局
                    count[g] = count.get(g, 0) + 1  # 累計使用次數
            u = np.array([np.mean([1.0 / count[g] for g in ks]) for ks in kp])  # 每題各局獨特度的平均
            w = w * u / u.mean()  # 乘上正規化後的獨特度
            w = w / w.mean()  # 讓權重平均為 1
        return Rp, yp[ok], kp, w  # 回傳 R、標籤、棋局清單與權重

    pools = [prep(X_tr, y_all, q_tr)]  # 第 0 次抽樣
    for k in range(1, a.resample):  # --resample K：第 e 個 epoch 用第 e mod K 次抽樣訓練
        Xk, yk, _, _, qk, _ = r_features(a, a.val_fold, k)  # 第 k 次抽樣的 R 特徵
        pools.append(prep(Xk, yk, qk))  # 同樣整理
    soft0 = None  # 第 0 次抽樣的軟目標（只有 --pseudo 時才有）
    if a.pseudo:  # CV 轉導：驗證題以軟目標（不用標籤）加入第一次抽樣，權重為 --pseudo-weight
        if is_all:  # 提交用模型沒有驗證題
            raise SystemExit("--pseudo is the CV simulation (validation questions as unlabelled data); not for --val-fold all")  # 停止
        pz = np.load(a.pseudo)  # 讀驗證題的機率檔
        Pp = np.asarray(pz["proba"], dtype=np.float32)  # 驗證題的機率 [Q_val, 10]
        if Pp.shape != (len(R_va), 10):  # 必須與驗證題對齊
            raise SystemExit(f"--pseudo proba {Pp.shape} does not match the {len(R_va)} validation questions")  # 停止
        Rp, yp, kp, wp = pools[0]  # 第 0 次抽樣
        soft0 = np.vstack([np.eye(10, dtype=np.float32)[yp], Pp])  # 真實題用 one-hot，偽標籤題用機率
        pools[0] = (np.vstack([Rp, R_va]), np.concatenate([yp, Pp.argmax(axis=1)]), kp + list(keys_va), np.concatenate([wp, np.full(len(R_va), a.pseudo_weight, dtype=wp.dtype)]))  # 驗證題併入第 0 次抽樣，權重為 --pseudo-weight
        print(f"pseudo: {len(R_va)} validation questions added with soft targets (weight {a.pseudo_weight})", flush=True)  # 回報加入的題數
    if a.limit:  # 只用於冒煙測試
        keep = np.random.RandomState(0).permutation(len(pools[0][1]))[:a.limit]  # 只留 limit 題
        pools[0] = (pools[0][0][keep], pools[0][1][keep], [pools[0][2][i] for i in keep], pools[0][3][keep])  # 截短第 0 次抽樣
        pool_pids[0] = pool_pids[0][keep]  # 玩家清單同步截短
    R_tr, y_tr, keys_tr, _ = pools[0]  # 之後以第 0 次抽樣的大小計算步數
    print(f"training draws: {[len(x[1]) for x in pools]} questions" + (f"; {len(bagged)} bagged players" if bagged else ""), flush=True)  # 回報各次抽樣的題數
    X, offsets, Xv, offsets_v, stats = None, None, None, None, None  # token 相關的變數（R0 不用）
    if a.arm != "R0":  # 需要 token 的臂
        X, offsets, stats = load_tokens(train_folds, a.cache, device, drop=a.drop, vec_dir=a.vec_dir, extra=extra_level(a), l1b_set=l1b_set_of(a), opp_hsl=opp_hsl_level(a) > 0)  # 訓練折的 token，並算出標準化統計
        if not is_all:  # 有驗證折
            Xv, offsets_v, _ = load_tokens([a.val_fold], a.cache, device, stats, drop=a.drop, vec_dir=a.vec_dir, l1b_set=l1b_set_of(a), opp_hsl=opp_hsl_level(a) > 0)  # 驗證折的 token，用訓練端的統計
        print(f"tokens loaded: {X.shape[0]} + {0 if is_all else Xv.shape[0]} plies ({time.time() - t0:.0f} s)", flush=True)  # 回報手數與耗時
    if a.arm == "S":  # section 4M M21b：可調整的寬度與深度
        HIDDEN["S"] = int(a.s_dim); globals()["S_LAYERS"] = int(a.s_layers)  # 依參數改寫 S 臂的寬度與層數
    if a.width_mult != 1.0 and a.arm in ("C", "T"):  # section 4M M134：加寬區塊
        HIDDEN[a.arm] = int(round(HIDDEN[a.arm] * a.width_mult))  # 區塊寬度乘上倍數
    model = Model(a.arm, R_tr.shape[1], a.dropout, pool=a.pooling, token_enc=a.token_enc).to(device)  # 建立模型
    n_params = sum(p.numel() for p in model.parameters())  # 總參數量
    blocks = sum(p.numel() for p in model.blocks.parameters()) if a.arm != "R0" else 0  # 區塊的參數量（比較 C 與 T 用）
    pool_t = [(torch.as_tensor(Rp, device=device), torch.as_tensor(yp, device=device).long(), torch.as_tensor(wp, device=device, dtype=torch.float32), kp,  # 每次抽樣的 R、標籤、權重與棋局清單放上 GPU
               torch.as_tensor(kernel_targets(yp), device=device, dtype=torch.float32) if (soft0 is None or j > 0) else torch.as_tensor(soft0, device=device))  # 核目標；--pseudo 時第 0 次抽樣改用 soft0（真實題 one-hot、偽標籤題機率）
              for j, (Rp, yp, kp, wp) in enumerate(pools)]  # 逐次抽樣
    if a.pseudo and a.arm != "R0":  # 偽標籤題的 token 在驗證折的張量裡：合併兩份局表
        shift = X.shape[0]  # 驗證折 token 接在訓練 token 之後的位移
        offsets = {**offsets, **{k: (s_ + shift, n_) for k, (s_, n_) in offsets_v.items()}}  # 合併局表
        X = torch.cat([X, Xv])  # 合併 token
    ext = None  # 外部集合的訓練狀態（沒用時為 None）
    if a.ext:  # section 4M M135：外部有段位的單位經由第二個、只在訓練用的輸出頭訓練同一個網路
        Xe, ye, qe, ext_ranks = ext_questions(a.ext)  # 外部題的 R 特徵、標籤、題目表與段位
        if Xe.shape[1] != X_tr.shape[1]:  # 欄數必須與這次的 R 相同
            raise SystemExit(f"--ext: the external R features have {Xe.shape[1]} columns, this run's R has {X_tr.shape[1]}")  # 停止
        pid_e = qe["player_id"].to_numpy(dtype=str)  # 外部題的單位
        units = np.array(sorted(set(pid_e)))  # 全部外部單位
        held_e = set(np.random.RandomState(4321).permutation(units)[:max(1, int(0.05 * len(units)))])  # 只用於報告：外部輸出頭在沒見過的單位上的表現
        is_h = np.array([x in held_e for x in pid_e])  # 哪些外部題屬於留出單位
        tr_e, he_e = np.flatnonzero(~is_h), np.random.RandomState(0).permutation(np.flatnonzero(is_h))[:2000]  # 訓練用的外部題；報告用的留出題（至多 2,000 題）
        if a.ext_frac < 1.0:  # 劑量反應臂：以種子固定的比例抽出參與訓練的外部單位（留出單位不變）
            kept = set(np.random.RandomState(99).permutation(np.array(sorted(set(units) - held_e)))[:int(a.ext_frac * (len(units) - len(held_e)))])  # 抽中的非留出單位
            tr_e = tr_e[np.array([pid_e[i] in kept for i in tr_e])]  # 只留這些單位的題目
        if a.limit:  # 只用於冒煙測試
            tr_e = tr_e[:a.limit]  # 截短
        Re = ((Xe - mean) / std).astype(np.float32)  # 用比賽訓練資料的統計標準化
        keys_e = qe["game_keys"].tolist()  # 外部題的棋局清單
        if a.arm != "R0":  # 需要 token 的臂
            Xx, off_e = ext_tokens(a.ext, device, stats, drop=a.drop)  # 以比賽的 token 統計標準化
            lost = sorted({k for ks in keys_e for k in ks} - set(off_e))  # 外部題中沒有 token 的局
            if lost:  # 有缺
                raise SystemExit(f"--ext: {len(lost)} games of the external questions have no tokens, e.g. {lost[:3]}")  # 停止
            shift = X.shape[0]  # 外部 token 接在後面的位移
            offsets = {**offsets, **{k: (s_ + shift, n_) for k, (s_, n_) in off_e.items()}}  # 合併局表
            X = torch.cat([X, Xx])  # 合併 token
            del Xx  # 釋放外部 token 的暫存
        ext_head = torch.nn.Linear(32, len(ext_ranks)).to(device)  # 外部段位的輸出頭，接在 MLP 的 32 維隱藏層之後
        ext = {"R": torch.as_tensor(Re, device=device), "y": torch.as_tensor(ye, device=device).long(), "keys": keys_e, "tr": tr_e, "held": he_e, "head": ext_head, "ranks": ext_ranks,  # 外部題的資料、輸出頭與段位
               "w": torch.as_tensor(class_weights(ye[tr_e], len(ext_ranks)), device=device, dtype=torch.float32), "rng": np.random.RandomState(1000 + a.seed), "run": 0.0, "n": 0}  # 類別平衡權重、抽批次的亂數與累計損失
        ext["w_full"] = torch.zeros(len(ye), device=device); ext["w_full"][torch.as_tensor(tr_e, device=device)] = ext["w"]  # 依題號索引的權重表（非訓練題為 0）
        print(f"external set: {len(units)} units ({len(held_e)} held out for the report), {len(tr_e)} training questions, {len(ext_ranks)} ranks {ext_ranks[0]}..{ext_ranks[-1]}, weight {a.ext_weight:g}, "  # 回報外部集合的規模與設定
              f"batch {a.ext_batch}; tokens now {0 if X is None else X.shape[0]} plies ({time.time() - t0:.0f} s)", flush=True)  # 以及目前的 token 手數與耗時

    def ext_logits(idx):  # 外部題經同一個網路與外部輸出頭的 logits
        """The external head's logits of external questions idx (the same network up to the last hidden layer)."""
        it = torch.as_tensor(idx, device=device)  # 題號放上 GPU
        r = ext["R"][it]  # 這些外部題的 R
        if a.arm == "R0":  # 只用 R
            return ext["head"](model.mlp[1](model.mlp[0](r))), it  # MLP 的第一層與 ReLU 之後接外部輸出頭
        tokens, mask, game_of = batch_tokens(X, offsets, [ext["keys"][j] for j in idx], device)  # 外部題的 token
        v = model.encode_games(tokens, mask)  # 每局向量
        z = model.drop(torch.relu(model.question(model.pool_games(v, game_of, r.shape[0]))))  # 每題向量（與 forward 相同的路徑）
        return ext["head"](model.mlp[1](model.mlp[0](torch.cat([r, z], dim=1)))), it  # 經 MLP 隱藏層後接外部輸出頭

    pc_head = torch.nn.Linear(Q_DIM, 32).to(device) if a.player_contrast > 0 else None  # M33：只在訓練用的投影頭
    ga_head = torch.nn.Linear(GAME_DIM, 10).to(device) if (a.game_aux > 0 and a.arm != "R0") else None  # M134：局層輔助損失的頭（只在訓練用）
    train_params = list(model.parameters()) + (list(pc_head.parameters()) if pc_head is not None else []) + (list(ga_head.parameters()) if ga_head is not None else []) + (list(ext["head"].parameters()) if ext is not None else [])  # 要更新的參數：模型與各個只在訓練用的頭
    opt = torch.optim.AdamW(train_params, lr=a.lr, weight_decay=a.wd)  # AdamW 最佳化器
    steps = a.epochs * (len(R_tr) // a.batch)  # 總步數上限
    if a.warmup_steps > 0:  # section 4M M21b：先線性 warmup，再餘弦
        wu = int(a.warmup_steps)  # warmup 步數
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (t + 1) / wu if t < wu else 0.5 * (1 + math.cos(math.pi * min(1.0, (t - wu) / max(1, steps - wu)))))  # 先線性升到 lr，再以餘弦降到 0
    else:  # 沒有 warmup
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)  # 直接餘弦退火
    rng = np.random.RandomState(a.seed)  # 打亂題目順序的亂數
    history, step, eval_states = [], 0, []  # 紀錄、步數與 SWA 用的評估狀態
    run_aux, n_aux = 0.0, 0  # M30 clip：累計的 InfoNCE，供紀錄用
    every = max(1, (len(R_tr) // a.batch) // a.evals_per_epoch)  # 每隔幾步評一次 inner
    best, best_state, best_step, since = -1.0, None, 0, 0  # 最佳 inner 分數、狀態、步數與未進步的次數
    stop = False  # 提早停止的旗標

    def logits_of(R_t, keys_tr, b):  # 一批訓練題的 logits
        bt = torch.as_tensor(b, device=device)  # 批次題號
        if a.arm == "R0":  # 只用 R
            return model(R_t[bt]), bt  # 前向
        tokens, mask, game_of = batch_tokens(X, offsets, [keys_tr[j] for j in b], device)  # 這一批的 token
        if model.training and (a.ply_drop > 0 or a.trunk_mask > 0):  # section 4M M133：token 擴增，只用於訓練批次
            G, L = mask.shape[0], mask.shape[1]  # 局數與最大手數
            if a.ply_drop > 0:  # 每局隨機丟掉一部分手（比例在 0 .. ply_drop 間均勻抽）
                keep = (torch.rand(G, L, 1, device=device) >= torch.rand(G, 1, 1, device=device) * a.ply_drop).float()  # 每局先抽丟棄比例，再逐手決定保留
                mask, tokens = mask * keep, tokens * keep  # 丟掉的手當成填充
            if a.trunk_mask > 0:  # 每局隨機一部分手（比例在 0 .. trunk_mask 間均勻抽）失去主幹向量（已標準化：0 = 平均）
                hit = (torch.rand(G, L, 1, device=device) < torch.rand(G, 1, 1, device=device) * a.trunk_mask).float()  # 每局先抽比例，再逐手決定是否遮掉
                tokens = torch.cat([tokens[:, :, :128] * (1.0 - hit), tokens[:, :, 128:]], dim=2)  # 被遮的手主幹 128 欄設為 0，其他欄保留
        return model(R_t[bt], tokens, mask, game_of), bt  # 前向

    for epoch in range(a.epochs):  # 逐 epoch
        if stop:  # 已經提早停止
            break  # 結束訓練
        model.train()  # 訓練模式
        R_t, y_t, weight, keys_tr, targets = pool_t[epoch % len(pool_t)]  # 這個 epoch 用的抽樣
        paired = pc_head is not None or a.pair_batches  # M33（-pb：只做配對，當作診斷）
        if paired and a.pc_views > 2:  # M39：每組 pc_views 題同一玩家的題目
            perm = grouped_perm(pool_pids[epoch % len(pool_t)], rng, a.pc_views)  # 同一玩家成組的排列
        else:  # 兩兩配對或一般隨機
            perm = paired_perm(pool_pids[epoch % len(pool_t)], rng) if paired else rng.permutation(len(keys_tr))  # 配對排列或一般的隨機排列
        run, n_run, t_ep = 0.0, 0, time.time()  # 累計損失、步數與 epoch 計時
        for i in range(0, len(perm) - a.batch + 1, a.batch):  # 逐批（不足一批的尾巴丟掉）
            logits, bt = logits_of(R_t, keys_tr, perm[i:i + a.batch])  # 前向
            if a.loss == "kernel" or (a.pseudo and epoch % len(pool_t) == 0):  # 用 pool_t 的軟目標：一般為核目標，--pseudo 的第 0 次抽樣為 soft0（真實題 one-hot、偽標籤題機率）
                ce = kernel_ce_torch(logits, targets[bt])  # 對軟目標的交叉熵
            else:  # 一般交叉熵
                ce = torch.nn.functional.cross_entropy(logits, y_t[bt], reduction="none")  # 逐題交叉熵
            loss = (ce * weight[bt]).sum() / weight[bt].sum()  # 加權平均
            if model.aux_loss is not None:  # M30 clip：再加 clip_weight × InfoNCE
                run_aux, n_aux = run_aux + model.aux_loss.item(), n_aux + 1  # 累計供紀錄
                loss = loss + a.clip_weight * model.aux_loss  # 加入損失
            if pc_head is not None:  # M33：再加 player_contrast × 批次中同玩家配對的 InfoNCE
                nce = player_contrast_loss(model.q_embed, pc_head, a.pc_temp) if a.pc_views == 2 else supcon_loss(model.q_embed, pc_head, a.pc_views, a.pc_temp)  # 兩題一組用 InfoNCE，多題一組用監督式對比損失
                run_aux, n_aux = run_aux + nce.item(), n_aux + 1  # 累計供紀錄
                loss = loss + a.player_contrast * nce  # 加入損失
            if ga_head is not None and model.game_vecs is not None:  # M134：每個局向量也要預測所屬題目的等級
                loss = loss + a.game_aux * torch.nn.functional.cross_entropy(ga_head(model.game_vecs), y_t[bt][model.game_owner])  # 加入局層輔助損失
            if ext is not None:  # M135：再加 ext_weight × 一批外部題的類別平衡交叉熵（放在比賽各項之後，因為那些項讀的是模型最後一次前向）
                be = ext["tr"][ext["rng"].randint(0, len(ext["tr"]), a.ext_batch)]  # 隨機抽一批外部訓練題
                le, it_e = ext_logits(be)  # 外部輸出頭的 logits
                we = ext["w_full"][it_e]  # 這些題的類別平衡權重
                ce_e = (torch.nn.functional.cross_entropy(le, ext["y"][it_e], reduction="none") * we).sum() / we.sum()  # 加權交叉熵
                ext["run"], ext["n"] = ext["run"] + ce_e.item(), ext["n"] + 1  # 累計供紀錄
                loss = loss + a.ext_weight * ce_e  # 加入損失
            opt.zero_grad(set_to_none=True)  # 清掉上一步的梯度
            loss.backward()  # 反向傳播
            torch.nn.utils.clip_grad_norm_(train_params, 5.0)  # 梯度裁剪
            opt.step()  # 更新參數
            sched.step()  # 更新學習率
            run, n_run, step = run + loss.item(), n_run + 1, step + 1  # 累計損失與步數
            if a.inner_frac > 0 and step % every == 0:  # 定期在 inner 留出題上評估
                Pi = predict(model, R_in, keys_in, X, offsets, device)  # inner 題的機率
                _, si = evaluate(Pi, y_in, q_in, sizes=tuple(range(5, 21)))  # 逐題官方分數（所有題目大小）
                inner = float(np.mean([si[y_in == c].mean() for c in np.unique(y_in)]))  # 類別平衡的平均分數
                history.append({"step": step, "epoch": step / (len(R_tr) // a.batch), "train_loss": run / max(n_run, 1), "inner": inner, **({"clip": run_aux / n_aux} if n_aux else {})})  # 記錄這次評估
                print(f"{a.arm} seed {a.seed} step {step}/{steps} (epoch {history[-1]['epoch']:.2f}): loss {run / max(n_run, 1):.4f}" + (f" (InfoNCE {run_aux / n_aux:.3f})" if n_aux else "")  # 回報步數、損失與 InfoNCE
                      + (f" (external CE {ext['run'] / max(ext['n'], 1):.3f})" if ext is not None else "") + f"  inner {inner:.4f}  {time.time() - t0:.0f} s", flush=True)  # 以及外部交叉熵、inner 分數與耗時
                if ext is not None:  # 有外部集合
                    history[-1]["ext_ce"] = ext["run"] / max(ext["n"], 1)  # 記錄外部交叉熵
                    ext["run"], ext["n"] = 0.0, 0  # 重新累計
                run, n_run, run_aux, n_aux = 0.0, 0, 0.0, 0  # 重新累計
                if a.swa:  # 需要 SWA
                    eval_states.append((step, {k: v.detach().clone() for k, v in model.state_dict().items()}))  # 保存這次評估的權重
                if inner > best:  # 有進步
                    best, best_step, since = inner, step, 0  # 更新最佳紀錄
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}  # 保存最佳權重
                else:  # 沒有進步
                    since += 1  # 累計未進步次數
                    if since >= a.patience:  # 達到 patience
                        stop = True  # 提早停止
                        break  # 跳出批次迴圈
                model.train()  # predict 切到了評估模式，切回訓練模式
        if a.inner_frac <= 0:  # 沒有 inner 留出時
            history.append({"epoch": epoch + 1, "train_loss": run / max(n_run, 1)})  # 每個 epoch 記錄一次
            print(f"{a.arm} seed {a.seed} epoch {epoch + 1}/{a.epochs}: loss {run / max(n_run, 1):.4f}  {time.time() - t_ep:.0f} s  peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)  # 回報損失、這個 epoch 的耗時與 GPU 記憶體峰值
    if best_state is not None:  # 有最佳狀態
        model.load_state_dict(best_state)  # 還原到 inner 最佳的權重
    if a.swa and eval_states:  # 以最佳狀態為中心、a.swa 個評估狀態的視窗取平均
        model.load_state_dict(swa_average(eval_states, best_step, a.swa))  # 改用平均後的權重
        print(f"swa: averaged {min(a.swa, len(eval_states))} evaluation states around step {best_step}", flush=True)  # 回報平均了幾個狀態
    ft_history, ft_best_step = [], None  # 第二階段的紀錄與最佳步數
    if a.loss == "es-ft":  # 期望分微調
        # section 4F 的第二階段：從 CE 的最佳狀態出發，以小的常數學習率最小化負的期望官方分數；由 inner
        # 官方分數決定保留哪一個狀態，CE 的狀態也在候選之列
        if a.inner_frac <= 0:  # 沒有 inner 留出就無法決定停止點
            raise SystemExit("--loss es-ft stops on held-out players: give --inner-frac")  # 停止
        if best_state is None:  # 第一階段沒有評估過
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}  # 以目前的權重作為起點
        S_t = torch.as_tensor(score_matrix(), device=device, dtype=torch.float32)  # 官方分數矩陣 S(k, y)
        opt = torch.optim.AdamW(model.parameters(), lr=a.ft_lr, weight_decay=a.wd)  # 新的最佳化器，常數的小學習率
        R_t, y_t, weight, keys_tr, _ = pool_t[0]  # 只用第 0 次抽樣
        since, stop = 0, False  # 重設提早停止的狀態
        for epoch in range(a.ft_epochs):  # 至多 ft_epochs 個 epoch
            if stop:  # 已經停止
                break  # 結束
            model.train()  # 訓練模式
            perm = rng.permutation(len(keys_tr))  # 打亂題目
            run, n_run = 0.0, 0  # 累計損失
            for i in range(0, len(perm) - a.batch + 1, a.batch):  # 逐批
                logits, bt = logits_of(R_t, keys_tr, perm[i:i + a.batch])  # 前向
                es = expected_score_loss_torch(logits, y_t[bt], S_t)  # 負的期望官方分數
                loss = (es * weight[bt]).sum() / weight[bt].sum()  # 加權平均
                if model.aux_loss is not None:  # clip 編碼器的 InfoNCE
                    loss = loss + a.clip_weight * model.aux_loss  # 一併加入
                opt.zero_grad(set_to_none=True)  # 清掉梯度
                loss.backward()  # 反向傳播
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 梯度裁剪
                opt.step()  # 更新參數（這一階段沒有學習率排程）
                run, n_run, step = run + loss.item(), n_run + 1, step + 1  # 累計損失與步數
                if step % every == 0:  # 定期評估 inner
                    Pi = predict(model, R_in, keys_in, X, offsets, device)  # inner 題的機率
                    _, si = evaluate(Pi, y_in, q_in, sizes=tuple(range(5, 21)))  # 逐題官方分數
                    inner = float(np.mean([si[y_in == c].mean() for c in np.unique(y_in)]))  # 類別平衡的平均
                    ft_history.append({"step": step, "train_loss": run / max(n_run, 1), "inner": inner})  # 記錄這次評估
                    print(f"{a.arm} seed {a.seed} ES step {step}: loss {run / max(n_run, 1):.4f}  inner {inner:.4f} (best {best:.4f})  {time.time() - t0:.0f} s", flush=True)  # 回報損失與 inner 分數
                    run, n_run = 0.0, 0  # 重新累計
                    if inner > best:  # 比目前最佳（含 CE 階段）還好
                        best, best_step, ft_best_step, since = inner, step, step, 0  # 更新最佳紀錄
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}  # 保存權重
                    else:  # 沒有進步
                        since += 1  # 累計未進步次數
                        if since >= a.ft_patience:  # 達到 patience
                            stop = True  # 停止
                            break  # 跳出批次迴圈
                    model.train()  # 切回訓練模式
        model.load_state_dict(best_state)  # 保留 inner 官方分數最好的狀態（可能就是 CE 的）
    ext_report = None  # 外部輸出頭的報告
    if ext is not None and len(ext["held"]):  # 只用於報告：訓練「結束時」的外部輸出頭（它不屬於還原的最佳狀態）在留出單位上的表現
        model.eval()  # 評估模式
        hit = near = 0  # 完全正確與差一級以內的題數
        with torch.no_grad():  # 不需要梯度
            for i in range(0, len(ext["held"]), 64):  # 每 64 題一批
                le, it_e = ext_logits(ext["held"][i:i + 64])  # 留出外部題的 logits
                dlt = (le.argmax(dim=1) - ext["y"][it_e]).abs()  # 預測段位與真實段位的差
                hit, near = hit + int((dlt == 0).sum()), near + int((dlt <= 1).sum())  # 累計
        ext_report = {"held_questions": int(len(ext["held"])), "exact": hit / len(ext["held"]), "within_one": near / len(ext["held"]), "ranks": ext["ranks"]}  # 題數、完全正確率、差一級以內的比例與段位名稱
        print(f"external head (state at the end of training, restored body) on {len(ext['held'])} held-out external questions: exact {ext_report['exact']:.3f}, within one rank {ext_report['within_one']:.3f}", flush=True)  # 回報
    tag = (a.arm + (f"-d{int(a.s_dim)}l{int(a.s_layers)}" if a.arm == "S" and (int(a.s_dim), int(a.s_layers)) != (192, 4) else "") + ("-wu" if a.warmup_steps > 0 else "") + (f"-no{a.drop}" if a.drop else "") + ("-rs" if a.resample > 1 else "") + ("-uq" if a.uniqueness else "") + ("-bg" if a.bag < 1 else "")  # 結果檔與檢查點檔名用的臂 tag：臂名稱、S 臂寬深、warmup、消融、重抽、獨特度、bagging
           + {"kernel": "-k", "es-ft": "-es"}.get(a.loss, "") + ("-pma" if a.pooling == "pma" else "") + ("-sumll" if a.pooling == "sumll" else "") + ("" if a.token_enc == "linear" else f"-{a.token_enc}")  # 損失、池化與 token 編碼器
           + ("" if a.player_contrast <= 0 else ("-pc" + ("" if a.player_contrast == 0.1 else f"{a.player_contrast:g}") + ("" if a.pc_temp == 0.1 else f"t{a.pc_temp:g}") + ("" if a.pc_views == 2 else f"v{a.pc_views}")))  # 玩家對比的權重、溫度與每組題數
           + ("-pb" if a.pair_batches and a.player_contrast <= 0 else "")  # 只配對、不加 InfoNCE 的診斷
           + (f"w{a.clip_weight:g}" if a.token_enc == "clip" and a.clip_weight != 0.1 else "") + (f"-swa{a.swa}" if a.swa else "") + ("-pl" if a.pseudo else "")  # clip 權重、SWA 與偽標籤
           + "".join("-" + x for x in data_segments(a)) + vec_tag(a.vec_dir) + ("-ra" if a.r_extra else "") + ("-rh" if a.r_extra_npz else "")  # 訓練資料各段、主幹向量與 R 的附加欄
           + (f"-ga{int(round(100 * a.game_aux))}" if a.game_aux > 0 else "") + (f"-w{a.width_mult:g}" if a.width_mult != 1.0 else "") + (f"-do{int(round(100 * a.dropout))}" if abs(a.dropout - 0.1) > 1e-9 else "")  # 局層監督、加寬與 dropout
           + (f"-pd{int(round(100 * a.ply_drop))}" if a.ply_drop > 0 else "") + (f"-tm{int(round(100 * a.trunk_mask))}" if a.trunk_mask > 0 else "")  # token 擴增：丟手與遮主幹向量
           + (("-ex" + ("" if a.ext_weight == 1.0 else f"{a.ext_weight:g}") + ("" if a.ext_frac >= 1.0 else f"u{int(round(100 * a.ext_frac))}")) if a.ext else "") + (f"-{a.tag_extra}" if a.tag_extra else ""))  # 外部集合與附加的 tag
    out = Path(a.out) / ("all" if is_all else f"val{a.val_fold}")  # 這次訓練的結果檔（.npz、.json）目錄：<out>/all 或 <out>/val<v>
    out.mkdir(parents=True, exist_ok=True)  # 確保目錄存在
    final = {"final": None, "final_nll": None}  # 驗證分數（all 沒有）
    if is_all:  # 提交用的模型
        print(f"{a.arm} seed {a.seed}: stopped at step {best_step or step} (inner {best:.4f}); trained on folds {train_folds}, no validation fold", flush=True)  # 只回報停止點與 inner 分數
    else:  # 有驗證折
        P = predict(model, R_va, keys_va, Xv, offsets_v, device)  # 在停止點評一次驗證題
        res, scores = evaluate(P, y_va, q_va)  # 官方分數與逐題分數
        print(f"{a.arm} seed {a.seed}: stopped at step {best_step or step} (inner {best:.4f}); validation {res['balanced_mean_over_n']:.4f}  NLL {res['nll_balanced']:.4f}", flush=True)  # 回報驗證分數與 NLL
        np.savez(out / f"{tag}_s{a.seed}.npz", proba=P.astype(np.float32), scores=scores)  # 保存驗證題的機率與逐題分數（compare 讀這個檔）
        final = {"final": res["balanced_mean_over_n"], "final_nll": res["nll_balanced"]}  # 記錄驗證分數
    (out / f"{tag}_s{a.seed}.json").write_text(json.dumps({"arm": tag, "drop": a.drop, "resample": a.resample, "uniqueness": a.uniqueness, "bag": a.bag, "mix": a.mix, "loss": a.loss, "pooling": a.pooling, "token_enc": a.token_enc, "clip_weight": a.clip_weight, "player_contrast": a.player_contrast, "pc_temp": a.pc_temp, "pc_views": a.pc_views, "pair_batches": a.pair_batches, "tag_extra": a.tag_extra, "ft_lr": a.ft_lr, "ft_epochs": a.ft_epochs, "ft_history": ft_history, "ft_best_step": ft_best_step, "seed": a.seed, "val_fold": a.val_fold, "n_params": n_params, "block_params": blocks,  # 寫出這次訓練的設定與結果紀錄
                                                             "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "wd": a.wd, "dropout": a.dropout, "history": history, "inner_frac": a.inner_frac,  # 訓練參數與評估紀錄
                                                             "best_step": best_step, "best_inner": best, "steps_per_epoch": len(R_tr) // a.batch,  # 停止點（compare 用來換算停止的 epoch）
                                                             "ext": a.ext or None, "ext_weight": a.ext_weight, "ext_batch": a.ext_batch, "ext_frac": a.ext_frac, "ext_report": ext_report,  # 外部集合的設定與報告
                                                             **final, "wall_s": time.time() - t0}, indent=1))  # 驗證分數與總耗時
    if is_all or a.save_dir:  # 權重，給 scripts/trunk_predict.py 用（提交用的模型一定存；驗證折模型依要求才存）
        save_dir = Path(a.save_dir) if a.save_dir else ROOT / "results" / "models" / "ct"  # 權重目錄（預設 results/models/ct）
        ck = save_dir / f"{'all' if is_all else f'val{a.val_fold}'}_{tag}_s{a.seed}.pt"  # 檢查點檔名
        side = save_checkpoint(ck, model, stats, (mean, std), {  # 存權重與兩組標準化；附屬檔記錄下列訓練事實
            "tag": tag, "seed": a.seed, "val_fold": a.val_fold, "train_folds": train_folds, "drop": a.drop, "mix": a.mix, "loss": a.loss, "pooling": a.pooling, "token_enc": a.token_enc, "clip_weight": a.clip_weight, "player_contrast": a.player_contrast, "pair_batches": a.pair_batches, "tag_extra": a.tag_extra, "r_extra": a.r_extra, "r_extra_npz": a.r_extra_npz, "pc_temp": a.pc_temp, "pc_views": a.pc_views, "epochs": a.epochs, "batch": a.batch,  # 臂、種子、折與各項選項
            "lr": a.lr, "wd": a.wd, "dropout": a.dropout, "inner_frac": a.inner_frac, "inner_questions": a.inner_questions, "held_players": len(held),  # 最佳化參數與 inner 留出設定
            "train_questions": int(len(R_tr)), "best_step": best_step, "best_inner": best, "steps_per_epoch": len(R_tr) // a.batch, "n_params": n_params,  # 訓練題數、停止點與參數量
            "block_params": blocks, "l1b_set": l1b_set_of(a), "l1b_heads": L1B_SETS[l1b_set_of(a)]["heads"],  # 區塊參數量、L1b 組與等級頭
            "l1b_dirs": {str(f): L1B_SETS[l1b_set_of(a)]["l1b"][f] for f in train_folds}, "opp_dirs": {str(f): L1B_SETS[l1b_set_of(a)]["opp"][f] for f in train_folds},  # 各訓練折本人側與對手側的 L1b 目錄
            "hsl_dir": str(HSL.relative_to(ROOT)), "vec_dir": str(Path(a.vec_dir).resolve().relative_to(ROOT)), "cache": str(a.cache),  # Human SL、主幹向量與快取目錄
            "train_folds_n": a.train_folds, "train_games": a.train_games, "train_sizes": a.train_sizes, "with_fold0": a.with_fold0,  # 資料軸的設定
            "extra_games": extra_level(a), "extra_dirs": ({str(k): {"hsl": str(EXTRAS[k]["hsl"]), "vec": str(EXTRAS[k]["vec"])} for k in extra_sets(extra_level(a))} or None),  # 額外棋局的等級與目錄
            "opp_hsl": opp_hsl_level(a) > 0, "opp_hsl_r": opp_hsl_level(a) == 2, "opp_hsl_dirs": ([str(OPP_HSL.relative_to(ROOT))] + [str(opp_hsl_dir(k).relative_to(ROOT)) for k in extra_sets(extra_level(a))]) if opp_hsl_level(a) else None,  # 對手 HSL 的設定與目錄
            "swa": a.swa, "pseudo": a.pseudo, "pseudo_weight": a.pseudo_weight, "ext": a.ext or None, "ext_weight": a.ext_weight, "ext_batch": a.ext_batch, **final, "wall_s": time.time() - t0})  # SWA、偽標籤、外部集合、驗證分數與耗時
        print(f"saved {ck} (sha256 {side['state_sha256'][:12]})", flush=True)  # 回報檢查點與雜湊
    return 0  # 成功


def compare(a):  # compare 子命令：比較各臂
    """Per-question scores averaged over seeds, per fold and pooled; conditions (1) CI lower bound > 0, (3) > 0 in all but one fold, (4) >= 0.005."""
    folds = a.val_folds or [a.val_fold]  # 要比較的驗證折
    arms = a.arms  # 要比較的臂（tag）
    pairs_spec = [tuple(x.split(":")) for x in a.pairs]  # 成對比較 "A:B"
    sizes = tuple(a.sizes) if getattr(a, "sizes", None) else SIZES  # 評估的題目大小
    base = argparse.Namespace(**{**vars(a), "train_folds": 0, "train_games": 0, "train_sizes": None, "mix": 0.0, "extra_games": False, "with_fold0": False, "l1b_set": 3, "opp_hsl": False, "opp_hsl_r": False})  # 每個資料臂的驗證題都相同（以預設的資料設定讀驗證題）
    ys, pids, ns, sc, per_fold, seeds, stops = [], [], [], {}, {}, {}, {}  # 標籤、玩家、局數、逐題分數、每折分數、每種子分數、停止點
    for v in folds:  # 逐驗證折
        _, _, _, y, _, q = r_features(base, v)  # 驗證題的標籤與題目表
        d = Path(a.out) / f"val{v}"  # train 寫出這個驗證折各臂結果的目錄
        pid, n = q["player_id"].to_numpy(dtype=str), q["n_games"].to_numpy()  # 驗證題的玩家與局數
        ys.append(y); pids.append(pid); ns.append(n)  # 收集
        for arm in arms:  # 逐臂
            files = [f for f in sorted(d.glob(f"{arm}_s*.npz")) if a.seeds is None or int(f.stem.rsplit("_s", 1)[1]) in a.seeds]  # 這個臂各種子的結果檔（可限定種子）
            if not files:  # 沒有結果
                raise SystemExit(f"compare: no {arm}_s*.npz for validation fold {v} in {d}" + (f" with seeds {a.seeds}" if a.seeds else ""))  # 停止
            s = [np.load(f)["scores"] for f in files]  # 各種子的逐題分數
            sc.setdefault(arm, []).append(np.mean(s, axis=0))  # 各種子平均後的逐題分數
            per_fold.setdefault(arm, []).append(float(np.mean([balanced_mean_over_n(x, y, n, sizes=sizes) for x in s])))  # 這一折各種子分數的平均
            seeds.setdefault(arm, {})[str(v)] = [balanced_mean_over_n(x, y, n, sizes=sizes) for x in s]  # 各種子的分數
            metas = [json.loads(f.with_suffix(".json").read_text()) for f in files]  # 各種子的訓練紀錄
            stops.setdefault(arm, {})[str(v)] = [m["best_step"] / m["steps_per_epoch"] for m in metas]  # 停止點換算成 epoch
    y, pid, n = np.concatenate(ys), np.concatenate(pids), np.concatenate(ns)  # 合併各折
    S = {arm: np.concatenate(sc[arm]) for arm in arms}  # 各臂合併後的逐題分數
    rep = {}  # 各臂的報告
    for arm in arms:  # 逐臂
        rep[arm] = {"score": float(np.mean(per_fold[arm])), "per_fold": per_fold[arm], "seeds": seeds[arm], "stop_epoch": stops[arm], "by_n": {str(m): balanced_mean_over_n(S[arm], y, n, sizes=(m,)) for m in SIZES}}  # 平均分數、每折、每種子、停止點與按局數的分數
        print(f"{arm}: {rep[arm]['score']:.4f}  folds " + " ".join(f"{x:.4f}" for x in per_fold[arm]) + "  by n " + " ".join(f"{m}: {rep[arm]['by_n'][str(m)]:.4f}" for m in SIZES)  # 回報分數、每折分數與按局數的分數
              + "  stop epoch " + " ".join(f"{np.mean(v):.2f}" for v in stops[arm].values()))  # 以及各折的平均停止 epoch
    pairs, need = {}, max(1, len(folds) - 1)  # 成對結果；條件 (3) 需要為正的折數
    for x, z in pairs_spec:  # 逐對比較
        dd = paired_diff(S[x], S[z], y, pid, n, sizes=sizes, n=a.n_boot)  # 成對差與玩家 bootstrap 的 CI
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]  # 每折的差
        cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= need), "4": bool(dd["diff"] >= 0.005)}  # 條件 (1) CI 下界 > 0、(3) 至多一折不為正、(4) 差 >= 0.005
        pairs[f"{x} vs {z}"] = {**dd, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}  # 三個條件都過才算通過
        print(f"{x} vs {z}: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {v:+.4f}" for m, v in dd["by_n"].items())  # 回報差、CI、每折差與按局數的差
              + f"  -> {'PASS' if pairs[f'{x} vs {z}']['pass'] else 'no'}")  # 以及是否通過
    if sizes != SIZES:  # 只評部分題目大小時
        print(f"evaluated on questions of {sizes} games only")  # 提醒
    Path(a.out, f"compare_{a.tag or ''}{'_'.join(map(str, folds))}.json").write_text(json.dumps({"folds": folds, "sizes": list(sizes), "arms": rep, "pairs": pairs}, indent=1, default=float))  # 寫出比較結果
    return 0  # 成功


def rfeat(a):  # rfeat 子命令：先算好 R 特徵快取
    r_features(a, a.val_fold, a.pool)  # 算出並寫入快取
    print(f"R features of validation fold {a.val_fold}, draw {a.pool} cached", flush=True)  # 回報
    return 0  # 成功


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 說明取模組說明的第一行
    ap.add_argument("cmd", choices=("build", "rfeat", "train", "compare"))  # 子命令
    ap.add_argument("--out", default=str(ROOT / "results/trunk_seq"))  # train 的結果檔與 compare 的比較結果的輸出目錄
    ap.add_argument("--cache", default=str(ROOT / "results/trunk_seq"), help="the score arrays and R features")  # 分數陣列與 R 特徵的快取目錄
    ap.add_argument("--val-folds", type=int, nargs="+", default=None)  # compare：多個驗證折
    ap.add_argument("--drop", choices=("trunk", "scores"), default=None, help="ablation: zero the trunk vectors or the per-move scores")  # 消融：主幹向量或逐手分數歸零
    ap.add_argument("--arms", nargs="+", default=list(ARMS))  # compare：要比較的臂（tag）
    ap.add_argument("--pairs", nargs="+", default=["T:C", "C:R0", "T:R0"])  # compare：成對比較
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="compare: only these seeds")  # compare：只用這些種子
    ap.add_argument("--tag", default=None, help="compare: prefix of the output file")  # compare：輸出檔名的前綴
    ap.add_argument("--inner-frac", type=float, default=0.0)  # inner 留出玩家的比例（0 = 不留出）
    ap.add_argument("--resample", type=int, default=1, help="number of training-question draws, cycled by epoch")  # 訓練題的抽樣次數
    ap.add_argument("--uniqueness", action="store_true", help="weight questions by the mean uniqueness of their games")  # 依棋局獨特度加權
    ap.add_argument("--bag", type=float, default=1.0, help="share of training players this model sees")  # bagging 的玩家比例
    ap.add_argument("--mix", type=float, default=0.0, help="section 4L.4: share of training questions made of same-rank games of distinct other players (tag -mix<pct>)")  # 混人題比例（section 4L.4）
    ap.add_argument("--pool", type=int, default=0, help="rfeat: which draw of training questions")  # rfeat：第幾次抽樣
    ap.add_argument("--train-folds", type=int, default=0, help="section 4M M1: only this many of the other folds train (2: fold (v mod 4) + 1 is dropped; tag -f2)")  # section 4M M1：訓練折數
    ap.add_argument("--train-games", type=int, default=0, help="section 4M M2: each training player keeps at most this many scored games (tag -g<n>)")  # section 4M M2：每人局數上限
    ap.add_argument("--train-sizes", type=int, nargs=2, default=None, metavar=("MIN", "MAX"), help="section 4M M2: training question sizes (default 5 20; tag s<min>-<max>)")  # section 4M M2：訓練題的大小範圍
    ap.add_argument("--vec-dir", default=str(VEC), help="trunk vectors (section 4M M3: results/trunk_vec_ft, tag -ft); `build --vec-dir` writes their score arrays")  # 主幹向量目錄（section 4M M3）
    ap.add_argument("--sizes", type=int, nargs="+", default=None, help="compare: evaluate only questions of these sizes (default 5 10 20)")  # compare：只評這些題目大小
    ap.add_argument("--pooling", choices=POOLINGS, default="meanstd", help="section 4M M4: pma = attention pooling over plies and over games (tag -pma)")  # 池化方式（section 4M M4、M134）
    ap.add_argument("--token-enc", choices=TOKEN_ENCODERS, default="linear", help="section 4M M30: dual = board and score towers (tag -dual); clip = dual + CLIP InfoNCE between them (tag -clip)")  # token 編碼器（section 4M M30）
    ap.add_argument("--clip-weight", type=float, default=0.1, help="M30 clip: weight of the InfoNCE in the training loss (tag -clipw<w> when not 0.1)")  # M30 clip：InfoNCE 的權重
    ap.add_argument("--player-contrast", type=float, default=0.0, help="section 4M M33: weight of the same-player InfoNCE between paired questions (tag -pc)")  # section 4M M33：玩家對比損失的權重
    ap.add_argument("--pair-batches", action="store_true", help="M33 diagnostic: same-player paired batches without the InfoNCE (tag -pb)")  # M33 診斷：只做配對批次
    ap.add_argument("--pc-temp", type=float, default=0.1, help="section 4M M39: temperature of the player-contrastive loss")  # section 4M M39：對比損失的溫度
    ap.add_argument("--pc-views", type=int, default=2, help="section 4M M39: same-player questions a group (> 2: supervised-contrastive loss)")  # section 4M M39：每組幾題同一玩家的題目
    ap.add_argument("--r-extra-npz", default="", help="section 4M M40: npz pattern with {v}: X_tr / X_va per-question columns appended to R (tag -rh)")  # section 4M M40：附加到 R 的逐題欄位
    ap.add_argument("--r-extra", default="", help="section 4M M36: score parts of the attention heads; R gets their 20 question columns (tag -ra)")  # section 4M M36：注意力頭的評分分段檔目錄
    ap.add_argument("--r-extra-games", default=str(ROOT / "results/r_variants/attn_games.csv"), help="M36: the games CSV (game_key, fold)")  # M36：棋局 CSV
    ap.add_argument("--game-aux", type=float, default=0.0, help="section 4M M134: weight of a training-only loss in which every game vector predicts its question's level (tag -ga<100 W>)")  # section 4M M134：局層輔助損失的權重
    ap.add_argument("--width-mult", type=float, default=1.0, help="section 4M M134: multiplies the width of the C / T blocks (tag -w<M>); a --dropout other than 0.1 adds -do<100 p> to the tag")  # section 4M M134：C / T 區塊的加寬倍數
    ap.add_argument("--ply-drop", type=float, default=0.0, help="section 4M M133: training only - every game of a batch drops a random share (uniform in 0 .. P) of its plies (tag -pd<100 P>)")  # section 4M M133：丟手比例的上限
    ap.add_argument("--trunk-mask", type=float, default=0.0, help="section 4M M133: training only - a random share (uniform in 0 .. P) of a game's plies gets the mean trunk vector (tag -tm<100 P>)")  # section 4M M133：遮主幹向量比例的上限
    ap.add_argument("--ext", default="", help="section 4M M135: root of an external-players set (ext_dirs); train: every step also takes --ext-batch external questions through the same network and a training-only head over the external ranks (tag -ex); build: write the set's score array")  # section 4M M135：外部玩家集合的根目錄
    ap.add_argument("--ext-weight", type=float, default=1.0, help="M135: weight of the external cross-entropy (tag -ex<w> when not 1)")  # M135：外部交叉熵的權重
    ap.add_argument("--ext-batch", type=int, default=32, help="M135: external questions per step")  # M135：每步的外部題數
    ap.add_argument("--ext-frac", type=float, default=1.0, help="M135: share of the external units that train (tag suffix u<pct> when below 1)")  # M135：參與訓練的外部單位比例
    ap.add_argument("--tag-extra", default="", help="appended as -X to the arm tag (a re-run beside the original files)")  # 附加在臂 tag 後的字串
    ap.add_argument("--extra-games", type=int, nargs="?", const=40, default=0, choices=(0, 40, 80), help="section 4M M8: training players also draw from their games 21-40 (40, tag -x40) or 21-80 (80, tag -x80)")  # section 4M M8：額外棋局（40 或 80）
    ap.add_argument("--swa", type=int, default=0, help="average the weights of the K inner evaluations centred on the best one instead of the best alone (tag -swaK)")  # 平均最佳點附近 K 次評估的權重
    ap.add_argument("--pseudo", default=None, help="CV transduction: an npz with proba [Q_val, 10] aligned to the validation questions, added as soft-labelled training questions (tag -pl)")  # CV 轉導用的偽標籤機率檔
    ap.add_argument("--pseudo-weight", type=float, default=1.0, help="weight of each pseudo-labelled question relative to a real one")  # 偽標籤題的權重
    ap.add_argument("--with-fold0", action="store_true", help="section 4M M8, --val-fold all only: fold 0's players join the training set (tag -f0)")  # section 4M M8：第 0 折的玩家加入訓練
    ap.add_argument("--build-folds", type=int, nargs="+", default=None, help="build: only these folds (default 1-4; 0 for the fold-0 vectors)")  # build：只處理這些折
    ap.add_argument("--l1b-set", type=int, choices=sorted(L1B_SETS), default=3, help="section 4M M9: which L1b heads' log p feed the tokens and R (4 = heads4; tag -h4)")  # section 4M M9：使用哪一組 L1b 等級頭
    ap.add_argument("--opp-hsl", action="store_true", help="section 4M M14: the opponent plies' Human SL token block holds the opponent's centred log p (results/scores_opp_hd[_x40]; tag -oh)")  # section 4M M14：對手手的 Human SL token 區塊改填對手著手的置中 log p
    ap.add_argument("--opp-hsl-r", action="store_true", help="section 4M M14: --opp-hsl, and the 39 opponent-HSL question features join R (64 -> 103; tag -ohr)")  # section 4M M14：另把對手 HSL 題目特徵併入 R
    ap.add_argument("--s-dim", type=int, default=192, help="S arm width (section 4M M21b; tag -d<dim>l<layers> when not 192/4)")  # S 臂的寬度（section 4M M21b）
    ap.add_argument("--s-layers", type=int, default=4, help="S arm depth")  # S 臂的層數
    ap.add_argument("--warmup-steps", type=int, default=0, help="linear LR warmup steps before the cosine schedule (tag -wu)")  # 學習率 warmup 的步數
    ap.add_argument("--inner-questions", type=int, default=4000)  # inner 評估用的題數上限
    ap.add_argument("--evals-per-epoch", type=int, default=4)  # 每個 epoch 評估 inner 的次數
    ap.add_argument("--patience", type=int, default=8)  # 連續幾次沒進步就停止
    ap.add_argument("--arm", choices=("R0", "C", "T", "S"), help="S (section 4M M21): 4-layer transformer over plies, d 192, full attention")  # 要訓練的臂
    ap.add_argument("--val-fold", default="1", help="1-4, or 'all': train on every fold's players (the submission's models), no validation fold")  # 驗證折（1-4）或 all
    ap.add_argument("--save-dir", default=None, help="train: write the weights (.pt + .json) here; 'all' runs always save, to results/models/ct")  # 權重的輸出目錄
    ap.add_argument("--seed", type=int, default=0)  # 種子
    ap.add_argument("--epochs", type=int, default=8)  # epoch 上限
    ap.add_argument("--batch", type=int, default=32)  # 每批題數
    ap.add_argument("--lr", type=float, default=1e-3)  # 學習率
    ap.add_argument("--wd", type=float, default=1e-4)  # 權重衰減
    ap.add_argument("--dropout", type=float, default=0.1)  # dropout 比例
    ap.add_argument("--loss", choices=("ce", "kernel", "es-ft"), default="ce", help="section 4F: the score kernel's soft targets, or the expected-score fine-tune")  # 損失（section 4F）
    ap.add_argument("--ft-epochs", type=int, default=2, help="es-ft: epochs cap of the second stage")  # es-ft 第二階段的 epoch 上限
    ap.add_argument("--ft-lr", type=float, default=1e-4, help="es-ft: constant learning rate of the second stage")  # es-ft 第二階段的學習率
    ap.add_argument("--ft-patience", type=int, default=8, help="es-ft: inner evaluations without improvement before stopping")  # es-ft 的 patience
    ap.add_argument("--limit", type=int, default=0, help="smoke: only this many training questions")  # 冒煙測試：只用這麼多訓練題
    ap.add_argument("--n-boot", type=int, default=5000)  # bootstrap 的次數
    a = ap.parse_args(argv)  # 解析參數
    if a.opp_hsl_r:  # -ohr 隱含 -oh
        a.opp_hsl = True  # 一併啟用 --opp-hsl
    if a.cmd == "train" and a.token_enc != "linear" and a.arm == "R0":  # R0 沒有 token
        ap.error("--token-enc is for the token arms (C, T, S)")  # 參數錯誤
    if a.cmd == "train" and a.r_extra and a.resample > 1:  # r-extra 只能接單一次抽樣
        ap.error("--r-extra takes a single training draw (no --resample)")  # 參數錯誤
    if a.cmd == "train" and a.pc_views < 2 or (a.cmd == "train" and a.player_contrast > 0 and a.batch % a.pc_views):  # 每組題數小於 2，或用玩家對比時批次大小不能被它整除
        ap.error("--pc-views must be >= 2 and divide --batch")  # 參數錯誤
    if a.cmd == "train" and (a.player_contrast > 0 or a.pair_batches) and (a.arm == "R0" or a.pseudo or a.batch % 2 or a.loss != "ce"):  # 配對批次的前提不成立
        ap.error("--player-contrast needs a token arm, an even --batch, --loss ce and no --pseudo")  # 參數錯誤
    if a.val_fold != "all":  # 不是 all 就必須是整數
        try:  # 嘗試轉成整數
            a.val_fold = int(a.val_fold)  # 轉換
        except ValueError:  # 不是整數
            ap.error("--val-fold takes 1-4 or 'all'")  # 參數錯誤
    if a.val_fold == 0 or 0 in (a.val_folds or []):  # 第 0 折不當驗證折
        ap.error("fold 0 is not used here")  # 參數錯誤
    if a.val_fold == "all" and a.cmd == "compare":  # compare 需要驗證折
        ap.error("compare takes validation folds")  # 參數錯誤
    if a.train_sizes and not (5 <= a.train_sizes[0] <= a.train_sizes[1] <= 20):  # 題目大小範圍不合法
        ap.error("--train-sizes takes 5 <= MIN <= MAX <= 20")  # 參數錯誤
    if a.train_games and a.train_games < (a.train_sizes[1] if a.train_sizes else 20):  # 局數上限小於最大的題目大小
        ap.error("--train-games must cover the largest training question size (give --train-sizes)")  # 參數錯誤
    if a.ext and a.cmd == "train" and (a.pooling != "meanstd" or a.pseudo or a.loss == "es-ft" or a.resample > 1 or a.arm == "S"):  # 外部集合只支援部分組合
        ap.error("--ext is implemented for the C / T / R0 arms with mean/std pooling, a single draw, no --pseudo and no es-ft stage")  # 參數錯誤
    if a.ext and a.cmd == "build":  # build --ext：外部集合的建置
        return ext_build(a)  # 建外部集合的分數陣列
    return {"build": build, "rfeat": rfeat, "train": train, "compare": compare}[a.cmd](a)  # 依子命令執行


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值作為結束碼

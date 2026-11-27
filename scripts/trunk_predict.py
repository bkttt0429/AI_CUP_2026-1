#!/usr/bin/env python
"""Official test CSVs -> a second submission from the trunk + TCN ensemble (architecture v2 sections 4A / 4H: the C and T
arms of scripts/trunk_seq_check.py, 12 models, folds 1-4 CV 0.7258 against the fused MLP's 0.7026), built on top of the main
pipeline's shard files. Nothing here changes the main submission (predict_submission.py --merge): this is a separate output.

  extract  replay every usable game (predict_submission.py's tasks, in its order; the same check_game) and store the per-ply
           trunk vectors of the pristine humanv0 trunk exactly as scripts/trunk_extract.py stores them for the training folds
           (both players' plies, the fixed random projections to 64 + 64 dims, fp16), as parts under --vec; sharded over GPUs
           like the scoring step (--shard i --nshards N; parts resume per game).
  predict  per question and per L1b model of the shards (heads3_A, heads3_B): the 64 R features (fused_opp_diff_hd, from the
           shards' sums and the games' HD[]) and the per-ply tokens (trunk vectors + the shards' per-move Human SL log p of the
           target, L1b log p of the target and of the opponent + flags), standardised as each checkpoint was trained; each
           checkpoint's softmax, averaged over the checkpoints and the L1b models; the expected-score decision. A checkpoint whose
           sidecar says opp_hsl (section 4M M14, the ct6 models) reads tokens whose opponent plies carry the opponent's Human SL
           log p from the shards' 18-profile opponent pass (hsl_opp_move_logp), and with opp_hsl_r an R with the 39 opponent-HSL
           question features (103 columns); shards without that pass are refused for such checkpoints. A question
           without a usable game, or with non-finite probabilities, takes the main submission's row (--fallback-submission) or
           FALLBACK and is listed in the report. Exit codes as predict_submission.py: 0 ok, 1 the file failed validation, 2 bad
           inputs / shards / models, 3 a fallback row (--allow-fallback accepts), 4 lenient-input warnings (--allow-input-warnings).
  offline  the same checkpoints on a mock test built from one training fold (experiments/make_mock_test.py --fold f), through the
           training-side tables and trunk vectors (results/scores_hd, results/l1b/*_hd, results/trunk_vec/fold f): the reference
           of the consistency check of the test-side path.
  compare  two probability files side by side: max / mean |dp|, questions whose decisions differ, and with --answers the
           official score of each.

  sbatch -p 8gpus --gres=gpu:8 -c 96 -t 02:00:00 -J ctvec scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 \\
      python -I scripts/trunk_predict.py extract --inputs data/tests/rank_prediction_test_public.csv data/tests/rank_prediction_test_private.csv \\
      --vec results/ct_submission/vec --workers 10
  sbatch -t 01:00:00 -J ctpred scripts/nano4/run.sbatch python -I scripts/trunk_predict.py predict --inputs ... --shards submission_rank.csv \\
      --nshards 8 --vec results/ct_submission/vec --models results/models/ct/all_*.pt --fallback-submission submission_rank.csv \\
      --out submission_rank_ct.csv
"""
import argparse  # 解析命令列參數與子命令
import hashlib  # 以雜湊判斷 token 統計量是否相同
import json  # 讀寫 meta、紀錄與報告
import multiprocessing as mp  # 重播棋局的 CPU 程序池
import sys  # 設定匯入路徑、輸出錯誤與結束碼
import time  # 計時
from collections import Counter  # 計數任務、臂與預測等級
from datetime import datetime  # 紀錄的寫出時間
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算
import pandas as pd  # 讀寫 CSV
import torch  # 主幹與檢查點的推論

ROOT = Path(__file__).resolve().parents[1]  # 專案根目錄（本檔在 scripts/）
for d in (ROOT / "src", ROOT / "scripts"):  # src 與 scripts 都要能匯入
    if str(d) not in sys.path:  # 還不在匯入路徑上時
        sys.path.insert(0, str(d))  # 加到最前面
from aicup.fused import FUSED_OPP_DIFF_HD_FEATURE_NAMES, fused_opp_diff_features  # noqa: E402  R 特徵的欄名與計算
from aicup.handicap import question_hd_features_from_games  # noqa: E402  題目的讓子特徵
from aicup.humansl import PROFILES  # noqa: E402  18 個等級剖面
from aicup.opphsl import N_OPP_HSL, opp_hsl_features, question_features as opp_hsl_question_features  # noqa: E402  39 個對手 HSL 特徵
from aicup.metric import LEVELS, expected_score_decision, score  # noqa: E402  等級、期望分數決策與官方逐題分數
from aicup.qeval import balanced_mean_over_n  # noqa: E402  逐題官方分數的類別平衡平均（各題目大小分別計算再平均）
from aicup.scoring import completed_parts, part_paths, resume_plan  # noqa: E402  分段檔的列舉、路徑與續跑
from aicup.setmodel_io import file_sha256  # noqa: E402  計算檔案 sha256 的函式（本檔沒有直接呼叫）
from aicup.submission import validate_submission, write_submission  # noqa: E402  提交檔的寫出與驗證
import predict_submission as ps  # noqa: E402  主流程：讀題、任務清單與分片檔
import trunk_seq_check as ts  # noqa: E402  主幹 + TCN 的訓練器：token、檢查點與預測

EXIT_OK, EXIT_INVALID, EXIT_INPUTS, EXIT_FALLBACK, EXIT_INPUT_WARNINGS = ps.EXIT_OK, ps.EXIT_INVALID, ps.EXIT_INPUTS, ps.EXIT_FALLBACK, ps.EXIT_INPUT_WARNINGS  # 結束碼與 predict_submission.py 相同
FALLBACK = ps.FALLBACK  # 無法決定的題目在主提交檔沒有答案時填的等級（同主流程）
P_COLS = [f"p_{lv}" for lv in LEVELS]  # 機率檔的欄名，每個等級一欄


# Section 4M M7（2026-11-04）：只用一個題目大小區間訓練的檢查點（trunk_seq_check.py --train-sizes lo hi，側檔
# train_sizes）只對可用局數落在該區間的題目投票；其他檢查點對所有題目投票（votes_of／ensemble）。


def task_key(k):  # 任務編號 → 向量分段檔中的 game_key
    """The vec parts' game_key of task k: a string, so the CSV round trip of the part tables keeps it."""
    return f"g{k:07d}"  # 字母 g 開頭並補零到 7 位，CSV 讀回時仍是字串


def key_task(key):  # game_key → 任務編號
    return int(key[1:])  # 去掉字母前綴


# ---------------------------------------------------------------- extract 子命令

def extract(a):  # extract 子命令：測試局的逐手主幹向量
    import trunk_extract as te  # 沿用訓練端的重播、投影與寫檔函式
    from aicup.humansl import load_humansl  # 載入 humanv0
    from aicup.l1b import neutral_rows, trunk_features  # 中性 metadata 與主幹輸出

    t0 = time.time()  # 計時起點
    try:  # 讀入測試題目
        questions, warnings = ps.load_questions(a.inputs, a.lenient_inputs)  # 與主流程相同的讀法
    except ValueError as e:  # 輸入格式錯誤
        print(e, file=sys.stderr)  # 印出原因
        return EXIT_INPUTS  # 結束碼 2
    tasks, owner, dropped, _ = ps.build_tasks(questions)  # 與主流程相同的任務清單與順序
    sha = ps.questions_sha(a.inputs, questions)  # 題目雜湊，記入紀錄
    mine = tasks[a.shard::a.nshards]  # 本分片負責的任務
    print(f"{len(questions)} questions, {len(tasks)} usable games ({len(dropped)} dropped); shard {a.shard} of {a.nshards}: {len(mine)} games"  # 回報題目數、可用局數與本分片局數
          + (f"; {len(warnings)} input warnings" if warnings else ""), flush=True)  # 以及輸入警告數
    vec = Path(a.vec)  # 向量的輸出資料夾
    vec.mkdir(parents=True, exist_ok=True)  # 建立資料夾
    meta = {task_key(k): {"game_key": task_key(k), "task": k, "question_id": questions[owner[k]]["question_id"], "color": c} for k, _, c in mine}  # 每局的表格欄位：鍵、任務、題號與本人顏色
    sgf_of = {task_key(k): (sgf, c) for k, sgf, c in mine}  # 鍵 → (棋譜, 本人顏色)
    plan = resume_plan(list(meta), vec, a.shard, a.part_size)  # 只排還沒完成的局
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]  # 模型設定，給 worker 建特徵器
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))  # 在初始化 CUDA 之前 fork 出程序池
    plies = 0  # 本次處理的手數
    try:  # 確保最後關閉程序池
        model, _ = load_humansl(str(te.CKPT), "cuda")  # 原始的 humanv0（不用微調主幹）
        model.eval()  # 推論模式
        rows, proj_g, proj_l = neutral_rows(), None, None  # 中性 metadata；投影稍後建立
        for n_part, (k, part_keys) in enumerate(plan):  # 逐個分段處理
            done = list(pool.imap(te.replay, [(q, *sgf_of[q]) for q in part_keys], chunksize=1))  # 以 CPU 程序池重播，保持順序
            if proj_g is None:  # scripts/trunk_extract.py 的投影（種子 PROJ_SEED），大小依主幹的通道數決定
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # 推論、bf16
                    trunk, _, _, _ = trunk_features(model, torch.as_tensor(done[0][1][:1]).cuda(), torch.as_tensor(done[0][2][:1]).cuda(),  # 用一個局面取得主幹的通道數
                                                    torch.as_tensor(done[0][4][:1]).cuda().long(), rows)  # 下子方與 metadata 列
                proj_g, proj_l = te.projections(trunk.shape[1], trunk.device)  # 與訓練端相同的投影
            plies += te.write_part(vec, a.shard, k, done, meta, model, proj_g, proj_l, rows, a.batch)  # 計算並寫出這個分段
            print(f"part {n_part + 1}/{len(plan)}: {plies} plies, {plies / max(time.time() - t0, 1e-9):.0f} plies/s", flush=True)  # 回報進度與速度
    finally:  # 無論成功或失敗
        pool.close()  # 不再接受任務
        pool.join()  # 等待 worker 結束
    rec = {"inputs": [str(p) for p in a.inputs], "questions_sha256": sha, "n_tasks_total": len(tasks), "shard": a.shard, "nshards": a.nshards,  # 本次抽取的紀錄：輸入、題目雜湊、任務總數與分片
           "games": len(mine), "parts_this_run": len(plan), "plies_this_run": int(plies), "proj_seed": te.PROJ_SEED, "proj_dim": te.PROJ_DIM,  # 局數、分段數、手數與投影設定
           "ckpt": te.CKPT.name, "lenient_inputs": a.lenient_inputs, "input_warnings": warnings, "wall_s": time.time() - t0,  # 檢查點、輸入警告與耗時
           "written": datetime.now().isoformat(timespec="seconds")}  # 寫出時間
    (vec / f"extract_s{a.shard}of{a.nshards}.json").write_text(json.dumps(rec, indent=1))  # 每個分片一個紀錄檔，predict 會收進報告
    print(f"shard {a.shard}: {len(mine)} games, {plies} plies this run, {(time.time() - t0) / 60:.1f} min", flush=True)  # 回報本分片的局數、手數與耗時
    return EXIT_OK  # 正常結束


# ---------------------------------------------------------------- predict 子命令

def vec_index(vec_dir):  # 建立向量分段檔的逐局索引
    """The games of the vec parts in the ply order of trunk_seq_check.raw_tokens: [{key, task, start, n, target [n] bool, t [n]}],
    and the total number of plies. ValueError when a game's plies are not the contiguous 0..n-1 block the parts promise."""
    games, start = [], 0  # 各局資訊與目前的起始位置
    for s, k in completed_parts(vec_dir):  # 依 raw_tokens 讀取的順序走過已完成的分段
        csv_path, npz_path = part_paths(vec_dir, s, k)  # 這個分段的兩個檔案
        tab = pd.read_csv(csv_path, dtype={"game_key": str, "question_id": str, "color": str}, keep_default_na=False)  # 每局表格；鍵、題號與顏色以字串讀入
        with np.load(npz_path) as z:  # 開啟逐手陣列
            row, t, target = z["game_row"].astype(np.int64), z["t"].astype(np.int64), z["target"].astype(bool)  # 每手所屬的列、局內序號與是否本人
        for g, rec in enumerate(tab.to_dict("records")):  # 逐局檢查
            idx = np.flatnonzero(row == g)  # 這一局各手的位置
            if len(idx) != int(rec["n_plies"]) or not np.array_equal(t[idx], np.arange(len(idx))):  # 手數或序號對不上
                raise ValueError(f"{npz_path}: game {rec['game_key']} has {len(idx)} plies stored, n_plies {rec['n_plies']}, or its t is not 0..n-1")  # 資料不一致就停止
            games.append({"key": rec["game_key"], "task": int(rec["task"]), "start": start, "n": len(idx), "target": target[idx], "t": t[idx]})  # 記下這一局在全部手中的位置與本人手的遮罩
            start += len(idx)  # 下一局的起點
    return games, start  # 各局索引與總手數


def load_opp_moves(shards_out, nshards):  # 讀取分片檔中對手各手的 L1b log p
    """{task: (logp [n_opp, K, 10] float64, t [n_opp])}: the opponent plies' L1b log p from the shard files (written since
    2026-11-04); ValueError when a shard lacks them."""
    moves = {}  # 任務 → (log p, 手序)
    for i in range(nshards):  # 逐一讀取分片檔
        path = ps.shard_path(shards_out, i, nshards)  # 第 i 個分片檔的路徑
        with np.load(path, allow_pickle=False) as z:  # 開啟分片檔（不允許 pickle）
            meta = json.loads(str(z["__meta__"]))  # 分片的 meta
            if not meta.get("l1b_opp_moves"):  # 較早的分片沒有對手的逐手資料
                raise ValueError(f"{path}: no per-move L1b log p of the opponents (l1b_opp_moves): the shards must come from "  # 說明分片必須是 2026-11-04 以後的版本
                                 "predict_submission.py of 2026-11-04 or later, scored with --l1b-heads and the opponent pass")  # 錯誤訊息的後半
            mt, logp, t = z["opp_move_task"].astype(np.int64), z["l1b_opp_move_logp"].astype(np.float64), z["opp_move_t"].astype(np.int64)  # 每手所屬任務、log p 與手序
            order = np.argsort(mt, kind="stable")  # 依任務排序，保持同一任務內的順序
            mt, logp, t = mt[order], logp[order], t[order]  # 套用排序
            tasks_here, starts = np.unique(mt, return_index=True)  # 各任務的起始位置
            ends = np.append(starts[1:], len(mt))  # 各任務的結束位置
            for k, s, e in zip(tasks_here.tolist(), starts.tolist(), ends.tolist()):  # 逐任務切片
                moves[k] = (logp[s:e], t[s:e])  # 這一局對手各手的 log p 與手序
    return moves  # 全部任務的對手逐手資料


def score_arrays(games, moves_hsl, moves_l1b, moves_opp, n_models, moves_opp_hsl=None):  # 組出 token 的分數欄
    """[K, plies, N_SCORES] float16 in the ply order of `games` (vec_index): per L1b model k, the target plies hold the centred
    L1b log p of the target's moves and the opponent plies the centred L1b log p of the opponent's moves (columns 0-10), the
    target plies also the centred Human SL log p (columns 11-29) -- trunk_seq_check.build's layout; with moves_opp_hsl (the
    shards' 18-profile opponent pass, section 4M M14) the opponent plies hold the opponent's centred Human SL log p there too
    (build --opp-hsl's layout), else zeros. Every game's target plies must be the shards' Human SL and L1b moves and its other
    plies the opponent moves, ply for ply (ValueError otherwise)."""
    total = sum(g["n"] for g in games)  # 總手數
    sc = np.zeros((n_models, total, ts.N_SCORES), np.float32)  # 每個 L1b 模型一份分數陣列
    bad = []  # 手序對不上的任務
    for g in games:  # 逐局填入
        k, tgt = g["task"], g["target"]  # 任務編號與本人手的遮罩
        hl, ht = moves_hsl[k]  # 本人的 Human SL 逐手 log p 與手序
        ll, lt = moves_l1b[k]  # 本人的 L1b 逐手 log p 與手序
        ol, ot = moves_opp.get(k, (np.zeros((0, n_models, len(LEVELS))), np.zeros(0, dtype=np.int64)))  # 對手的 L1b 逐手資料（沒有就是空的）
        ohl, oht = (None, g["t"][~tgt]) if moves_opp_hsl is None else moves_opp_hsl.get(k, (np.zeros((0, len(PROFILES))), np.zeros(0, dtype=np.int64)))  # 對手的 Human SL 逐手資料（不需要時以向量本身的手序代替）
        if not (np.array_equal(g["t"][tgt], ht) and np.array_equal(ht, lt) and np.array_equal(g["t"][~tgt], ot) and np.array_equal(g["t"][~tgt], oht)):  # 各來源的手序必須與向量分段檔逐手一致
            bad.append(k)  # 記下對不上的任務
            continue  # 跳過這一局
        rows = g["start"] + np.arange(g["n"])  # 這一局在全部手中的位置
        for m in range(n_models):  # 逐個 L1b 模型
            sc[m, rows[tgt], :len(LEVELS) + 1] = ts.centred(ll[:, m, :])  # 本人的手：平均值與置中的 L1b log p
            sc[m, rows[~tgt], :len(LEVELS) + 1] = ts.centred(ol[:, m, :])  # 對手的手：平均值與置中的 L1b log p
        sc[:, rows[tgt], len(LEVELS) + 1:] = ts.centred(hl)[None]  # 本人的手：平均值與置中的 Human SL log p（第 11–29 欄，各模型共用）
        if ohl is not None:  # 需要對手的 Human SL 時
            sc[:, rows[~tgt], len(LEVELS) + 1:] = ts.centred(ohl)[None]  # 對手的手也填入平均值與置中的 Human SL log p
    if bad:  # 有任何一局對不上
        raise ValueError(f"{len(bad)} games whose plies do not line up with the shards' moves, e.g. tasks {bad[:3]}")  # 停止並舉例
    return sc.astype(np.float16)  # 以 fp16 回傳，與訓練端 build 寫的分數陣列同格式


def question_r(qis, by_q, per_task, l1b_per_task, opp_per_task, task_games, m, opp_hsl_per_task=None):  # 每題的 R 特徵（題目層特徵）
    """[Q, 64]: the fused_opp_diff_hd features of the questions `qis` from shard model m's sums and the games' HD[] (what
    predict_submission.py's fused_probs computes for the _hd document, feature for feature); with opp_hsl_per_task (the shards'
    18-profile opponent pass: {k: (the opponent's n_moves, sums [18])}, section 4M M14) [Q, 103]: + the 39 opponent-HSL features."""
    rows = []  # 每題一列
    width = len(FUSED_OPP_DIFF_HD_FEATURE_NAMES) + (N_OPP_HSL if opp_hsl_per_task is not None else 0)  # 預期的欄數：64 或 103
    for qi in qis:  # 逐題計算
        ks = by_q[qi]  # 這一題的可用任務
        hsl, n = np.stack([per_task[k][1] for k in ks]), np.array([per_task[k][0] for k in ks])  # 本人的 Human SL 總和與手數
        l1b, opp = np.stack([l1b_per_task[k][m] for k in ks]), np.stack([opp_per_task[k][1][m] for k in ks])  # 模型 m 的本人與對手 L1b 總和
        opp_n = np.array([opp_per_task[k][0] for k in ks])  # 對手的手數
        feats = [fused_opp_diff_features(hsl, l1b, n, opp, opp_n), question_hd_features_from_games([task_games[k] for k in ks])]  # 融合特徵加上讓子特徵
        if opp_hsl_per_task is not None:  # 需要對手 HSL 特徵時
            feats.append(opp_hsl_question_features(hsl, np.stack([opp_hsl_per_task[k][1] for k in ks]), np.array([opp_hsl_per_task[k][0] for k in ks])))  # 再接上 39 個對手 HSL 題目特徵
        rows.append(np.concatenate(feats))  # 這一題完整的特徵列
    X = np.stack(rows) if rows else np.zeros((0, width))  # 沒有題目時給空陣列
    if X.shape[1] != width:  # 欄數不符
        raise ValueError(f"{X.shape[1]} R features computed, expected {width}")  # 停止
    return X  # 每題的 R 特徵


def votes_of(side, n_games):  # 某個檢查點對哪些題目投票
    """Section 4M M7 routing by question size: a checkpoint whose sidecar records train_sizes [lo, hi] votes only on questions of lo..hi
    games; one without votes everywhere. Returns (bool [Q], band label)."""
    sizes = side.get("train_sizes")  # 訓練時的題目大小區間
    if not sizes:  # 沒有限制
        return np.ones(len(n_games), dtype=bool), "all"  # 對所有題目投票
    lo, hi = int(sizes[0]), int(sizes[1])  # 區間的上下限
    return (n_games >= lo) & (n_games <= hi), f"{lo}-{hi}"  # 只對區間內的題目投票


def ensemble(ckpts, R_by_model, keys, X_raw_by_model, offsets, device, model_names=None):  # 所有檢查點的機率平均
    """(P [Q, 10] mean over the checkpoints x L1b models that vote on each question, {tag: P mean over that tag's checkpoints}, passes,
    {band: questions it votes on}): every checkpoint's softmax on the tokens standardised with its own token stats (one standardised
    copy per distinct stats) and its own R standardisation; votes_of routes by question size. A question without a vote is NaN.
    model_names (section 4M M9): the shards' L1b model names by index; a checkpoint whose sidecar lists l1b_heads only runs on those.
    X_raw_by_model {(m, oh): X} and R_by_model {(m, ohr): R} (section 4M M14): a checkpoint reads the tokens with the opponent plies'
    Human SL block filled when its sidecar says opp_hsl, and the R with the 39 opponent-HSL columns when it says opp_hsl_r;
    ValueError when the variant it needs was not built or R's width is not its n_r."""
    Q = len(keys)  # 題目數
    n_games = np.array([len(k) for k in keys], dtype=int)  # 每題的可用局數
    total, count, per_arm, n_arm, passes, routing = np.zeros((Q, len(LEVELS))), np.zeros(Q), {}, {}, 0, {}  # 機率總和、投票數、各臂累計與次數、推論趟數與分流統計
    for m in sorted({mk[0] for mk in X_raw_by_model}):  # 逐個 L1b 模型
        cache = {}  # 標準化後的 token（同一模型內重用）
        for model, stats, (r_mean, r_std), side in ckpts:  # 逐個檢查點
            if model_names is not None and side.get("l1b_heads") and model_names[m] not in side["l1b_heads"]:  # 檢查點只吃特定等級頭的 token
                continue  # 這個檢查點的 token 來自其他等級頭的 log p
            oh, ohr = bool(side.get("opp_hsl")), bool(side.get("opp_hsl_r"))  # 需要的 token 與 R 變體
            if (m, oh) not in X_raw_by_model or (m, ohr) not in R_by_model:  # 需要的變體沒有建立
                raise ValueError(f"checkpoint {side.get('tag')} s{side.get('seed')} needs {'the opponent-HSL' if oh else 'the plain'} tokens and "  # 說明缺少哪一種 token 與 R
                                 f"{'the 103-column' if ohr else 'the 64-column'} R, which were not built here")  # 錯誤訊息的後半
            X_raw = X_raw_by_model[(m, oh)]  # 未標準化的 token
            h = (hashlib.sha1(np.ascontiguousarray(stats[0], dtype=np.float32).tobytes() + np.ascontiguousarray(stats[1], dtype=np.float32).tobytes()).hexdigest(), oh)  # token 統計量的雜湊，用來判斷能否重用
            if h not in cache:  # 統計量與上一個檢查點不同
                cache = {h: torch.as_tensor(ts.standardise_tokens(X_raw.copy(), stats), device=device)}  # 重新標準化，只保留一份以節省記憶體
            if R_by_model[(m, ohr)].shape[1] != len(r_mean):  # R 的寬度與檢查點不符
                raise ValueError(f"checkpoint {side.get('tag')} s{side.get('seed')} takes {len(r_mean)} R features, {R_by_model[(m, ohr)].shape[1]} computed")  # 停止
            R = ((R_by_model[(m, ohr)] - r_mean) / r_std).astype(np.float32)  # 以檢查點自己的統計量標準化 R
            P = ts.predict(model, R, keys, cache[h], offsets, device) if Q else np.zeros((0, len(LEVELS)))  # 每題的 softmax 機率
            v, band = votes_of(side, n_games)  # 這個檢查點投票的題目
            routing[band] = int(v.sum())  # 記下各區間投票的題數
            total += P * v[:, None]  # 只累加有投票的題目
            count += v  # 投票次數
            tag = side.get("tag") or side["arm"]  # 臂的標籤
            per_arm[tag] = per_arm.get(tag, 0) + P * v[:, None]  # 各臂的機率累計
            n_arm[tag] = n_arm.get(tag, 0) + v  # 各臂的投票次數
            passes += 1  # 推論趟數加一（每個檢查點 × L1b 模型算一趟，不是棋譜中的 pass）
        del cache  # 釋放這個模型的 token
    with np.errstate(invalid="ignore", divide="ignore"):  # 沒有投票的題目是 0/0，結果為 NaN
        P = total / count[:, None]  # 平均機率
        arms = {tag: per_arm[tag] / n_arm[tag][:, None] for tag in per_arm}  # 各臂的平均機率
    return P, arms, passes, routing  # 回傳集成結果


def predict(a):  # predict 子命令：產生第二份提交檔
    t0 = time.time()  # 計時起點
    out = Path(a.out)  # 提交檔路徑
    report_path = Path(a.report) if a.report else out.with_name(out.stem + "_report.json")  # 報告路徑
    try:  # 讀入測試題目
        questions, input_warnings = ps.load_questions(a.inputs, a.lenient_inputs)  # 與主流程相同的讀法
    except ValueError as e:  # 輸入格式錯誤
        print(e, file=sys.stderr)  # 印出原因
        return EXIT_INPUTS  # 結束碼 2
    tasks, owner, dropped, passes = ps.build_tasks(questions)  # 與主流程相同的任務清單
    sha = ps.questions_sha(a.inputs, questions)  # 題目雜湊，用來核對分片
    hd_counts = ps.handicap_counts(questions)  # 各 HD 值的局數
    print(f"{len(questions)} questions, {sum(len(q['games']) for q in questions)} games, {len(tasks)} usable, {len(dropped)} dropped; handicap HD[] "  # 回報題目數、局數與可用／捨棄局數
          + " ".join(f"{k}:{v}" for k, v in hd_counts.items()), flush=True)  # 以及各 HD 值的局數
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 有 GPU 就用 GPU
    try:  # 載入分片、向量與檢查點；任何不一致都以結束碼 2 結束
        if hd_counts.get("absent", 0) and not a.allow_missing_hd:  # 有局沒有 HD[]
            raise ValueError(f"{hd_counts['absent']} of the games carry no HD[]; the checkpoints take the handicap features (zeros for an unknown "  # 檢查點需要讓子特徵，預設拒絕
                             "HD): pass --allow-missing-hd to go on, or keep the main submission")  # 錯誤訊息的後半
        per_task, shard_metas, moves_hsl = ps.load_shards(a.shards, a.nshards, len(tasks), sha, with_moves=True)  # 本人的 Human SL 總和與逐手 log p
        l1b_models, l1b_per_task, moves_l1b, opp_per_task = ps.load_shards_l1b(a.shards, a.nshards, with_moves=True)  # L1b 模型、本人總和與逐手 log p、對手總和
        if not l1b_models:  # 分片沒有 L1b 分數
            raise ValueError("the shards carry no L1b scores (the shard step needs --l1b-heads)")  # 分片步驟需要 --l1b-heads
        if opp_per_task is None:  # 分片沒有對手分數
            raise ValueError("the shards carry no opponent scores (the shard step must not pass --no-l1b-opponent)")  # 分片步驟不能加 --no-l1b-opponent
        moves_opp = load_opp_moves(a.shards, a.nshards)  # 對手各手的 L1b log p
        names = [m["name"] for m in l1b_models]  # 分片中 L1b 模型的名稱
        use = [i for i, n in enumerate(names) if not a.l1b_names or n in a.l1b_names]  # 要使用的 L1b 模型（依 --l1b-names 篩選）
        missing = sorted(set(a.l1b_names or []) - set(names))  # 指定了但分片沒有的模型
        if missing or not use:  # 有缺，或一個都沒用到
            raise ValueError(f"--l1b-names {missing} are not among the shards' L1b models {names}")  # 停止
        games, n_plies = vec_index(Path(a.vec))  # 向量分段檔的逐局索引
        have = Counter(g["task"] for g in games)  # 每個任務在向量中出現的次數
        if sorted(have) != list(range(len(tasks))) or any(v != 1 for v in have.values()):  # 每個任務必須恰好出現一次
            dup = [k for k, v in have.items() if v > 1]  # 重複的任務
            lost = sorted(set(range(len(tasks))) - set(have))  # 缺少的任務
            raise ValueError(f"the vec parts under {a.vec} hold {len(games)} games for {len(tasks)} tasks: {len(lost)} missing (e.g. {lost[:3]}), "  # 說明缺少與重複的任務數
                             f"{len(dup)} stored twice (e.g. {dup[:3]}); run extract for every shard first")  # 提示先對每個分片執行 extract
        ckpts = [ts.load_checkpoint(p, device) for p in a.models]  # 載入所有檢查點與側檔
        if not ckpts:  # 沒有給檢查點
            raise ValueError("--models is empty")  # 停止
        bad = [p for p, (_, stats, _, side) in zip(a.models, ckpts) if stats is None or side["arm"] == "R0"]  # 沒有 token 統計量的檢查點（R0 臂）
        if bad:  # 有這類檢查點
            raise ValueError(f"checkpoints without token statistics (R0 arm) cannot predict from tokens: {bad}")  # 它們不能從 token 預測
        # section 4M M14：檢查點需要哪些 token／R 變體（一般版，或含對手的 Human SL）
        need_x = {bool(side.get("opp_hsl")) for _, _, _, side in ckpts}  # 需要的 token 變體
        need_r = {bool(side.get("opp_hsl_r")) for _, _, _, side in ckpts}  # 需要的 R 變體
        opp_hsl_per_task, moves_opp_hsl = (ps.load_shards_opp_hsl(a.shards, a.nshards, with_moves=True) if (True in need_x or True in need_r) else (None, None))  # 有需要時才讀對手的 Human SL 分數
        sc = score_arrays(games, moves_hsl, moves_l1b, moves_opp, len(l1b_models)) if False in need_x else None  # 一般 token 的分數欄
        sc_oh = score_arrays(games, moves_hsl, moves_l1b, moves_opp, len(l1b_models), moves_opp_hsl) if True in need_x else None  # 含對手 Human SL 的分數欄
    except (FileNotFoundError, ValueError) as e:  # 分片、向量或檢查點有問題
        print(e, file=sys.stderr)  # 印出原因
        return EXIT_INPUTS  # 結束碼 2
    print(f"shards: {len(tasks)} games, L1b models {names} (using {[names[i] for i in use]}); vec: {len(games)} games, {n_plies} plies; "  # 回報分片、L1b 模型與向量
          f"{len(ckpts)} checkpoints ({Counter(s['arm'] for _, _, _, s in ckpts)}) on {device} ({time.time() - t0:.0f} s)", flush=True)  # 以及檢查點數、各臂數量與裝置
    vec = Path(a.vec)  # 向量資料夾
    by_q = {}  # 題目 → 它的任務
    for k, qi in enumerate(owner):  # 依任務順序
        by_q.setdefault(qi, []).append(k)  # 歸到所屬的題目
    qis = sorted(by_q)  # 有可用局的題目
    keys = [[task_key(k) for k in by_q[qi]] for qi in qis]  # 每題各局在向量中的 game_key
    task_games = {k: (sgf, c) for k, sgf, c in tasks}  # 任務 → (棋譜, 顏色)，計算讓子特徵用
    R_by_model, X_raw_by_model, offsets = {}, {}, None  # 各模型與變體的 R、token 與各局位置
    for m in use:  # 逐個使用的 L1b 模型
        for ohr in sorted(need_r):  # 逐個需要的 R 變體
            R_by_model[(m, ohr)] = question_r(qis, by_q, per_task, l1b_per_task, opp_per_task, task_games, m, opp_hsl_per_task if ohr else None)  # 計算 R 特徵
        for oh, arr in ((False, sc), (True, sc_oh)):  # 逐個 token 變體
            if arr is None:  # 不需要的變體
                continue  # 略過
            sc_path = vec / f"scores_{'oh_' if oh else ''}{names[m]}.npy"  # 分數陣列存在向量資料夾內
            np.save(sc_path, arr[m])  # 存檔，raw_tokens 從檔案讀取
            X_raw_by_model[(m, oh)], offsets = ts.raw_tokens([(vec, sc_path)])  # 組成未標準化的 token
    del sc, sc_oh  # 釋放分數陣列
    P, per_arm, n_pass, routing = ensemble(ckpts, R_by_model, keys, X_raw_by_model, offsets, device, model_names=names)  # 集成所有檢查點
    if n_pass == 0:  # 沒有任何檢查點接受這些 L1b 模型
        print(f"no checkpoint accepts any of the shards' L1b models {names} (sidecar l1b_heads)", file=sys.stderr)  # 說明原因
        return EXIT_INPUTS  # 結束碼 2
    print(f"{n_pass} passes ({len(ckpts)} checkpoints x {len(use)} L1b models) over {len(qis)} questions; size routing {routing} ({time.time() - t0:.0f} s)", flush=True)  # 回報推論趟數、題目數與分流
    fallback_rows = {}  # 退路提交檔的答案
    if a.fallback_submission:  # 有給主提交檔
        fb = pd.read_csv(a.fallback_submission, dtype=str, keep_default_na=False)  # 讀入主提交檔
        fallback_rows = dict(zip(fb["question_id"], fb["pred_rank"]))  # 題號 → 答案
    pos = {qi: j for j, qi in enumerate(qis)}  # 題目索引 → P 的列
    rows, probs, fallback, arm_rows = [], [], [], {arm: [] for arm in per_arm}  # 提交列、機率、退路題與各臂的答案
    for qi, q in enumerate(questions):  # 依輸入順序逐題
        j = pos.get(qi)  # 這一題在 P 中的列（沒有可用局時為 None）
        if j is not None and np.isfinite(P[j]).all():  # 有可用局，而且機率都是有限值
            rows.append((q["question_id"], LEVELS[expected_score_decision(P[j])]))  # 期望分數決策
            probs.append([float(v) for v in P[j]])  # 記下機率
            for arm in per_arm:  # 各臂自己的決策（報告用）
                arm_rows[arm].append(LEVELS[expected_score_decision(per_arm[arm][j])] if np.isfinite(per_arm[arm][j]).all() else None)  # 該臂沒有投票時為 None
        else:  # 無法決定的題目
            rows.append((q["question_id"], fallback_rows.get(q["question_id"], FALLBACK)))  # 用主提交檔的答案或 FALLBACK
            probs.append([float("nan")] * len(LEVELS))  # 機率記為 NaN
            fallback.append(q["question_id"])  # 列入退路題
            for arm in per_arm:  # 各臂也沒有答案
                arm_rows[arm].append(None)  # 記為 None
    out.parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    write_submission(rows, out)  # 寫出提交檔
    expect_prefix = {p: n for p, n in (("pub_", a.expect_public), ("priv_", a.expect_private)) if n > 0}  # Public／Private 題數的期望值
    problems = validate_submission(out, [q["question_id"] for q in questions], a.expect_rows or None, expect_prefix or None)  # 驗證提交檔
    pd.DataFrame(probs, columns=P_COLS).assign(question_id=[q["question_id"] for q in questions]).to_csv(out.with_name(out.stem + "_proba.csv"), index=False)  # 寫出每題的機率檔
    decided = [i for i, (qid, _) in enumerate(rows) if qid not in set(fallback)]  # 由模型決定的題目
    voted = {arm: [i for i in decided if arm_rows[arm][i] is not None] for arm in per_arm}  # 依題目大小分流的臂只對部分題目投票
    agree_arm = {arm: float(np.mean([arm_rows[arm][i] == rows[i][1] for i in voted[arm]])) if voted[arm] else None for arm in per_arm}  # 各臂與集成決策的一致率
    main_cmp = None  # 與主提交檔的比較
    if fallback_rows:  # 有主提交檔時
        same = [rows[i][1] == fallback_rows.get(rows[i][0]) for i in decided if rows[i][0] in fallback_rows]  # 逐題比較決策
        main_cmp = {"questions_compared": len(same), "same_decision_share": float(np.mean(same)) if same else None,  # 比較的題數與決策相同的比例
                    "differing": int(len(same) - sum(same))}  # 決策不同的題數
    code = (EXIT_INVALID if problems else EXIT_FALLBACK if fallback and not a.allow_fallback  # 結束碼：驗證失敗優先，其次是未接受的退路題
            else EXIT_INPUT_WARNINGS if input_warnings and not a.allow_input_warnings else EXIT_OK)  # 再其次是未接受的輸入警告，否則正常
    report = {"inputs": [str(p) for p in a.inputs], "shards": str(a.shards), "nshards": a.nshards, "questions_sha256": sha,  # 報告：輸入、分片與題目雜湊
              "shard_metas": [{k: m.get(k) for k in ("shard", "n_tasks", "positions", "opp_positions", "l1b_opp_moves", "hsl_opponent", "written")} for m in shard_metas],  # 各分片 meta 的重點欄位
              "l1b_models": l1b_models, "l1b_models_used": [names[i] for i in use], "vec": str(a.vec), "vec_plies": int(n_plies),  # L1b 模型與向量
              "vec_extract": [json.loads(p.read_text()) for p in sorted(vec.glob("extract_s*of*.json"))],  # 各分片 extract 的紀錄
              "checkpoints": [{"path": str(p), "state_sha256": side.get("state_sha256"), "arm": side.get("arm"), "tag": side.get("tag"), "seed": side.get("seed"),  # 每個檢查點：路徑、雜湊、臂、標籤與種子
                               "val_fold": side.get("val_fold"), "train_folds": side.get("train_folds"), "train_sizes": side.get("train_sizes"), "l1b_heads": side.get("l1b_heads"),  # 驗證折、訓練折、題目大小區間與等級頭
                               "opp_hsl": bool(side.get("opp_hsl")), "opp_hsl_r": bool(side.get("opp_hsl_r")),  # 使用的 token／R 變體
                               "best_inner": side.get("best_inner"), "git_head": side.get("git_head")}  # 內層留出集的最佳分數與程式版本
                              for p, (_, _, _, side) in zip(a.models, ckpts)],  # 依 --models 的順序
              "size_routing": routing, "arm_questions_voted": {arm: len(v) for arm, v in voted.items()},  # 分流與各臂投票的題數
              "passes": n_pass, "questions": len(questions), "games_declared": int(sum(q["declared_games"] for q in questions)),  # 推論趟數、題數與宣告的局數
              "games_present": int(sum(len(q["games"]) for q in questions)), "games_used": len(tasks), "games_dropped": dropped, "passes_in_games": passes,  # 讀到、使用與捨棄的局，以及棋譜中的 pass 數
              "handicap": hd_counts, "allow_missing_hd": a.allow_missing_hd,  # 讓子計數與設定
              "features": "fused_opp_diff_hd + per-ply tokens" + (" (+ opponent HSL: section 4M M14)" if True in need_x or True in need_r else ""),  # 使用的特徵
              "expectations": {"rows": a.expect_rows, "pub_": a.expect_public, "priv_": a.expect_private},  # 題數的期望值
              "fallback_submission": a.fallback_submission, "fallback_questions": fallback, "allow_fallback": a.allow_fallback,  # 退路設定與退路題
              "lenient_inputs": a.lenient_inputs, "input_warnings": input_warnings, "allow_input_warnings": a.allow_input_warnings,  # 寬鬆讀入的設定與警告
              "predicted_counts": dict(Counter(r for _, r in rows)), "arm_agreement_with_ensemble": agree_arm, "vs_fallback_submission": main_cmp,  # 各等級預測數、各臂一致率與主提交檔比較
              "device": device, "total_wall_s": time.time() - t0, "problems": problems, "exit_code": code}  # 裝置、總耗時、問題與結束碼
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float))  # 寫出報告 JSON
    print(f"wrote {out} ({len(rows)} rows) and {report_path}; problems: {problems or 'none'}; predicted {report['predicted_counts']}; "  # 回報輸出檔、驗證問題與預測分布
          f"arm agreement {agree_arm}" + (f"; vs main submission: {main_cmp}" if main_cmp else ""), flush=True)  # 以及各臂一致率與主提交檔比較
    if fallback:  # 有退路題
        print(f"WARNING: {len(fallback)} questions took {'the fallback submission' if fallback_rows else FALLBACK}: {fallback[:5]}"  # 警告哪些題用了退路答案
              + ("" if a.allow_fallback else f" (exit {EXIT_FALLBACK}; pass --allow-fallback to accept)"), file=sys.stderr, flush=True)  # 並提示 --allow-fallback
    if input_warnings and code == EXIT_INPUT_WARNINGS:  # 寬鬆讀入且未接受警告
        print(f"WARNING: {len(input_warnings)} questions were read leniently (exit {EXIT_INPUT_WARNINGS}; pass --allow-input-warnings to accept)",  # 警告寬鬆讀入的題數
              file=sys.stderr, flush=True)  # 輸出到 stderr
    return code  # 回傳結束碼


# ---------------------------------------------------------------- offline / compare 子命令

def offline(a):  # offline 子命令：訓練端路徑的參考機率
    """The checkpoints on a mock test's questions through the training-side tables of their fold (the reference for compare)."""
    from aicup.handicap import read_hd_table  # 讀取 HD 表
    from aicup.ladder import FOLDS_CSV  # 玩家分折表
    from aicup.pilot import pilot_questions  # 由玩家建立題目
    from aicup.scoring import fold_players, player_games, read_scored_games  # 選局與讀取評分結果
    import hd_check  # 訓練端的題目特徵（含讓子特徵）
    import l1b_cv  # 讀取一折的 Human SL 與 L1b 分數

    t0 = time.time()  # 計時起點
    f = a.fold  # 模擬測試集取自的折
    if f not in ts.FOLDS:  # 只允許 folds 1-4
        raise SystemExit(f"fold {f} is not one of {ts.FOLDS} (fold 0 is not used here)")  # fold 0 不在這裡使用
    mock = Path(a.mock_dir)  # 模擬測試集的資料夾
    ans = pd.read_csv(mock / "answers.csv", dtype=str, keep_default_na=False)  # 答案與來源題目
    players = fold_players(player_games(), pd.read_csv(FOLDS_CSV), f)  # 與 experiments/make_mock_test.py 相同的建構方式
    qs = pilot_questions(players).set_index("question_id")  # 這一折的 pilot 題目
    unknown = [q for q in ans["source_question"] if q not in qs.index]  # 對不上的模擬題目
    if unknown:  # 有對不上的
        raise SystemExit(f"{len(unknown)} mock questions are not fold {f}'s pilot questions, e.g. {unknown[:3]}")  # 停止
    sel = qs.loc[ans["source_question"]].reset_index()  # 依模擬測試集的順序選出題目
    hsl, l1b = l1b_cv.load_fold(ts.HSL, ROOT / "results/l1b" / ts.L1B[f], f)  # 這一折的 Human SL 與 L1b 分數表
    opp = read_scored_games(ROOT / "results/l1b" / ts.OPP[f] / f"fold{f}")  # 對手的 L1b 分數表
    hd_all = read_hd_table(ROOT / "results/data/hd_table.csv")  # 全部局的 HD 表
    X, y, n, _ = hd_check.features(hsl, l1b, opp, hd_all[hd_all["game_key"].isin(set(hsl["game_key"]))], sel)  # 訓練端的題目特徵、答案與局數
    R = X["fused_opp_diff_hd"]  # 64 維的 R 特徵
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 有 GPU 就用 GPU
    ckpts = [ts.load_checkpoint(p, device) for p in a.models]  # 載入檢查點
    R_by = {(0, False): R}  # 只有一個 L1b 來源（索引 0）
    if any(side.get("opp_hsl_r") for _, _, _, side in ckpts):  # section 4M M14：這一折訓練端的對手 HSL 表
        R_by[(0, True)] = np.hstack([R, opp_hsl_features(hsl, read_scored_games(ts.OPP_HSL / f"fold{f}"), sel)])  # 接上 39 個對手 HSL 特徵
    X_by, offsets = {}, None  # 各變體的 token 與各局位置
    for oh in sorted({bool(side.get("opp_hsl")) for _, _, _, side in ckpts}):  # 逐個需要的 token 變體
        X_by[(0, oh)], offsets = ts.raw_tokens([(ts.VEC / f"fold{f}", ts.scores_path(a.cache, f, opp_hsl=oh))])  # 訓練端的向量與分數陣列
    keys = sel["game_keys"].tolist()  # 每題各局的鍵
    print(f"{len(sel)} questions of fold {f}, {next(iter(X_by.values())).shape[0]} plies, {len(ckpts)} checkpoints on {device} ({time.time() - t0:.0f} s)", flush=True)  # 回報題數、手數、檢查點數與裝置
    P, per_arm, n_pass, _ = ensemble(ckpts, R_by, keys, X_by, offsets, device)  # 與測試端相同的集成
    out = Path(a.out)  # 機率檔路徑
    out.parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    pd.DataFrame(P, columns=P_COLS).assign(question_id=ans["question_id"].tolist()).to_csv(out, index=False)  # 寫出參考機率
    pred = np.array([expected_score_decision(p) for p in P])  # 期望分數決策
    official = balanced_mean_over_n([score(p, t) for p, t in zip(pred, y)], y, n)  # 逐題官方分數的類別平衡平均（本專案 CV 的主要指標）
    print(f"wrote {out}: {n_pass} passes; official score on these {len(sel)} fold-{f} questions {official:.4f} ({time.time() - t0:.0f} s)", flush=True)  # 回報推論趟數與上面的平衡分數
    return EXIT_OK  # 正常結束


def compare(a):  # compare 子命令：比較兩個機率檔
    """Two probability files (question_id + p_<level>) side by side; with --answers (rank, n_games) the official score of each."""
    A, B = (pd.read_csv(p, dtype={"question_id": str}).set_index("question_id") for p in (a.a, a.b))  # 讀入兩個機率檔
    ids = [q for q in A.index if q in B.index]  # 兩邊都有的題目
    if len(ids) != len(A) or len(ids) != len(B):  # 題目集合不同
        print(f"WARNING: {len(A)} / {len(B)} questions, {len(ids)} in both", flush=True)  # 警告題數不一致
    PA, PB = A.loc[ids, P_COLS].to_numpy(dtype=float), B.loc[ids, P_COLS].to_numpy(dtype=float)  # 對齊後的機率矩陣
    ok = np.isfinite(PA).all(axis=1) & np.isfinite(PB).all(axis=1)  # 兩邊都是有限值的題目
    d = np.abs(PA[ok] - PB[ok])  # 逐項的機率差
    da, db = np.array([expected_score_decision(p) for p in PA[ok]]), np.array([expected_score_decision(p) for p in PB[ok]])  # 兩邊各自的決策
    rep = {"a": str(a.a), "b": str(a.b), "questions": len(ids), "compared": int(ok.sum()), "max_abs_dp": float(d.max()) if ok.any() else None,  # 結果：檔名、題數、比較題數與最大差
           "mean_abs_dp": float(d.mean()) if ok.any() else None, "decisions_differ": int((da != db).sum()), "tolerance": a.tol,  # 平均差、決策不同的題數與容許值
           "pass": bool(ok.any() and d.max() <= a.tol)}  # 最大差不超過容許值才算通過
    if a.answers:  # 有答案時
        ans = pd.read_csv(a.answers, dtype={"question_id": str, "rank": str}).set_index("question_id").loc[ids]  # 依題目對齊的答案
        y = ans["rank"].map({lv: i for i, lv in enumerate(LEVELS)}).to_numpy(dtype=int)[ok]  # 正確等級的索引
        n = ans["n_games"].to_numpy(dtype=int)[ok]  # 每題局數（平衡平均依題目大小分組）
        for name, pred in (("score_a", da), ("score_b", db)):  # 兩個檔各算一次
            rep[name] = balanced_mean_over_n([score(p, t) for p, t in zip(pred, y)], y, n)  # 逐題官方分數的類別平衡平均
    print(json.dumps(rep, indent=1))  # 印出結果
    if a.report:  # compare 絕不寫 --out（提交檔的路徑）
        Path(a.report).write_text(json.dumps(rep, indent=1))  # 寫出比較結果
    return EXIT_OK if rep["pass"] else EXIT_INVALID  # 通過為 0，否則為 1


# ---------------------------------------------------------------- main 進入點

def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以模組說明的第一行作為描述
    ap.add_argument("cmd", choices=("extract", "predict", "offline", "compare"))  # 子命令
    ap.add_argument("--inputs", nargs="+", default=ps.DEFAULT_INPUTS, help="the test CSVs, public first (as predict_submission.py)")  # 測試 CSV（Public 在前）
    ap.add_argument("--lenient-inputs", action="store_true", help="as predict_submission.py: read a malformed question from its non-empty columns")  # 寬鬆讀入格式錯誤的題目
    ap.add_argument("--allow-input-warnings", action="store_true")  # 寬鬆讀入的警告不算失敗
    ap.add_argument("--vec", default=str(ROOT / "results" / "ct_submission" / "vec"), help="extract: where the parts go; predict: where they are")  # 向量分段檔的資料夾（extract 寫入、predict 讀取）
    ap.add_argument("--shard", type=int, default=0)  # extract 的分片編號
    ap.add_argument("--nshards", type=int, default=1, help="extract: shards of the games; predict: the main run's shard files")  # 分片數（extract：局切成幾份；predict：主流程分片檔的份數）
    ap.add_argument("--workers", type=int, default=10)  # 重播棋局的 CPU worker 數
    ap.add_argument("--part-size", type=int, default=100)  # 每個分段檔的局數
    ap.add_argument("--batch", type=int, default=256)  # 主幹推論的批次大小
    ap.add_argument("--shards", default=str(ROOT / "submission_rank.csv"), help="predict: the main run's --out, next to which its shard files are")  # 主流程的 --out，分片檔在它旁邊
    ap.add_argument("--models", nargs="*", default=[], help="predict / offline: the checkpoints (.pt + .json) of trunk_seq_check.py train")  # 要集成的檢查點
    ap.add_argument("--l1b-names", nargs="*", default=None, help="predict: only these shard L1b models feed tokens and R (default: all)")  # 只用這些 L1b 模型
    ap.add_argument("--out", default=str(ROOT / "submission_rank_ct.csv"))  # 第二份提交檔的路徑
    ap.add_argument("--report", default=None, help="predict: default <out stem>_report.json; compare: where to write its JSON (optional)")  # 報告路徑
    ap.add_argument("--fallback-submission", default=None, help="predict: the main submission; its row is used for a question this path cannot decide")  # 退路用的主提交檔
    ap.add_argument("--allow-fallback", action="store_true")  # 有退路題也以 0 結束
    ap.add_argument("--allow-missing-hd", action="store_true", help="predict: go on although some games carry no HD[] (their HD features are zeros)")  # 允許沒有 HD[] 的局
    ap.add_argument("--expect-rows", type=int, default=800)  # 預期的總題數
    ap.add_argument("--expect-public", type=int, default=400)  # 預期的 Public 題數
    ap.add_argument("--expect-private", type=int, default=400)  # 預期的 Private 題數
    ap.add_argument("--mock-dir", default=None, help="offline: a experiments/make_mock_test.py directory (answers.csv names the fold's pilot questions)")  # offline 的模擬測試集
    ap.add_argument("--fold", type=int, default=1, help="offline: the fold the mock test was built from")  # 模擬測試集取自的折
    ap.add_argument("--cache", default=str(ROOT / "results/trunk_seq"), help="offline: trunk_seq_check's score arrays")  # 訓練端分數陣列的位置
    ap.add_argument("--a", default=None, help="compare: a probability file")  # compare 的第一個機率檔
    ap.add_argument("--b", default=None, help="compare: the other")  # compare 的第二個機率檔
    ap.add_argument("--answers", default=None, help="compare: answers.csv of the mock test")  # 模擬測試集的答案
    ap.add_argument("--tol", type=float, default=1e-3, help="compare: the max |dp| that passes")  # compare 的容許誤差
    a = ap.parse_args(argv)  # 解析參數
    if a.nshards < 1 or not 0 <= a.shard < a.nshards:  # 分片參數不合理
        ap.error("--shard needs 0 <= shard < --nshards")  # 停止
    if a.cmd in ("predict", "offline") and not a.models:  # 需要檢查點的子命令
        ap.error(f"{a.cmd} needs --models")  # 沒給就停止
    if a.cmd == "offline" and not a.mock_dir:  # offline 需要模擬測試集
        ap.error("offline needs --mock-dir")  # 沒給就停止
    if a.cmd == "compare" and not (a.a and a.b):  # compare 需要兩個檔
        ap.error("compare needs --a and --b")  # 沒給就停止
    return {"extract": extract, "predict": predict, "offline": offline, "compare": compare}[a.cmd](a)  # 執行對應的子命令


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以子命令的回傳值作為結束碼

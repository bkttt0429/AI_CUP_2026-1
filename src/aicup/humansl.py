"""Reproduce KataGo's Human SL rank profiles and score a player's actual moves.

`make_profile_meta` mirrors `SGFMetadata::makeBasicRankProfile` in KataGo's
cpp/neuralnet/sgfmetadata.cpp so that Python evaluation uses exactly the
metadata the analysis engine uses for `humanSLProfile = rank_<r>`.
"""
import datetime  # 剖面中的對局日期

import numpy as np  # 數值陣列
import torch  # 網路推論
from katago.game.board import Board  # KataGo 的棋盤與顏色常數
from katago.game.features import Features  # 輸入特徵的編碼器
from katago.game.sgfmetadata import SGFMetadata  # Human SL 的 metadata
from katago.train.load_model import load_model  # 載入 KataGo 檢查點

from aicup.sgf import to_gtp  # 座標轉成 GTP 表示法

# pilot 評分用的 18 個 Human SL 等級剖面，由弱到強（索引 0 = 12k，17 = 6d）。
PROFILES = [f"{n}k" for n in range(12, 0, -1)] + [f"{n}d" for n in range(1, 7)]  # 12k ... 1k 接 1d ... 6d


def inverse_rank(s):  # KataGo 的反向等級編碼
    """KataGo's inverse rank: 9d -> 1 ... 1d -> 9, 1k -> 10 ... 20k -> 29."""
    s = s.lower()  # 不分大小寫
    n = int(s[:-1])  # 數字部分
    if s.endswith("d"):  # 段位
        return 10 - n  # 9d 為 1
    if s.endswith("k"):  # 級位
        return 9 + n  # 1k 為 10
    raise ValueError(f"unrecognised rank {s!r}")  # 無法辨識的等級


def make_profile_meta(black_rank, white_rank=None, preaz=False):  # 建立 rank_<r> 剖面的 metadata
    """Metadata for `rank_<b>` (both players) or `rank_<b>_<w>`; `preaz` picks the 2016 profile."""
    meta = SGFMetadata()  # 空的 metadata
    meta.inverseBRank = inverse_rank(black_rank)  # 黑方等級
    meta.inverseWRank = inverse_rank(white_rank or black_rank)  # 白方等級，未給時與黑方相同
    meta.bIsHuman = True  # 黑方為人類
    meta.wIsHuman = True  # 白方為人類
    meta.gameRatednessIsUnknown = True  # 對局是否列入等級評定（rated）設為未知
    meta.tcIsUnknown = False  # 用時制度已知
    meta.tcIsByoYomi = True  # 讀秒制
    meta.mainTimeSeconds = 1200.0  # 基本時間 1200 秒
    meta.periodTimeSeconds = 30.0  # 每次讀秒 30 秒
    meta.byoYomiPeriods = 5  # 讀秒 5 次
    meta.gameDate = datetime.date(2016, 9, 1) if preaz else datetime.date(2020, 3, 1)  # preaz 用 2016 年的對局日期，否則用 2020 年
    meta.source = SGFMetadata.SOURCE_KGS  # 來源為 KGS
    return meta  # 回傳 metadata


def profile_meta_for(own, opp, target_color, preaz=False):  # 依目標玩家的顏色擺放雙方等級
    """Profile metadata with the target player at rank `own` and the opponent at `opp`."""
    if target_color == "B":  # 目標玩家執黑
        return make_profile_meta(own, opp, preaz)  # 黑方用 own
    return make_profile_meta(opp, own, preaz)  # 目標玩家執白時白方用 own


def meta_rows(specs):  # 預先建立所有設定的 metadata 列
    """Metadata input rows for each (own, opp) spec: [S, 2, 192], second axis = target colour B/W.

    Row [s, c] is the row KataGo builds when the target (colour c) is the side to move, which
    is the only situation in which we score a move.
    """
    rows = np.zeros((len(specs), 2, 192), dtype=np.float32)  # [設定數, 2 種顏色, 192]
    for s, (own, opp) in enumerate(specs):  # 逐一設定
        for c, (color, pla) in enumerate((("B", Board.BLACK), ("W", Board.WHITE))):  # 目標玩家執黑與執白兩種
            rows[s, c] = profile_meta_for(own, opp, color).get_metadata_row(nextPlayer=pla, boardArea=361)  # 目標玩家輪到下時的列
    return rows  # 回傳所有列


def score_positions(model, bins, globs, legal, target, colors, rows, batch=64):  # 批次為目標落子評分
    """Log p of each target move under every metadata spec: [N, S] float32.

    Board features are uploaded once per batch and reused for all S specs; only the metadata
    row changes. FP32 throughout, illegal points masked before log-softmax, exactly as
    `_masked_policy_logp` (the path verified against the analysis engine).
    """
    device = next(model.parameters()).device  # 模型所在的裝置
    rows_t = torch.as_tensor(rows, dtype=torch.float32, device=device)  # metadata 列搬到裝置上
    n, s_count = len(target), rows.shape[0]  # 局面數與設定數
    out = np.empty((n, s_count), dtype=np.float32)  # 輸出 [N, S]
    with torch.no_grad():  # 只做推論
        for i in range(0, n, batch):  # 逐批次
            j = min(n, i + batch)  # 批次結尾
            b = torch.as_tensor(bins[i:j], device=device).float()  # 盤面平面
            g = torch.as_tensor(globs[i:j], device=device).float()  # 全域特徵
            illegal = ~torch.as_tensor(legal[i:j], device=device)  # 非法點
            tgt = torch.as_tensor(target[i:j], device=device).long()[:, None]  # 實際落點
            col = torch.as_tensor(colors[i:j], device=device).long()  # 目標玩家的顏色，用來選 metadata 列
            for s in range(s_count):  # 逐一設定，只換 metadata
                outputs = model(b, g, input_meta=rows_t[s, col])  # 前向計算
                logits = model.postprocess_output(outputs)[0][0][:, 0, :].float()  # 取 policy 第 0 通道
                logp = torch.log_softmax(logits.masked_fill(illegal, float("-inf")), dim=1)  # 遮罩非法點後的 log-softmax
                out[i:j, s] = logp.gather(1, tgt)[:, 0].cpu().numpy()  # 取實際落點的 log p
    return out  # 回傳 [N, S]


def load_humansl(ckpt_path, device, use_swa=False):  # 載入 Human SL 網路
    model, swa_model, _ = load_model(ckpt_path, use_swa, device=device, pos_len=19, verbose=False)  # 以 KataGo 的載入函式讀取檢查點
    model = swa_model if use_swa else model  # 依設定選用 SWA 權重
    model.eval()  # 推論模式
    return model, Features(model.config, 19)  # 回傳模型與 19 路的特徵編碼器


def stack_inputs(features, states, meta):  # 把多個局面疊成一個批次
    """Batch several GameStates into (bin [B,C,19,19], global [B,G], meta [B,192]) float32 tensors.

    The metadata row is built per state for its own side to move, so self/opponent ranks
    swap automatically between Black-to-move and White-to-move positions.
    """
    bins, globs, metas = [], [], []  # 三種輸入的收集清單
    for gs in states:  # 逐一局面
        b, g = gs.get_input_features(features)  # 盤面與全域特徵
        bins.append(b[0])  # 去掉批次維度
        globs.append(g[0])  # 去掉批次維度
        metas.append(meta.get_metadata_row(nextPlayer=gs.board.pla, boardArea=gs.board.x_size * gs.board.y_size))  # 依輪到的一方建立 metadata 列
    return (torch.tensor(np.stack(bins), dtype=torch.float32),  # 盤面張量
            torch.tensor(np.stack(globs), dtype=torch.float32),  # 全域特徵張量
            torch.tensor(np.stack(metas), dtype=torch.float32))  # metadata 張量


def _masked_policy_logp(model, features, gs, meta):  # 單一局面的 policy 對數機率
    """FP32 log-softmax over policy channel 0 for the side to move, illegal points at -inf."""
    board = gs.board  # 目前的盤面
    bin_in, glob_in = gs.get_input_features(features)  # 輸入特徵
    device = next(model.parameters()).device  # 模型所在的裝置
    meta_row = meta.get_metadata_row(nextPlayer=board.pla, boardArea=board.x_size * board.y_size)  # 輪到的一方的 metadata 列
    with torch.no_grad():  # 只做推論
        outputs = model(  # 前向計算
            torch.tensor(bin_in, dtype=torch.float32, device=device),  # 盤面平面
            torch.tensor(glob_in, dtype=torch.float32, device=device),  # 全域特徵
            input_meta=torch.tensor(meta_row, dtype=torch.float32, device=device).reshape(1, -1),  # metadata 列加上批次維度
        )  # 結束呼叫
        # postprocess_output -> 每個頭 -> 每個輸出張量；[0][0][0] 是樣本 0 的 policy_logits，
        # 形狀為 [num_policy_outputs, pos_len*pos_len + 1]，與 GameState 相同。
        logits = model.postprocess_output(outputs)[0][0][0][0].float()  # 取 policy 第 0 通道
    legal = torch.zeros(features.pass_pos + 1, dtype=torch.bool)  # 合法手遮罩（含 pass）
    for i in range(features.pass_pos):  # 逐一盤上點
        legal[i] = board.would_be_legal(board.pla, features.tensor_pos_to_loc(i, board))  # 依棋盤規則判斷是否合法
    legal[features.pass_pos] = True  # pass 永遠合法
    logits = logits.masked_fill(~legal.to(logits.device), float("-inf"))  # 非法點設為 -inf
    return torch.log_softmax(logits, dim=0).cpu().numpy()  # 回傳 362 維的 log p


def target_move_logps(model, features, moves, target_color, meta, rules=None, stones=()):  # 重播一局並為目標落子評分
    """Replay `moves`; before each move of `target_color`, score it under `meta`.

    Returns one dict per target move: t (index in `moves`), gtp, logp of the move actually
    played, and the full masked policy_logp vector (362) at that position. stones: Black setup
    stones (x, y) of a handicap game, placed before the moves (aicup.sgf.setup_and_moves).
    """
    from aicup.pilot import start_state  # 在函式內才匯入建立起始局面的函式
    gs = start_state(stones, rules)  # 含讓子石的起始局面
    out = []  # 每個目標落子一筆
    for t, (c, x, y) in enumerate(moves):  # 逐手重播
        pla = Board.BLACK if c == "B" else Board.WHITE  # 這一手的顏色
        loc = Board.PASS_LOC if x is None else gs.board.loc(x, y)  # 這一手的位置（pass 沒有座標）
        if c == target_color:  # 只為目標玩家的手評分
            if gs.board.pla != pla:  # 輪到的一方與紀錄不符
                raise ValueError(f"move {t}: {c} to play but board expects {gs.board.pla}")  # 拒絕
            logp = _masked_policy_logp(model, features, gs, meta)  # 落子前的 policy
            pos = features.pass_pos if x is None else features.loc_to_tensor_pos(loc, gs.board)  # 落點在 policy 中的索引
            out.append({"t": t, "gtp": "pass" if x is None else to_gtp(x, y),  # 手數索引與 GTP 座標
                        "logp": float(logp[pos]), "policy_logp": logp})  # 實際落子的 log p 與完整的 policy
        gs.play(pla, loc)  # 下這一手
    return out  # 回傳每個目標落子的結果

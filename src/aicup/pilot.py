"""Human SL pilot: CPU-side position extraction, player selection and pilot questions.

Workers replay a game once and keep, for every move of the target colour, the board
features, the legality mask and the index of the move actually played. The GPU side
(`aicup.humansl.score_positions`) then scores the same features under every profile.
"""
import os  # 讀取 AICUP_RULES 環境變數

import numpy as np  # 特徵陣列
from katago.game.board import Board  # KataGo 的盤面（判斷合法步）
from katago.game.gamestate import GameState  # KataGo 的對局狀態與網路輸入特徵

from aicup.sgf import handicap, parse_moves, setup_and_moves  # SGF 解析與讓子資訊

PASS_POS = 19 * 19  # 虛手在策略輸出中的索引（361）


def start_state(stones=(), rules=None):  # 建立開局狀態
    """A GameState at the start of a game: empty, Black to play; with `stones` (the Black setup stones (x, y) of a handicap
    game) placed outside the move history and White to play, as KataGo treats an SGF's AB[] stones."""
    gs = GameState(19, rules or GameState.RULES_CHINESE)  # 19 路棋盤，預設中國規則
    if stones:  # 讓子局有預置子
        for x, y in stones:  # 逐一放上黑子
            gs.board.set_stone(Board.BLACK, gs.board.loc(x, y))  # 直接放子，不記入著手歷史
        gs.board.set_pla(Board.WHITE)  # 讓子局由白方先下
        gs.boards = [gs.board.copy()]  # 盤面歷史從放好讓子石的局面開始
    return gs  # 開局狀態


def replay_rules(sgf):  # 重播時使用的規則
    """The rules the net is shown when replaying `sgf`: None (RULES_CHINESE, whiteKomi 7.5: production) unless the environment sets
    AICUP_RULES=jp (section 4M M27): RULES_JAPANESE with the server's komi, 6.5 in even games (HD0) and 0 in 讓先 / handicap games."""
    mode = os.environ.get("AICUP_RULES", "")  # 讀取規則切換
    if not mode:  # 沒有設定
        return None  # 沿用正式流程的中國規則
    if mode != "jp":  # 只認得 jp
        raise ValueError(f"AICUP_RULES={mode!r}: only 'jp' is known")  # 拒絕未知的設定
    r = dict(GameState.RULES_JAPANESE)  # 複製日本規則
    r["whiteKomi"] = 6.5 if (handicap(sgf) or 0) == 0 else 0.0  # 互先 6.5 目，讓先或讓子局 0 目
    return r  # M27 對照實驗用的規則


def game_positions(sgf, target_color, features, handicap=True):  # 擷取目標玩家每一手的網路輸入
    """Inputs for every move of `target_color`, taken from the position before that move. handicap: replay HD >= 2 records
    with their handicap stones on the board and the actual colours (aicup.sgf.setup_and_moves); False replays the record as
    written (the behaviour before dataset v1.1 made the handicap known)."""
    stones, moves = setup_and_moves(sgf) if handicap else ([], parse_moves(sgf))  # 讓子石與著手序列
    gs = start_state(stones, replay_rules(sgf))  # 開局狀態
    bins, globs, legal, target, ts = [], [], [], [], []  # 收集每手的輸入
    for t, (c, x, y) in enumerate(moves):  # 逐手重播
        pla = Board.BLACK if c == "B" else Board.WHITE  # 這一手的顏色
        loc = Board.PASS_LOC if x is None else gs.board.loc(x, y)  # 落點（虛手為 PASS_LOC）
        if c == target_color:  # 只記錄目標玩家的手
            board = gs.board  # 落子前的盤面
            b, g = gs.get_input_features(features)  # 網路的盤面與全域輸入
            b8 = b[0].astype(np.uint8)  # 二值盤面特徵壓成 uint8
            if not np.array_equal(b8, b[0]):  # 出現非 0/1 的值
                raise ValueError(f"move {t}: non-binary board feature, cannot pack as uint8")  # 無法無損壓縮，停止
            mask = np.zeros(PASS_POS + 1, dtype=bool)  # 合法步遮罩（含虛手）
            for i in range(PASS_POS):  # 逐一檢查盤上 361 點
                mask[i] = board.would_be_legal(pla, features.tensor_pos_to_loc(i, board))  # 是否為合法落點
            mask[PASS_POS] = True  # 虛手永遠合法
            bins.append(b8)  # 保存盤面特徵
            globs.append(g[0].astype(np.float32))  # 保存全域特徵
            legal.append(mask)  # 保存合法步遮罩
            target.append(PASS_POS if x is None else features.loc_to_tensor_pos(loc, board))  # 實際下出那一手的索引
            ts.append(t)  # 這一手在整局中的序號
        gs.play(pla, loc)  # 下這一手，前進到下一個局面
    n = len(ts)  # 目標玩家的手數
    c_shape = tuple(features.bin_input_shape)  # 盤面特徵的形狀（沒有手時建空陣列用）
    return {  # 每手輸入的陣列字典
        "bins": np.stack(bins) if n else np.zeros((0,) + c_shape, dtype=np.uint8),  # 盤面特徵
        "globs": np.stack(globs) if n else np.zeros((0, features.global_input_shape[0]), dtype=np.float32),  # 全域特徵
        "legal": np.stack(legal) if n else np.zeros((0, PASS_POS + 1), dtype=bool),  # 合法步遮罩
        "target": np.asarray(target, dtype=np.int16),  # 實際落點的索引
        "t": np.asarray(ts, dtype=np.int16),  # 手序
        "color": np.full(n, 0 if target_color == "B" else 1, dtype=np.int8),  # 目標顏色（黑 0、白 1）
    }  # 結束字典


def select_players(questions, fold, per_rank=20, seed=42):  # 挑選 pilot（前導實驗）的玩家
    """Up to `per_rank` players per rank that have a 20-game question in this fold's file.

    Returns DataFrame[rank, player_id, game_keys] where game_keys is the list of 20 keys
    from the player's `_n20_0` question. Ranks are processed in a fixed order and players
    sorted by id before sampling, so the result depends only on (questions, seed).
    """
    import pandas as pd  # 在函式內匯入：重播的部分用不到 pandas

    q = questions[questions["question_id"].str.startswith(f"f{fold}_") & questions["question_id"].str.endswith("_n20_0")]  # 這一折中每位玩家的第一個 20 局題目
    rng = np.random.RandomState(seed)  # 固定種子的抽樣器
    rows = []  # 收集選出的玩家
    for rank in sorted(q["rank"].unique(), key=str):  # 依固定順序處理各等級
        cand = q[q["rank"] == rank].sort_values("player_id")  # 該等級的玩家，依 id 排序
        take = cand.iloc[sorted(rng.choice(len(cand), size=min(per_rank, len(cand)), replace=False))]  # 不重複地抽出至多 per_rank 位
        for r in take.itertuples():  # 逐位選出的玩家
            rows.append({"rank": r.rank, "player_id": r.player_id, "game_keys": r.game_keys.split(";")})  # 棋局鍵由分號字串拆成清單
    return pd.DataFrame(rows, columns=["rank", "player_id", "game_keys"])  # 選出的玩家表


def pilot_questions(players):  # 由玩家切出巢狀的驗證題目
    """Per player: one 20-game question, two disjoint 10-game halves, four disjoint 5-game quarters."""
    import pandas as pd  # 在函式內匯入：重播的部分用不到 pandas

    rows = []  # 收集題目
    for p in players.itertuples():  # 逐位玩家
        keys = list(p.game_keys)  # 該玩家選出的棋局
        for n in (20, 10, 5):  # 三種題目大小
            for j in range(len(keys) // n):  # 不重疊地依序切題
                rows.append({"question_id": f"{p.rank}_{p.player_id[:8]}_n{n}_{j}", "rank": p.rank,  # 題號與等級
                             "player_id": p.player_id, "n_games": n, "game_keys": keys[j * n:(j + 1) * n]})  # 玩家、局數與連續的一段棋局
    return pd.DataFrame(rows, columns=["question_id", "rank", "player_id", "n_games", "game_keys"])  # 驗證用的題目表

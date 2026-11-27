"""Test-file reading, per-game sanity checks and submission writing/validation (layer L6).

Official test format (Tutorial README): question_id, num_games, sgf_1..sgf_20,
color_1..color_20, with only the first num_games sgf/color columns filled. Submission:
question_id, pred_rank, public questions then private, every question exactly once.

read_test refuses a file whose num_games disagrees with the filled sgf columns (2026-10-25, item 1 of
the to-do list): a question is only read when exactly its first num_games sgf columns are non-empty,
1 <= num_games <= MAX_GAMES. read_test_lenient reads such a question from its non-empty columns and
returns the problems as warnings instead -- the deadline-day escape hatch of predict_submission
--lenient-inputs. validate_submission checks the submission against the questions read:
header, every id once, the ids in the inputs' order (public file first), no private id before a public
one, the ranks in LEVELS and, when given, the expected total and per-prefix counts of the official
test (800 = 400 pub_ + 400 priv_), so a truncated or wrongly ordered input is caught before upload.
"""
from collections import Counter  # 找出重複的題號
from pathlib import Path  # 取出檔名

import pandas as pd  # 讀寫 CSV
from katago.game.board import Board  # KataGo 棋盤，用來重播並檢查合法性

from aicup.metric import LEVELS  # 10 個合法的等級
from aicup.pilot import start_state  # 建立擺好讓子石的起始局面
from aicup.sgf import board_size, handicap, setup_and_moves  # SGF 的棋盤大小、讓子數與實際顏色的著手


MAX_GAMES = 20  # 每題最多 20 局
PREFIXES = ("pub_", "priv_")  # 官方的 question_id 前綴；提交檔中 Public 題在 Private 題之前


def _read_test(path):  # 讀取測試檔並列出格式問題
    """([{question_id, declared_games, games: [(sgf, colour), ...]}], problems) of an official-format test
    CSV: every question read from its non-empty sgf columns, and one problem per question whose num_games
    is not in 1..MAX_GAMES or whose non-empty columns are not exactly the first num_games."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)  # 全部以字串讀入，空欄位保持為空字串（不轉成 NaN）
    out, bad = [], []  # 題目清單與問題清單
    for r in df.to_dict("records"):  # 逐題處理
        qid = r["question_id"]  # 題號
        raw = str(r["num_games"]).strip()  # 宣告的局數（原始文字）
        try:  # 局數可能不是數字
            declared = int(float(raw)) if raw else 0  # 轉成整數，空白視為 0
        except ValueError:  # 不是數字
            declared = -1  # 標成無效
        filled = [i for i in range(1, MAX_GAMES + 1) if str(r.get(f"sgf_{i}", "")).strip()]  # 非空 sgf 欄位的編號
        if not 1 <= declared <= MAX_GAMES:  # 局數超出範圍
            bad.append(f"{qid}: num_games {raw!r} is not in 1..{MAX_GAMES}")  # 記下問題
        elif filled != list(range(1, declared + 1)):  # 非空欄位不是恰好前 num_games 欄
            bad.append(f"{qid}: num_games {declared} but the non-empty sgf columns are {filled}")  # 記下問題
        games = [(str(r[f"sgf_{i}"]).strip(), str(r.get(f"color_{i}", "")).strip().upper()) for i in filled]  # 從非空欄位讀出棋譜與顏色
        out.append({"question_id": qid, "declared_games": max(declared, 0), "games": games})  # 加入這一題（局數不是數字或小於 0 時記為 0，其餘照宣告值）
    return out, bad  # 回傳題目與問題


def read_test(path, max_bad=5):  # 嚴格讀取測試檔
    """[{question_id, declared_games, games: [(sgf, colour), ...]}] of an official-format test CSV.

    Every question must declare 1 <= num_games <= MAX_GAMES and have exactly its first num_games sgf
    columns non-empty (the official format fills only those); otherwise ValueError names the file and
    up to max_bad offending questions, so a malformed file stops the pipeline before any scoring.
    Colours are read as given (upper-cased); check_game judges them."""
    out, bad = _read_test(path)  # 讀取並檢查格式
    if bad:  # 有格式錯誤的題目
        more = f" (and {len(bad) - max_bad} more)" if len(bad) > max_bad else ""  # 超出列舉數量時的提示
        raise ValueError(f"{path}: {len(bad)} malformed questions, e.g. " + "; ".join(bad[:max_bad]) + more)  # 在評分之前就中止流程
    return out  # 回傳題目清單


def read_test_lenient(path):  # 寬鬆讀取測試檔
    """(questions, warnings): read_test without the refusal. A malformed question is read from whatever sgf
    columns are non-empty (declared_games keeps the declared value, 0 when it is not a number) and described
    in `warnings`, one string per question prefixed with the file name; [] for a well-formed file, whose
    questions equal read_test's."""
    out, bad = _read_test(path)  # 讀取並檢查格式
    return out, [f"{Path(path).name}: {b}" for b in bad]  # 問題改成加上檔名的警告


def check_game(sgf, color):  # 檢查一局是否可用
    """(ok, info): a game is usable when it is 19x19, colours alternate starting with Black (with White, for a handicap
    game HD >= 2: its stones are placed and its marks swapped as in aicup.sgf.setup_and_moves, exactly as the scoring
    replays it), every move replays legally and the target colour has at least one move. info["handicap"] is the
    record's HD (None when absent)."""
    info = {"moves": 0, "passes": 0, "target_moves": 0, "handicap": handicap(sgf), "error": None}  # 檢查結果與統計
    if color not in ("B", "W"):  # 顏色不合法
        info["error"] = f"invalid colour {color!r}"  # 記下錯誤
        return False, info  # 不可用
    size = board_size(sgf)  # 棋盤大小
    if size != 19:  # 不是 19 路
        info["error"] = f"board size {size}"  # 記下錯誤
        return False, info  # 不可用
    try:  # 讓子數可能不在表中
        stones, moves = setup_and_moves(sgf)  # 讓子石與實際顏色的著手
    except ValueError as e:  # 不支援的讓子數
        info["error"] = str(e)  # 記下錯誤
        return False, info  # 不可用
    info["moves"] = len(moves)  # 總手數
    gs = start_state(stones)  # 擺好讓子石的起始局面
    expected = "W" if stones else "B"  # 讓子局由白方先下，否則黑方先下
    for t, (c, x, y) in enumerate(moves):  # 依序重播每一手
        if c != expected:  # 顏色沒有輪流
            info["error"] = f"move {t}: {c} played but {expected} to move"  # 記下錯誤
            return False, info  # 不可用
        expected = "W" if c == "B" else "B"  # 下一手換對方
        info["passes"] += x is None  # 累計虛手
        info["target_moves"] += c == color  # 累計目標顏色的手數
        try:  # 重播可能失敗
            gs.play(Board.BLACK if c == "B" else Board.WHITE, Board.PASS_LOC if x is None else gs.board.loc(x, y))  # 在棋盤上下這一手
        except Exception as e:  # IllegalMoveError 或其他重播無法處理的情況
            info["error"] = f"move {t}: {e}"  # 記下錯誤
            return False, info  # 不可用
    if info["target_moves"] == 0:  # 目標顏色一手都沒有下
        info["error"] = "no moves by the target colour"  # 記下錯誤
        return False, info  # 不可用
    return True, info  # 可用


def write_submission(rows, path):  # 寫出提交檔
    pd.DataFrame(rows, columns=["question_id", "pred_rank"]).to_csv(path, index=False)  # 依官方的兩個欄位寫成 CSV


def validate_submission(path, expected_ids, expected_total=None, expected_prefix_counts=None):  # 上傳前驗證提交檔
    """Problems with a submission file; an empty list means it is valid.

    expected_ids: the question ids of the inputs in the order read (public file first). Checked: the
    header; every expected id exactly once and nothing else; the rows in exactly that order; no priv_ row
    before a pub_ row; every id with a PREFIXES prefix; every pred_rank in LEVELS. expected_total (the
    official 800) and expected_prefix_counts ({"pub_": 400, "priv_": 400}) are independent expectations
    on the questions read, so a truncated input file is caught even though the file matches it."""
    expected_ids = list(expected_ids)  # 轉成清單以便比對順序
    df = pd.read_csv(path, dtype=str, keep_default_na=False)  # 以字串讀入提交檔
    if list(df.columns) != ["question_id", "pred_rank"]:  # 表頭不對
        return [f"header is {list(df.columns)}, expected ['question_id', 'pred_rank']"]  # 表頭錯誤時不再檢查其他項目
    problems = []  # 問題清單
    ids = df["question_id"].tolist()  # 提交檔的題號
    dups = sorted(i for i, c in Counter(ids).items() if c > 1)  # 重複的題號
    if dups:  # 有重複
        problems.append(f"duplicate ids: {len(dups)}, e.g. {dups[:3]}")  # 記下問題
    missing = sorted(set(expected_ids) - set(ids))  # 缺少的題號
    if missing:  # 有缺少
        problems.append(f"missing ids: {len(missing)}, e.g. {missing[:3]}")  # 記下問題
    unknown = sorted(set(ids) - set(expected_ids))  # 多出來的題號
    if unknown:  # 有多出來的
        problems.append(f"unknown ids: {len(unknown)}, e.g. {unknown[:3]}")  # 記下問題
    bad = sorted(set(df.loc[~df["pred_rank"].isin(LEVELS), "pred_rank"]))  # 不合法的等級值
    if bad:  # 有不合法的值
        problems.append(f"invalid rank values: {bad[:5]}")  # 記下問題
    if len(df) != len(expected_ids):  # 列數不符
        problems.append(f"{len(df)} rows, expected {len(expected_ids)}")  # 記下問題
    elif ids != expected_ids:  # 列數相同但順序不同
        k = next(i for i, (a, b) in enumerate(zip(ids, expected_ids)) if a != b)  # 第一個順序不同的列
        problems.append(f"question order differs from the inputs: row {k} is {ids[k]!r}, expected {expected_ids[k]!r}")  # 記下問題
    no_prefix = [i for i in ids if not i.startswith(PREFIXES)]  # 沒有官方前綴的題號
    if no_prefix:  # 有不合規的題號
        problems.append(f"ids without a {'/'.join(PREFIXES)} prefix: {len(no_prefix)}, e.g. {no_prefix[:3]}")  # 記下問題
    first_priv = next((k for k, i in enumerate(ids) if i.startswith("priv_")), None)  # 第一個 Private 題的列號
    last_pub = max((k for k, i in enumerate(ids) if i.startswith("pub_")), default=None)  # 最後一個 Public 題的列號
    if first_priv is not None and last_pub is not None and first_priv < last_pub:  # Private 題出現在 Public 題之前
        problems.append(f"a priv_ question (row {first_priv}) comes before a pub_ question (row {last_pub}): public must come first")  # 記下問題
    if expected_total is not None and len(expected_ids) != expected_total:  # 讀入的題數與官方總數不符
        problems.append(f"{len(expected_ids)} questions read from the inputs, expected {expected_total}")  # 記下問題（輸入檔可能被截斷）
    for prefix, want in (expected_prefix_counts or {}).items():  # 逐一核對各前綴的題數
        got = sum(i.startswith(prefix) for i in expected_ids)  # 讀入的該前綴題數
        if got != want:  # 與預期不符
            problems.append(f"{got} {prefix} questions read from the inputs, expected {want}")  # 記下問題
    return problems  # 空清單代表提交檔有效

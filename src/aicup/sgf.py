"""Parse the competition's single-line SGF strings and convert coordinates.

Coordinates are (x, y) with x the column 0-18 left to right and y the row
0-18 top to bottom, matching both SGF letters and KataGo's Board.loc(x, y).
"""
import re  # 以正規表示式解析 SGF

_MOVE_RE = re.compile(r";([BW])\[([a-s]{2}|tt)?\]")  # 一手棋：顏色與座標（空白或 tt 為虛手）
_SIZE_RE = re.compile(r"SZ\[(\d+)\]")  # 棋盤大小屬性
_HD_RE = re.compile(r"HD\[(\d+)\]")  # 讓子數屬性
_RE_RE = re.compile(r"RE\[([^\]]*)\]")  # 對局結果屬性
_PROP_RE = re.compile(r"(?<![A-Za-z])([A-Z]{1,2})\[")  # 任一屬性識別字（前面不可緊接字母）
GTP_COLS = "ABCDEFGHJKLMNOPQRST"  # GTP 的欄字母，跳過字母 I

# 資料集 v1.1（2026-10-31）在每個 sgf_content 加上 HD[n]（issue #42）。官方 Q&A（Q13）：HD[0] 是貼 6.5 目的互先局
# （約 80%），HD[1] 是「讓先」（不貼目，約 16%，顏色與紀錄相同），HD[2]-HD[9] 是讓子局（約 3%）：讓子石
# 沒有記錄（沒有 AB），而且著手的 B/W 標記是「顛倒」的（標成 B 的第一手其實是白方下的，也就是較強的
# 一方）。標準讓子點（SGF 座標），依子數列出：
HANDICAP_STONES = {2: ["pd", "dp"], 3: ["pd", "dp", "dd"], 4: ["pd", "dp", "dd", "pp"], 5: ["pd", "dp", "dd", "pp", "jj"],  # 2 到 5 子的讓子點
                   6: ["pd", "dp", "dd", "pp", "dj", "pj"], 7: ["pd", "dp", "dd", "pp", "dj", "pj", "jj"],  # 6、7 子的讓子點
                   8: ["pd", "dp", "dd", "pp", "dj", "pj", "jd", "jp"], 9: ["pd", "dp", "dd", "pp", "dj", "pj", "jd", "jp", "jj"]}  # 8、9 子的讓子點


def handicap(sgf):  # 讀取讓子數
    """HD[n] of the record, or None when the property is absent (dataset v1.0)."""
    m = _HD_RE.search(sgf)  # 尋找 HD 屬性
    return int(m.group(1)) if m else None  # 有就轉成整數


def result(sgf):  # 讀取對局結果
    """RE[...] of the record (e.g. W+R, B+3.5), or None when absent."""
    m = _RE_RE.search(sgf)  # 尋找 RE 屬性
    return m.group(1) if m else None  # 回傳結果字串


def properties(sgf):  # 列出紀錄中的屬性
    """The SGF property identifiers present in the record, in order of first appearance (e.g. ['SZ', 'HD', 'B', 'W'])."""
    seen, out = set(), []  # 已見過的集合與輸出清單
    for m in _PROP_RE.finditer(sgf):  # 逐一找出屬性
        k = m.group(1)  # 屬性識別字
        if k not in seen:  # 第一次出現
            seen.add(k)  # 記為已見過
            out.append(k)  # 依出現順序加入
    return out  # 回傳清單


def sgf_to_xy(coord):  # SGF 座標轉成 (x, y)
    """'pd' -> (15, 3): the (x, y) of an SGF coordinate."""
    return ord(coord[0]) - 97, ord(coord[1]) - 97  # 字母 a 對應 0


def parse_moves(sgf):  # 解析著手序列
    """Return [(color, x, y), ...]; a pass is (color, None, None). The colours are the record's marks: see setup_and_moves."""
    moves = []  # 著手清單
    for m in _MOVE_RE.finditer(sgf):  # 依序找出每一手
        color, coord = m.group(1), m.group(2)  # 顏色與座標
        if coord is None or coord == "tt":  # 空座標或 tt 代表虛手
            moves.append((color, None, None))  # 記為虛手
        else:  # 一般的著手
            moves.append((color, ord(coord[0]) - 97, ord(coord[1]) - 97))  # 轉成 (x, y)
    return moves  # 回傳著手清單


def other_colour(c):  # 對方的顏色
    return "W" if c == "B" else "B"  # 黑白互換


def setup_and_moves(sgf):  # 取得讓子石與實際顏色的著手
    """(stones, moves): how to replay one of the competition's records so that the colours are the ACTUAL ones.

    stones: the Black setup stones [(x, y), ...] of a handicap game (HD >= 2: the standard points of HANDICAP_STONES, which the
    record omits), else []. moves: parse_moves with the actual colours: for HD >= 2 every recorded mark is swapped (Q13: the
    record marks White's first move, the stronger side's, as B), otherwise unchanged (HD absent, i.e. dataset v1.0, HD[0] and
    HD[1] 讓先 are recorded as played). The `color` column of the data is the actual colour (issue #42's census), so the target
    colour of a game needs no change. ValueError for a handicap the table does not cover (HD >= 10)."""
    hd = handicap(sgf)  # 讓子數
    moves = parse_moves(sgf)  # 紀錄上的著手
    if hd is None or hd < 2:  # 不是讓子局（或資料集 v1.0）
        return [], moves  # 不擺子，顏色照紀錄
    if hd not in HANDICAP_STONES:  # 表中沒有的讓子數
        raise ValueError(f"unsupported handicap HD[{hd}]")  # 無法重播
    return [sgf_to_xy(c) for c in HANDICAP_STONES[hd]], [(other_colour(c), x, y) for c, x, y in moves]  # 擺上讓子石並把顏色換回實際的


def board_size(sgf):  # 讀取棋盤大小
    m = _SIZE_RE.search(sgf)  # 尋找 SZ 屬性
    return int(m.group(1)) if m else 19  # 沒有時預設 19 路


def to_gtp(x, y, size=19):  # (x, y) 轉成 GTP 座標
    return f"{GTP_COLS[x]}{size - y}"  # 欄字母加上由下往上數的列號


def from_gtp(s, size=19):  # GTP 座標轉成 (x, y)
    return GTP_COLS.index(s[0].upper()), size - int(s[1:])  # 欄字母轉 x，列號換回由上往下的 y

"""Parse the competition's single-line SGF strings and convert coordinates.

Coordinates are (x, y) with x the column 0-18 left to right and y the row
0-18 top to bottom, matching both SGF letters and KataGo's Board.loc(x, y).
"""
import re

_MOVE_RE = re.compile(r";([BW])\[([a-s]{2}|tt)?\]")
_SIZE_RE = re.compile(r"SZ\[(\d+)\]")
_HD_RE = re.compile(r"HD\[(\d+)\]")
_RE_RE = re.compile(r"RE\[([^\]]*)\]")
_PROP_RE = re.compile(r"(?<![A-Za-z])([A-Z]{1,2})\[")
GTP_COLS = "ABCDEFGHJKLMNOPQRST"  # GTP skips the letter I

# Dataset v1.1 (2026-10-31) adds HD[n] to every sgf_content (issue #42). The official Q&A (Q13): HD[0] even game with 6.5 komi
# (about 80%), HD[1] "讓先" (no komi, about 16%, colours as recorded), HD[2]-HD[9] handicap games (about 3%): the handicap
# stones are NOT recorded (no AB) and the B/W marks of the moves are REVERSED (the first move marked B was played by White,
# the stronger side). The standard handicap points, in SGF coordinates, by number of stones:
HANDICAP_STONES = {2: ["pd", "dp"], 3: ["pd", "dp", "dd"], 4: ["pd", "dp", "dd", "pp"], 5: ["pd", "dp", "dd", "pp", "jj"],
                   6: ["pd", "dp", "dd", "pp", "dj", "pj"], 7: ["pd", "dp", "dd", "pp", "dj", "pj", "jj"],
                   8: ["pd", "dp", "dd", "pp", "dj", "pj", "jd", "jp"], 9: ["pd", "dp", "dd", "pp", "dj", "pj", "jd", "jp", "jj"]}


def handicap(sgf):
    """HD[n] of the record, or None when the property is absent (dataset v1.0)."""
    m = _HD_RE.search(sgf)
    return int(m.group(1)) if m else None


def result(sgf):
    """RE[...] of the record (e.g. W+R, B+3.5), or None when absent."""
    m = _RE_RE.search(sgf)
    return m.group(1) if m else None


def properties(sgf):
    """The SGF property identifiers present in the record, in order of first appearance (e.g. ['SZ', 'HD', 'B', 'W'])."""
    seen, out = set(), []
    for m in _PROP_RE.finditer(sgf):
        k = m.group(1)
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def sgf_to_xy(coord):
    """'pd' -> (15, 3): the (x, y) of an SGF coordinate."""
    return ord(coord[0]) - 97, ord(coord[1]) - 97


def parse_moves(sgf):
    """Return [(color, x, y), ...]; a pass is (color, None, None)."""
    moves = []
    for m in _MOVE_RE.finditer(sgf):
        color, coord = m.group(1), m.group(2)
        if coord is None or coord == "tt":
            moves.append((color, None, None))
        else:
            moves.append((color, ord(coord[0]) - 97, ord(coord[1]) - 97))
    return moves


def board_size(sgf):
    m = _SIZE_RE.search(sgf)
    return int(m.group(1)) if m else 19


def to_gtp(x, y, size=19):
    return f"{GTP_COLS[x]}{size - y}"


def from_gtp(s, size=19):
    return GTP_COLS.index(s[0].upper()), size - int(s[1:])

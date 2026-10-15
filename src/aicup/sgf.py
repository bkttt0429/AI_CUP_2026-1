"""Parse the competition's single-line SGF strings and convert coordinates.

Coordinates are (x, y) with x the column 0-18 left to right and y the row
0-18 top to bottom, matching both SGF letters and KataGo's Board.loc(x, y).
"""
import re

_MOVE_RE = re.compile(r";([BW])\[([a-s]{2}|tt)?\]")
_SIZE_RE = re.compile(r"SZ\[(\d+)\]")
GTP_COLS = "ABCDEFGHJKLMNOPQRST"  # GTP skips the letter I


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

#!/usr/bin/env python
"""Section 4M M181 (2026-11-16): resume of the pilots w768b / ctl96 (m161q_pilots.py) from their checkpoints, on the data of
their first half. The first half ran while results/trunk_vec_x80 was missing (850,051 stored games, 846,807 trained on), so
the loader is restricted to the first two directories here; the trainer's own log must say "games 846807" again.

  sbatch -p 8gpus --gres=gpu:2 -c 24 ... run.sbatch python -I logs/nano4/m161q_pilots_resume.py --jobs w768b ctl96
"""
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m161q_pilots as mp  # noqa: E402

mp.base.DIRS = mp.base.DIRS[:2]
if __name__ == "__main__":
    assert all(d.name != "trunk_vec_x80" for d in mp.base.DIRS), mp.base.DIRS
    mp.main()

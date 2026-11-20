"""M90: the streaming loader (load_games) against the first version (load_games_v1) on the smoke subset, field by field."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path.cwd() / "logs/nano4")]
import m88_seqid as m  # noqa: E402

m.FULL = True; t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
G1, T1 = m.load_games_v1(True, log); G2, T2 = m.load_games(True, log); bad = 0
for name, A, B in (("train", G1, G2), ("test", T1, T2)):
    assert set(A) == set(B), (set(A), set(B))
    for k in A:
        same = A[k].shape == B[k].shape and A[k].dtype == B[k].dtype and np.array_equal(A[k], B[k]); bad += not same; print(f"{name} {k}: {A[k].shape} {A[k].dtype} {'same' if same else 'DIFFERENT ' + str(B[k].shape) + ' ' + str(B[k].dtype)}", flush=True)
print("LOADER CHECK " + ("PASSED" if not bad else "FAILED"), flush=True); sys.exit(1 if bad else 0)

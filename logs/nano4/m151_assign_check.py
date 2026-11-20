import sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd(); sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m86_bundle as b
s2 = b.s2
V, T, key_pid, used = b.sources(["SP2e~m15"]); z = V["SP2e~m15"]; t = T["SP2e~m15"]
ones = np.ones(len(z["y"]), bool); edges, cal, band, g = s2.fit(z, ones)
tb = np.where(~t["target_black"], np.searchsorted(edges[True], t["opp_m"]), np.searchsorted(edges[False], t["opp_m"])); tg = s2.group_of(t["hd"], t["target_black"])
print("validation rows", len(z["y"]), "test rows", len(t["hd"]))
for nm, hd, blk, bd in (("validation", z["hd"], z["target_black"], band), ("test", t["hd"], t["target_black"], tb)):
    print(f"\n{nm}: target is Black overall {blk.mean():.4f}; HD shares: " + " ".join(f"HD{h}: {(hd == h).mean():.4f}" for h in range(0, 5)) + f" HD>=5: {(hd >= 5).mean():.4f}")
    for h in (0, 1, 2):
        s = hd == h
        print(f"  HD{h}: target Black {blk[s].mean():.4f} (n {s.sum()}); by recognition band of the opponent (0-1 | 2-3 | 4 | >=5): " + " | ".join(f"{blk[s & m].mean():.4f} (n {int((s & m).sum())})" for m in (bd <= 1, (bd >= 2) & (bd <= 3), bd == 4, bd >= 5)))
    s = hd == 0
    print("  share of even rows by band: " + " ".join(f"{k}: {(bd[s] == k).mean():.4f}" for k in range(9)))
d = (z["opp_lab"] - z["y"])
for h, nm in ((0, "even"), (1, "HD1")):
    for col in (True, False):
        s = (z["hd"] == h) & (band >= 5) & (z["target_black"] == col)
        print(f"validation {nm}, target {'Black' if col else 'White'}, near-certain readings {int(s.sum())}: opponent label minus target label: mean {d[s].mean():+.3f}; share <0 / =0 / >0: {(d[s] < 0).mean():.3f} / {(d[s] == 0).mean():.3f} / {(d[s] > 0).mean():.3f}")
# by level: is the recognised opponent more often above or below?
for c in range(10):
    s = (z["hd"] == 0) & (band >= 5) & (z["y"] == c)
    if s.sum() > 30:
        print(f"  target level {c}: n {int(s.sum())}: below {(d[s] < 0).mean():.3f} same {(d[s] == 0).mean():.3f} above {(d[s] > 0).mean():.3f}")

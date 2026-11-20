"""Section 4M M50 follow-up (2026-11-07; user: Public ≈ train so Private ≈ train, hence ct10): how much does the labelled Public outcome
(ct10 265/117/18 vs ct9 260/122/18) say about ct10 - ct9 on Private, given Public ≈ Private? Report only.
  1. ct10 vs ct9 changed questions by half and direction.
  2. Likelihood of the observed Public difference under the M50 pseudo-test distributions (uniform / validation prior), normal approximation.
  3. Posterior for the per-question mean difference theta on this test population: flat prior, and a prior centred on the balanced value
     (sd 0.005), then P(ct10 > ct9 on Private) and the expected Private difference.
  python -I logs/nano4/m50_public_evidence.py
"""
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
E1 = math.exp(-1.0)
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


def main():
    d9 = pd.read_csv(ROOT / "submission_rank_ct9.csv").set_index("question_id")["pred_rank"]
    d10 = pd.read_csv(ROOT / "submission_rank_ct10.csv").set_index("question_id")["pred_rank"].reindex(d9.index)
    d12 = pd.read_csv(ROOT / "submission_rank_ct12.csv").set_index("question_id")["pred_rank"].reindex(d9.index)
    pub = d9.index.str.startswith("pub")
    for name, dx in (("ct10", d10), ("ct12", d12)):
        ch = dx != d9
        print(f"{name} vs ct9: {ch.sum()} of 800 differ (public {ch[pub].sum()}, private {ch[~pub].sum()})")
        for half, m in (("public", pub), ("private", ~pub)):
            dirs = pd.Series([f"{a}->{b}" for a, b in zip(d9[m & ch], dx[m & ch])]).value_counts()
            print(f"    {half}: " + ", ".join(f"{k} {v}" for k, v in dirs.items()))
    obs = 5 * (1 - E1) / 400
    print(f"observed Public ct10 - ct9: {obs:+.4f} (5 adjacent -> exact)")
    r = json.load(open(ROOT / "results/trunk_seq/es/m50_lp.json"))
    for s in ("012", "345"):
        pt = r[s]["pseudo_tests"]
        mu_u, sd_u = pt["uniform"]["ct10-ct9"]["mean"], pt["uniform"]["ct10-ct9"]["sd"]
        mu_v, sd_v = pt["validation"]["ct10-ct9"]["mean"], pt["validation"]["ct10-ct9"]["sd"]
        z_u, z_v = (obs - mu_u) / sd_u, (obs - mu_v) / sd_v
        lr = math.exp(-z_v ** 2 / 2) / math.exp(-z_u ** 2 / 2) * sd_u / sd_v
        p_u = 0.5 * math.erfc(z_u / math.sqrt(2)); p_v = 0.5 * math.erfc(z_v / math.sqrt(2))
        print(f"seeds {s}: under the uniform prior the Public result is {z_u:+.2f} sd (one-sided p {p_u:.3f}); under the validation prior {z_v:+.2f} sd (p {p_v:.3f}); "
              f"likelihood ratio validation : uniform = {lr:.1f}")
        # posterior for theta (the per-question mean on this population); Private mean ~ N(theta, sd_u)
        for prior_name, (m0, s0) in (("flat", (0.0, float("inf"))), ("balanced-centred N(mu_u, 0.005)", (mu_u, 0.005))):
            if math.isinf(s0):
                m_post, s_post = obs, sd_u
            else:
                w1, w0 = 1 / sd_u ** 2, 1 / s0 ** 2
                m_post, s_post = (obs * w1 + m0 * w0) / (w1 + w0), math.sqrt(1 / (w1 + w0))
            s_pred = math.sqrt(s_post ** 2 + sd_u ** 2)
            p_win = 0.5 * math.erfc(-m_post / s_pred / math.sqrt(2))
            print(f"    prior {prior_name}: theta {m_post:+.4f} ± {s_post:.4f}; Private ct10 - ct9 expected {m_post:+.4f} ± {s_pred:.4f}, P(> 0) {p_win:.2f}")


if __name__ == "__main__":
    main()

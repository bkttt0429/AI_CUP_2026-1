# Finance and credit-risk methods for keeping decision adjustments under noisy public evaluations, and for detecting and correcting grade lag at a threshold

Scope and notation. These notes translate quant-finance backtest-overfitting methods and credit-rating / PD-validation methods to the Go-rank setting. All numeric derivations in the *Inferences* blocks are mine, done by hand from the formulas cited, and are marked as derivations. Notation:

- Score per question: exact = 1, adjacent = e^-1 ≈ 0.368, else 0 (same convention as the sibling note `ordinal_decision_adjustment.md`). A 1D→A flip therefore gains g = 1 − e^-1 = 0.632 if the truth is A, and loses g if the truth is 1D. A true 2D loses 0.368 and a true B gains 0.368; these are ignored below unless stated.
- Consistency check: 8 correct flips × 0.632 / 400 = **+0.01264 public**. This exactly equals the observed public gain of the confirmed push (0.7748456 − 0.7622032 = 0.0126424), so the score model is right.
- A "probe" = one upload that changes n public predictions; the decoded score gives k = number of changed questions that became right. Success probability per flip = p. Edge per flip e = 2p − 1. Expected private gain of a push = n_priv × g × e / 400. **n_priv is known in advance**, because we know which private predictions change; only their outcomes are unknown.
- There are two relevant nulls:
  - **break-even null p = 0.5**: no score gain. This is the right null for the keep decision.
  - **model-calibrated null p0 ≈ 0.42**: under an r = 2 multiplier the flipped questions have model ratio P(A)/P(1D) ∈ [½, 1), so the model's own P(A | A∪1D) ∈ [⅓, ½). This is the right null for "is the model miscalibrated at this boundary".

---

## 1. Backtest overfitting and multiple testing (PBO/CSCV, Deflated Sharpe, MinBTL, Harvey–Liu–Zhu, White RC / Hansen SPA / Romano–Wolf): keep threshold and haircut for a push evaluated on ~5–15 public questions

### Takeaway
Each finance tool has an exact small-sample binomial analogue here.
- Bonferroni/Holm over the trials actually run, and the HLZ "t > 3" hurdle, give hard keep thresholds:
  - n = 8: 8/8 passes only if few trials were run.
  - n = 13: needs ≥ 11/13 (3 trials) or ≥ 12/13 (16 trials, or t > 3).
  - Below n ≈ 9 a probe can never clear t > 3.
- For a final-submission bet, an expected-value threshold under an explicit prior is the right rule, not family-wise error control. That rule keeps at much lower counts (≥ 8/13) and accepts that roughly 30–45% of keeps are null, because a false keep costs much less than a missed real edge.
- For the confirmed 8/8 push, every haircut considered keeps it positive. The expected private effect is +0.0045 to +0.0112 for 8 private flips, against +0.0126 observed on public. A planning value of about 50% of the public gain is the conservative end of that range.

### Cited Findings
- **PBO/CSCV.** Bailey, Borwein, López de Prado & Zhu define the probability of backtest overfitting (PBO) and estimate it with model-free, nonparametric "combinatorially symmetric cross-validation" (CSCV). PBO is the probability that the configuration chosen as best in-sample underperforms the median configuration out-of-sample. The authors argue that hold-out is unreliable for investment backtests. Published in J. Computational Finance (2017, DOI 10.21314/JCF.2016.322) — [SSRN 2326253](https://papers.ssrn.com/abstract=2326253); [WMU ScholarWorks](https://scholarworks.wmich.edu/math_pubs/42); [Risk.net](https://www.risk.net/journal-of-computational-finance/2471206/the-probability-of-backtest-overfitting).
- CSCV splits the sample into equal blocks and evaluates every symmetric assignment of blocks to in-sample and out-of-sample halves — [tradingstrategy.ai summary](https://tradingstrategy.ai/docs/learn/backtesting.html).
- The R package `pbo` implements CSCV with S partitions (examples use S = 8 and 16). It outputs PBO, performance degradation, probability of loss and stochastic dominance, and any performance measure can be used — [pbo README](https://archive.linux.duke.edu/cran/web/packages/pbo/readme/README.html).
- **Deflated Sharpe Ratio (DSR).** Bailey & López de Prado, J. Portfolio Management 40(5) 2014, pp. 94–107 (SSRN 2460551).
  - PSR(SR₀) = Φ[(SR̂ − SR₀)√(T−1) / √(1 − γ̂₃SR̂ + ((γ̂₄−1)/4)SR̂²)].
  - The DSR uses SR₀ = √Var{SR̂ₙ} · [(1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(Ne))], with γ ≈ 0.5772 (Euler–Mascheroni) and Var taken across the N trials.
  - Use un-annualised SR and Pearson kurtosis (Gaussian = 3).
  - Worked example: N = 100 gives an expected-maximum bracket of ≈ 2.531. A 1.50 annualised Sharpe over 60 months then has DSR ≈ 0.624, which does not reject at 95%.
  - Sources: [quantt.co.uk DSR walkthrough](https://www.quantt.co.uk/resources/deflated-sharpe-ratio); [López de Prado slides, Nomura 2015](https://www.nomura.com/events/9th-annual-global-quantitative-investment-strategies-conference/resources/upload/10_00_Marcos_Lopez_de_Prado_20150510.pdf).
- The same walkthrough says to count every configuration tried, abandoned ones included, and to report both the raw and the effective number of trials. Estimating the effective N for correlated trials "adds estimation uncertainty" — [quantt.co.uk](https://www.quantt.co.uk/resources/deflated-sharpe-ratio).
- **Minimum Backtest Length (MinBTL).** "Pseudo-Mathematics and Financial Charlatanism" (Bailey, Borwein, López de Prado, Zhu, 2013/2014) argues that a fixed sample can be fit to almost any target if enough configurations are tried. A minimum backtest length is therefore needed for a given number of trials. A commonly cited upper bound is ≈ 2 ln N / E[max SR]² (in years). MinBTL is a necessary but not sufficient condition.
  - One summary: trying 45+ configurations on 5-year samples almost guarantees an in-sample SR of 1.0 with an expected out-of-sample SR of 0.
  - Sources: [paper PDF, carmamaths.org](https://carmamaths.org/jon/backtest.pdf); [Kipnis lecture notes](https://cs.runi.ac.il/~kipnis/notes/STATS_207_L18.pdf); [CXO Advisory](https://www.cxoadvisory.com/big-ideas/insidiousness-of-overfitting-investment-strategies-via-iterative-backtests).
- **Harvey–Liu–Zhu (RFS 29(1), 2016, pp. 5–68).** The paper studies 316 factors from published and working papers. A newly discovered factor should clear t > 3.0 rather than 2.0, and the authors conclude that most claimed findings in financial economics are likely false. The framework allows correlated tests and missing data — [NBER w20592](https://nber.org/papers/w20592); [RePEc](https://econpapers.repec.org/RePEc:oup:rfinst:v:29:y:2016:i:1:p:5-68.); [CXO summary](https://www.cxoadvisory.com/big-ideas/taming-the-factor-zoo/).
- **Harvey & Liu, "Backtesting" (JPM, Fall 2015).** The haircut is the percentage difference between the reported Sharpe ratio and the multiple-testing-adjusted one. The adjustment uses Bonferroni, Holm or BHY, with HLZ's test count as an example. Holm and BHY need the empirical p-values of previously tried strategies.
  - The authors call the usual 50% haircut "a serious mistake". The haircut is nonlinear: the highest Sharpe ratios are only moderately penalised and marginal ones heavily.
  - Sources: [Duke PDF](https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF); [quantstrat `haircut.Sharpe` implementation](https://rdrr.io/github/braverock/quantstrat/src/R/haircut.Sharpe.R); [Man Group summary](https://www.man.com/insights/backtesting).
- **Empirical backtest-to-live decay.** Suhonen, Lennkh & Perez (JPM 2017) studied ~215 bank "alternative beta" strategies.
  - Median Sharpe was 1.20 in backtest and 0.31 live, a ~72–73% haircut (62% in a 3-year window around launch).
  - Only 18/215 did better live, and 65/215 had a negative live Sharpe.
  - The most complex strategies took a haircut more than 30% larger.
  - One summary concludes that the industry 50% derating "is not conservative enough".
  - Sources: [Alpha Architect](https://alphaarchitect.com/?p=36892); [MathInvestor](https://mathinvestor.org/2017/08/backtest-overfitting-in-smart-beta-investments/); [CXO Advisory](https://www.cxoadvisory.com/big-ideas/live-performance-of-alternative-beta-products/).
- McLean & Pontiff (JF 2016) study the out-of-sample and post-publication predictability of 97 anomaly variables. A working draft reports a decline of ~32%, and press coverage cites ~35% post-publication decay — [Ivey working paper](https://ivey.uwo.ca/media/3775549/pontiff.pdf); [Fortune 2012](https://fortune.com/2012/11/02/academics-academia-destroys-stock-picking-strategies/).
- **White's Reality Check (2000), Romano–Wolf StepM (Econometrica 2005), Hansen SPA (2005).**
  - The bootstrap Reality Check asks only whether the best strategy beats the benchmark, using the bootstrapped distribution of the maximum statistic.
  - Romano–Wolf's stepwise procedure (first step = modified Reality Check) finds all strategies that beat the benchmark while asymptotically controlling the family-wise error rate.
  - Hansen's SPA is more powerful and less sensitive to poor and irrelevant alternatives than the Reality Check. A stepwise SPA (Hsu & Hsu) is more powerful than StepM.
  - Python `arch` implements SPA, StepM and the Hansen–Lunde–Nason Model Confidence Set.
  - Sources: [Romano & Wolf working paper (UPF 712)](https://econ-papers.upf.edu/papers/712.pdf); [arch docs](https://arch.readthedocs.io/en/latest/multiple-comparison/multiple-comparison-reference.html); [stepwise SPA (HKU)](https://hub.hku.hk/bitstream/10722/141768/1/content.pdf?accept=1).
- **Winner's curse.** Andrews, Kitagawa & McCloskey, "Inference on Winners" (QJE 139(1), 2024): when the target is chosen by optimising over the data, conventional estimates are biased and conventional CIs unreliable.
  - They give conditional (median-unbiased) estimators and CIs, plus hybrid ones.
  - Their hybrid CIs are shorter than projection CIs everywhere, and shorter than conditional CIs except in the well-separated case.
  - Sources: [NBER w25456 / RePEc](https://econpapers.repec.org/RePEc:nbr:nberwo:25456); [CeMMAP WP](https://discovery.ucl.ac.uk/10087224/7/Kitagawa_CWP311818.pdf).

### Inferences
(All derivations below are mine. Binomial counts are exact; z-scores use the normal approximation and are crude at n ≤ 15.)

**A. Exact evidence of the 8/8 push.**
- P(8/8 | p = 0.5) = 1/256 = 0.0039.
- P(8/8 | p0 = 0.42) = 0.42⁸ ≈ 0.00097.
- z under p = 0.5 is (8 − 4)/√2 = 2.83. Under p0 = 0.42 it is (8 − 3.36)/1.396 = 3.32.
- If 16 null candidates were each probed with 8 flips, P(at least one 8/8 by luck) = 1 − (255/256)¹⁶ ≈ 0.061 at p = 0.5, and ≈ 0.010 at p = 0.40.

**B. Keep thresholds (minimum k out of n public flips).** N = number of probes actually evaluated on public. Under DSR/PBO logic, untested candidates do not count unless public information was used to choose among them.

| n flips | single test α=0.05 (p=0.5) | Bonferroni N=3 (α=0.0167) | Bonferroni N=16 (α=0.0031) | HLZ "t>3" (z≥3, p=0.5) | EV rule, weak prior (see F) |
|---|---|---|---|---|---|
| 5 | 5/5 (p=0.031) | impossible | impossible | impossible (max z 2.24) | ≥4/5 |
| 8 | ≥7/8 (0.035) | 8/8 (0.0039) | impossible (0.0039>0.0031) | impossible (max z 2.83) | ≥5/8 |
| 10 | ≥9/10 (0.011) | ≥9/10 | 10/10 (0.00098) | 10/10 (z=3.16) | ≥7/10 |
| 13 | ≥10/13 (0.046) | ≥11/13 (0.011) | ≥12/13 (0.0017) | ≥12/13 (z=3.05) | ≥8/13 |
| 15 | ≥12/15 (0.018) | ≥13/15 (0.0037) | ≥14/15 (0.0005) | ≥14/15 (z=3.36) | ≥10/15 |

- A t > 3 hurdle needs n ≥ 9 (9/9 gives z = 3.0 exactly).
- The 8/8 push passes Bonferroni if only the 3 actually tested probes count. It just fails if all 16 candidates count. It passes either way under the model-calibrated null p0 = 0.42 (0.00097 × 16 = 0.016).
- Which N applies depends on whether the 1D→A idea was picked after looking at public feedback, for example from the earlier composition-bet upload on the same boundary. If it was, N is larger and those overlapping public questions are not fresh evidence.

**C. MinBTL analogue: minimum probe size.** Under p = 0.5 the per-flip edge estimate (2p̂−1) has SD 1/√n. The expected best apparent edge among N skill-less candidates is ≈ E[max_N]/√n. To claim an edge e* you need n > (E[max_N]/e*)². The 2 ln N/e*² form from the MinBTL bound is the looser upper bound.

E[max_N] from the DSR bracket:

| N | E[max_N] |
|---|---|
| 3 | 0.853 |
| 9 | 1.52 |
| 13 | 1.70 |
| 16 | 1.80 |
| 100 | 2.53 |

Minimum probe size n:

| N | e* = 0.5 (p = 0.75) | e* = 0.3 (p = 0.65) |
|---|---|---|
| 3 | 2.9 (bound 8.8) | 8.1 (bound 24) |
| 16 | 13.0 (bound 22) | 36 (bound 62) |

So if 16 candidates are probed with 13 public questions each, the luckiest null candidate looks, in expectation, like a p ≈ 0.75 push. Probes of 5–13 questions cannot separate modest edges (p ≈ 0.65) from noise in a family that large.

**D. DSR-style deflation of the 8/8.** With no true edge, the cross-candidate dispersion equals the sampling SE, so DSR reduces to Φ(z − E[max_N]).

| Null | N = 3 | N = 16 |
|---|---|---|
| p = 0.5 (z = 2.83) | Φ(1.98) = 0.976 | Φ(1.03) = 0.849 |
| p0 = 0.42 (z = 3.32) | Φ(2.47) = 0.993 | Φ(1.52) = 0.936 |

- The 8/8 push clears the usual DSR 0.95 bar only when N is small or the null is model-calibrated.
- A literal DSR on per-flip "returns" breaks down here: a perfect 8/8 record has zero variance, so the Sharpe is infinite. The binomial forms above are the correct small-sample versions.

**E. Haircut of the 8/8 push's expected private effect** (assuming n_priv = 8; scale linearly for the true n_priv; observed public = +0.0126):

| Method | Implied p (or edge factor) | Haircut | Expected private gain |
|---|---|---|---|
| Naive (p̂ = 1) | 1.00 | 0% | +0.0126 |
| Jeffreys posterior Beta(8.5, 0.5) mean | 0.944 | 11% | +0.0112 |
| Uniform prior, posterior Beta(9,1) mean | 0.900 | 20% | +0.0101 |
| Harvey–Liu Bonferroni, N=3 (p 0.0117 → z 2.27) | edge × 2.27/2.83 | 20% | +0.0101 |
| DSR-style z − E[max], N=3 (2.83 → 1.98) | edge × 0.70 | 30% | +0.0089 |
| Harvey–Liu Bonferroni, N=16 (p 0.0625 → z 1.53) | edge × 0.54 | 46% | +0.0068 |
| Two-point model (π=0.2 real; p1=0.75 vs p0=0.40) → posterior π=0.974 | — | 52% | +0.0061 |
| Pluto–Tasche-type 95% lower bound p ≥ 0.05^(1/8) = 0.688 (Section 3) | 0.688 | 62% | +0.0048 |
| DSR-style, N=16 (2.83 → 1.03) | edge × 0.36 | 64% | +0.0046 |
| Skeptical prior Beta(4.2, 5.8) (= model-calibrated 0.42 worth 10 questions) → Beta(12.2, 5.8) | 0.678 | 64% | +0.0045 |

- The industry "halve it" rule (+0.0063) is equivalent to a model-calibrated prior worth ≈ 6 questions: (0.42·n₀ + 8)/(n₀ + 8) = 0.75 gives n₀ ≈ 6.
- Suhonen's 62–73% live haircut applies to complex products with heavy search. Our push is a single-parameter, mechanism-motivated adjustment, so the lower end is more plausible. This is a judgement, not a sourced fact.

**F. False-keep probability and the EV keep rule** (the core answer for the 13 weak-prior candidates).
- Model: a candidate is real with prior π (p1 = 0.75, per-flip gain +0.316) or null (p0 = 0.40, per-flip loss −0.126).
- Keep iff posterior P(real) > 0.126/(0.126 + 0.316) = 0.285.
- With π = 0.2, the prior EV is negative (0.2·0.316 − 0.8·0.126 = −0.038 per flip), so an untested candidate should not be kept.

| Probe size | EV rule | Power (true p = 0.75) | False-pass rate (null p = 0.40) | P(null \| kept) | EV per private flip |
|---|---|---|---|---|---|
| n = 6 | k ≥ 4 | 0.83 | 0.18 | **0.46** | +0.034 |
| n = 8 | k ≥ 5 | 0.89 | 0.17 | **0.44** | — |
| n = 13 | k ≥ 8 | 0.92 | 0.098 | **0.30** | +0.048 |
| n = 13, stricter | k ≥ 10 | 0.58 | 0.008 | 0.05 | +0.036 (lower) |
| n = 13, π = 0.1 | k ≥ 9 | 0.79 | 0.032 | 0.27 | — |

- Lesson: family-wise error control (HLZ / Bonferroni / Romano–Wolf) is the right tool for *claims* ("there is a rating lag at boundary b"). For the *bet* (include in the final file) the EV rule is better even though about 30% of kept 13-question pushes are null. A false keep costs ≈ 0.4× what a real keep gains.
- This is the same asymmetry Harvey & Liu point to when they reject a flat 50% haircut.
- The EV rule is only as good as π and p1. Estimate them by empirical Bayes from the tested family:
  - 1 of 3 tested pushes was real. Those were the most promising, so for the remaining 13 "weak-prior" candidates, π ≈ 0.1–0.2 is the realistic range.

**G. Expected value of information (EVOI) of one probe** (same model, n_pub = n_priv):

EVOI = E_k[max(0, E[private gain | k])] − max(0, prior EV)

| Prior π | Probe size | EVOI (private score) |
|---|---|---|
| 0.2 | n = 6 | ≈ +0.0005 |
| 0.2 | n = 13 | ≈ +0.0016 |
| 0.1 | n = 13 | ≈ +0.0007 |

- Each upload costs one slot, and the final file must be re-uploaded last. Probe only candidates with n ≥ ~10 public flips and a stated mechanism.

**H. Selecting several pushes (Romano–Wolf / Holm) and the winner's curse.**
- Probes on disjoint questions are independent exact binomials, so Holm's step-down (reject p₍j₎ ≤ α/(m − j + 1)) is the exact small-sample counterpart of StepM.
- Nested pushes (r = 1.5 ⊂ r = 2) share questions. The overlap is known exactly, so the joint null distribution can be enumerated instead of bootstrapped.
- SPA's "poor alternatives" point applies directly: the two failed pushes should not dilute the Reality-Check-style p-value of the winner.
- AKM's finding that conditional and conventional inference coincide in the "well-separated" case suggests a small selection correction for the 8/8 push, since it is well separated from the two failures. Its main haircut comes from finite-sample regression (8/8 does not mean p = 1) and from the prior, not from winner's-curse selection among 3.

**I. Where PBO/CSCV fits.**
- CSCV needs a full T × N performance matrix. On the public set we observe outcomes only for changed questions, so the matrix is mostly missing.
- PBO is usable on the 33k validation out-of-fold posteriors for choosing push *strength* (r grid) or boundary sets. That only measures in-distribution overfitting, not the test-specific lag.
- Applicability: low to medium, at zero upload cost.

**Applicability summary.**
- Exact-binomial Bonferroni/Holm thresholds and the MinBTL-style probe-size rule: high value, zero cost.
- EV/Bayes keep rule: high value, risk = misspecified π and p1.
- DSR: medium (needs a variance, degenerate at perfect records).
- PBO/CSCV: low for public probes, medium for tuning r on validation.

### Gaps
- I could not retrieve full texts of the DSR, PBO, MinBTL or Harvey–Liu PDFs (binary only). The formulas above come from the secondary walkthrough and summaries cited.
  - The CSCV logit step is the paper's standard construction but was not verified this session: λ = ln[ω̄/(1−ω̄)], with ω̄ the relative out-of-sample rank of the in-sample winner, and PBO = share of splits with λ ≤ 0.
  - The exact MinBTL constant was also not verified.
- McLean & Pontiff's published headline figures (often quoted as 26% lower out-of-sample and 58% lower post-publication) were not confirmed by the retrieved sources. Only ~32% (draft) and ~35% (press) were confirmed.
- No source gives finance-style multiple-testing guidance for sample sizes this small (5–15 Bernoulli outcomes). The binomial translations are my derivations.
- I do not know the counts of the two failed pushes or the n_priv of each candidate. The numbers above use n_priv = 8 and an illustrative two-point prior.

---

## 2. Credit-rating analogues of grade lag: through-the-cycle vs point-in-time, rating momentum, market-implied vs agency ratings, and the investment-grade cliff

### Takeaway
The finance literature documents exactly the mechanism hypothesised for the kyu→dan gate:
- Agency ratings move only after the point-in-time credit quality has drifted past a threshold (≈ 1.8 notches), and then only partially.
- Upgrades lag more than downgrades (¾ year vs ½ year).
- Gaps between market-implied and agency ratings carry predictive information, more so for opaque or volatile issuers.
- Cliff thresholds (BBB−/BB+) carry discrete consequences that distort behaviour around them.

The translation predicts a directional, gate-specific lag (model says 1D, label says A), consistent with the 8/8 result. It also predicts that most other model-vs-label gaps are model noise ("the market does most of the adjusting"), which matches the two failed pushes.

### Cited Findings
- Altman & Rijken: agencies use a "through-the-cycle" (TTC) method to suppress rating sensitivity to short-term credit fluctuations, while investors take a "point-in-time" (PIT) view. A conservative migration policy is a second source of stability, and both explain the perceived rigidity of agency ratings — [Altman & Rijken, NYU archive 2451/26557](https://archive.nyu.edu/handle/2451/26557); [2451/26405](https://archive.nyu.edu/handle/2451/26405).
- In their agency-rating prediction model, a migration is triggered only when the predicted rating departs from the actual one by at least a threshold of **1.8 notch steps**. When triggered, the rating is adjusted only partly: **70% on the downside, 60% on the upside** — [Altman & Rijken working papers, NYU archive](https://archive.nyu.edu/handle/2451/26559). These figures come from search abstracts of the 2003–2006 Altman–Rijken series; which of the papers contains them was not verified.
- TTC procedures delay rating migration on average by **½ year for downgrades and ¾ year for upgrades**, and hurt one-year default prediction. Rating stability comes primarily from the prudent migration policy rather than the focus on permanent credit risk — [Altman & Rijken, FAJ 62(1) 2006](https://econpapers.repec.org/RePEc:taf:ufajxx:v:62:y:2006:i:1:p:54-70).
- Moody's Market Implied Ratings (MIR): changes in MIR usually lead changes in Moody's ratings, which is why MIR is positioned as an early alert — [Moody's MIR methodology](https://www.moodys.com/sites/products/ProductAttachments/MIR_Methodology.pdf).
- A Moody's study found that gaps between credit-implied trading levels and ratings tend to narrow over a year, with **the market doing most of the adjusting**. When Moody's moves toward the market it is more often a downgrade than an upgrade. Bond-price gaps were a better leading indicator of downgrades than CDS — [GlobalCapital](https://www.globalcapital.com/securitization/article/28mww5lmiy5jx6hb2zaww/moodys-rebuffs-criticism-it-lags-the-market).
- Hasan, Shen, Zhang & Poon (J. Financial Research 2023): rating gaps (market-implied minus agency) add information on default risk beyond ratings, short-term and over longer horizons. The predictive power is greater for **more opaque and more volatile issuers**, was stronger in 2008, and weakened after Dodd–Frank. Some gap signals precede negative outlooks and watchlists — [PolyU repository](https://ira.lib.polyu.edu.hk/handle/10397/99688?mode=full); [IWH Halle](https://iwh-halle.de/en/publications/detail/market-implied-ratings-and-their-divergence-from-credit-ratings).
- Rating momentum: Altman & Kao (1992) are the original reference for rating drift. Lando & Skødeberg (JBF 2002) found significant non-Markov effects (duration dependence and dependence on the previous rating), mainly for downgrades. A later model puts recently downgraded firms in an "excited" state and finds that non-Markov models give higher PDs for investment-grade issuers — [Bundesbank/CBS confidence-set paper](https://research-api.cbs.dk/ws/files/59164378/Confidence_sets_for_continuous_time_rating.pdf); [arXiv 1809.09889](https://arxiv.org/pdf/1809.09889.pdf); [Quant. Finance 2020](https://informahealthcare.com/doi/abs/10.1080/14697688.2020.1726439). Fledelius, Lando & Nielsen (2004) estimated non-parametrically how transition intensities depend on duration and previous state — [JOIM 2(2) 2004](https://joim.com/wp-content/uploads/emember/downloads/a0027.pdf).
- **Investment-grade cliff.** Insurers often cap or ban high-yield holdings, which creates a "rating cliff" at BBB→BB and forced sales. Solvency II capital charges add pressure.
  - Insurers selling fallen angels face substantial price-pressure effects (Ambrose, Cai & Helwege).
  - The ESRB severe scenario puts forced fallen-angel sales at up to €373 bn.
  - BlackRock counters that insurers pre-empt downgrades, so broad forced selling is more muted.
  - Sources: [AllianceBernstein](https://insuranceaum.com/for-insurers-the-fallen-angel-wave-is-a-call-for-flexibility); [Bank Underground](https://bankunderground.co.uk/2019/05/22/what-happens-when-angels-fall/); [NY Fed conference paper](https://www.newyorkfed.org/medialibrary/media/research/conference/2010/cb/Ambrose_Cai_Helwege1.pdf); [ESRB](https://www.esrb.europa.eu/pub/pdf/A_system-wide_scenario_analysis_of_large-scale_corporate_bond_downgrades.en.pdf); [BlackRock](https://BlackRock.com/corporate/literature/whitepaper/policy-spotlight-lessons-from-covid-19-european-bbb-bonds-and-fallen-angels-july-2020.pdf).
- Kisgen (JF 61(3) 2006): capital-structure behaviour fits discrete costs and benefits attached to rating changes. Firms near a rating change (the "plus or minus" notch test) adjust their leverage — [RePEc](https://ideas.repec.org/a/bla/jfinan/v61y2006i3p1035-1072.html); [secondary description of the plus/minus test](https://repositorio.ucp.pt/server/api/core/bitstreams/92250560-cb93-419c-a475-577e7125eca9/content).
- Practitioners analyse rating transition matrices specifically around rating cliffs (title-level only; content not retrieved) — [J.P. Morgan AM, "Peering over the ratings cliff"](https://am.jpmorgan.com/se/en/asset-management/liq/insights/portfolio-insights/fixed-income/fixed-income-perspectives/leveraging-rating-transition-matrices-around-rating-cliffs).

### Inferences
- **Mapping.** Official Go grade = agency rating (TTC, sticky, moves only past a threshold and only partly). Model grade from move features = market-implied rating (PIT). The quantity to model is the gap, model grade minus label.
- **The Altman–Rijken asymmetry predicts the sign.** Upgrades lag more, so the dominant error at a promotion gate is "plays like 1D, labelled A". That is exactly the confirmed 1D→A push.
  - Corollary: pushes in the opposite direction at the same gate should fail.
  - Lag should concentrate at boundaries with special promotion rules (the kyu→dan gate) rather than ordinary within-band boundaries. This matches the two failed pushes elsewhere, if they were at other boundaries.
- **The threshold-and-partial-adjustment model gives a natural maximum push depth.** If labels move only once true strength exceeds the label by τ (≈1.8 notches in the agency case), then lagging players sit in a band of width ≈ τ above the gate in model space. Beyond that band, true 1D players dominate again.
  - Implied test: order decoded flips by model margin. The success rate should be high near the boundary and fall as the push goes deeper.
  - Practical rule: do not extend r beyond the tested range (r = 2) without a separate probe.
- **"The market does most of the adjusting".** Most rating-vs-market gaps close by the market moving toward the rating. Translated: most model-vs-label gaps are model error, not label lag. Only gaps with a mechanism and a predicted direction deserve a probe, so the prior π for generic boundary pushes should be low (≈ 0.1–0.2, Section 1F).
- **Heterogeneity (Hasan et al.).** The gap is more informative for opaque or volatile issuers. Translated: lag-driven errors should concentrate in questions where the rating system had little or noisy information, such as fewer games or volatile results. That is a candidate covariate for a *conditional* push: push harder where information is thin. It is untested and needs only a covariate already in the features.
- **The cliff makes the kyu→dan gate special.** Discrete consequences at the gate (dan status) should produce:
  - bunching of labels just below the gate (players parked at A);
  - a stricter or slower promotion process at that gate than elsewhere;
  - Kisgen-style behaviour near the threshold, a feature signature among players about to cross.
  A bunching check is free: compare the validation label histogram near A/1D with the smooth model-implied density, looking for excess label mass at A relative to model mass. Bunching estimators come from public finance and are not sourced here.
- **Transition-matrix tool.** Build the 10×10 matrix "model-implied grade → official grade" on validation out-of-fold data, and the partial analogue on decoded public flips. Then test cells at the gate (Section 3). If test games come from a later period than training, a TTC/PIT-type drift would shift several boundaries at once. The failure of pushes elsewhere argues for a local gate effect rather than global drift.
- **Momentum.** If per-question data has an order or dates of games, Lando–Skødeberg-type duration and previous-move dependence suggests recently improving players are the likeliest to be under-labelled. Without such information this is not usable.
- **Applicability.** High as a source of *priors and testable predictions* (direction, gate-specificity, depth limit, heterogeneity), at zero upload cost. Using these priors is what turns the 13 weak-prior candidates into a few mechanism-backed ones.

### Gaps
- I did not retrieve the original Altman & Kao (1992) paper or Moody's own gap studies. The "market does most of the adjusting" claim comes from press coverage of a Moody's study.
- I found no finance source quantifying how lag differs between the IG cliff boundary and other notches. Whether agencies are slower to cross BBB−/BB+ than other notches is not established in the sources retrieved.
- Go-server promotion rules at the kyu→dan gate (whether promotion is asymmetric or dan ranks are protected) were out of scope. They are needed to turn the analogy into a quantitative prior.

---

## 3. Credit-scoring monitoring and recalibration: PSI/CSI, central-tendency (calibration-in-the-large) recalibration, Basel per-grade back-testing (binomial, Hosmer–Lemeshow, Jeffreys), low-default and small-count methods, reject inference

### Takeaway
Per-grade PD back-testing is a direct template for per-boundary tests on decoded flips.
- Use exact binomial or Jeffreys per boundary and Hosmer–Lemeshow across boundaries.
- The small-count lessons transfer intact: tests on ≤ 10 outcomes are prior-dominated, and 0 failures in 8 only bounds the failure rate below ~31% at 95% (Pluto–Tasche style).
- Tasche's PD-calibration result supports correcting a boundary by a *scaled likelihood ratio* (a log-odds intercept shift, i.e. the r-multiplier on P(A)/P(1D)) rather than by scaling probabilities.
- PSI with a statistically derived threshold can tell population drift (p(x)) apart from label-convention lag (p(y|x)). At n = 800 the significance threshold is ≈ 0.02, far below the 0.10 rule of thumb.

### Cited Findings
- BCBS Working Paper 14, "Studies on the Validation of Internal Rating Systems" (rev. May 2005), lists the binomial test and the chi-square (Hosmer–Lemeshow) test among its quantitative calibration tests — [BIS WP14](https://www.bis.org/publ/bcbs_wp14.pdf); [BIS page](https://www.bis.org/publ/bcbs_wp14.htm).
- An EBA staff paper on systematic PD back-testing runs separate binomial tests per rating grade to detect miscalibration, and notes that Hosmer–Lemeshow assesses calibration across all grades at once — [EBA staff paper](https://eba.europa.eu/sites/default/files/2026-04/9917133d-9e43-4ceb-89ff-e3578f731c13/Staff%20paper%20-%20Systematic%20backtesting%20of%20probability%20of%20default%20models%20with%20regulatory%20data.pdf). See also [Validation of internal rating systems and PD estimates, arXiv physics/0606071](https://ar5iv.labs.arxiv.org/html/physics/0606071).
- **Jeffreys test (ECB).** It builds a credibility interval for the PD and checks whether the applied PD falls inside. p-value = F_{D+½, N−D+½}(PD), the Beta CDF, with D defaults among N. It applies per grade or at portfolio level. The ECB null is one-sided (applied PD ≥ true PD) — [arXiv 2005.03698 "Proving prediction prudence"](https://arxiv.org/pdf/2005.03698).
  - Critiques: banks overstate PDs to pass, and badly wrong PDs (e.g. 20%) can still pass.
  - Small-count behaviour: a 0.03% bucket with 6 loans and 0 defaults still gets a red flag, because the prior dominates.
  - Sources: [GARP, how to pass the Jeffreys test](https://www.garp.org/risk-intelligence/credit/probability-of-default-how-to-pass-the-jeffreys-test-and-improve-predictive-ability); [GARP critique](https://www.garp.org/risk-intelligence/credit/critical-deficiency-jeffreys-test-220513).
- **Low-default portfolios (Pluto & Tasche, "most prudent estimation").** PDs are estimated by upper confidence bounds while keeping the PD ordering across grades. With zero defaults, P(no defaults) = (1 − p)ⁿ under independence, which yields the bound. There are extensions for correlation and multiple periods. The upper bound equals a quantile of a Bayesian posterior (Tasche 2013) — [arXiv cond-mat/0411699](https://arxiv.org/abs/cond-mat/0411699); [BIS research note](https://www.bis.org/2026-07/rjadr.pdf).
- **Recalibration to a new central tendency (Tasche, "The art of PD curve calibration", J. Credit Risk 2013).** Calibration moves grade PDs to a new portfolio-wide average.
  - Popular "scaled PDs" are theoretically questionable and do poorly on test data.
  - The **"scaled likelihood ratio"** approach is sound and performs better.
  - Calibration has two steps (following Bohn & Stein) because the training good/bad mix differs from the live portfolio.
  - Sources: [arXiv 1212.3716v4](https://arxiv.org/pdf/1212.3716v4); [ar5iv v5](https://ar5iv.labs.arxiv.org/html/1212.3716v5); [RePEc](https://econpapers.repec.org/RePEc:rsk:journ1:2310092).
- **PSI.** PSI = Σᵢ (testᵢ − basisᵢ)·ln(testᵢ/basisᵢ) over bins.
  - Rule of thumb: < 0.10 no significant shift, > 0.25 major shift. These cut-offs trace to Lewis (1994) and come with no type-I/II error rates.
  - Yurdakul & Naranjo (J. Risk Model Validation 2020) derive a statistical threshold: χ²_{q, B−1}·(1/N + 1/M), with B bins and sample sizes N and M. feature-engine uses q = 99.9%.
  - Sources: [feature-engine docs](https://feature-engine.trainindata.com/en/latest/user_guide/selection/DropHighPSIFeatures.html); [RePEc](https://ideas.repec.org/a/rsk/journ5/7725371.html); [Yurdakul dissertation](https://scholarworks.wmich.edu/dissertations/3208/); [arXiv 2307.11878 on arbitrary Lewis constants](https://arxiv.org/html/2307.11878v4).

### Inferences
- **Per-boundary binomial / HL on decoded flips** (derivation). For the 1D→A push, the model expects E = Σ qᵢ ≈ 8 × 0.42 = 3.36 successes, with Var = Σ qᵢ(1 − qᵢ) ≈ 1.95.
  - Observed 8 gives exact one-sided p = 0.42⁸ ≈ 0.00097.
  - The HL-type χ²₁ = (8 − 3.36)²/1.95 ≈ 11.0 (p ≈ 0.0009).
  - This is a calibration failure at the A|1D boundary: the model's P(1D) is too high there.
- **Jeffreys / most-prudent bounds for the 8/8** (derivation, treating "failure" = true 1D).
  - The Jeffreys posterior on the failure rate is Beta(0.5, 8.5), mean 0.056.
  - Under a uniform prior, P(p_A < 0.5 | 8/8) = 0.5⁹ ≈ 0.002.
  - Pluto–Tasche-style one-sided bounds on p_A: 0.05^(1/8) = 0.688 (95%), 0.1^(1/8) = 0.750 (90%), median ≈ 0.917.
  - So the regulator-prudent planning value is p ≈ 0.69, which is 62% below the naive edge (Section 1E).
- **Small-count lesson (GARP example).** With n ≤ 8 any per-boundary test is driven by the prior, so the prior should be chosen deliberately: the model-calibrated rate, or an empirical-Bayes rate pooled across boundaries. It should not be left to an "uninformative" default.
- **Scaled likelihood ratio = intercept shift.** Multiplying P(A) by r while leaving the dan classes fixed is a shift of ln r in the A-vs-1D log-odds. That is Tasche's preferred scaled-likelihood-ratio recalibration applied to one boundary. The "scaled PD" analogue (adding or multiplying probability mass directly) is the variant Tasche finds unsound.
  - The two-step recipe maps cleanly: (1) keep the ensemble's ranking and shape; (2) re-estimate each boundary's central tendency for the test population.
  - Sources for step 2: the unlabelled test posterior mixture (EM prior adaptation, sibling note) and the decoded public counts.
  - Estimating δ from an 8/8 record has the same separation problem as a zero-default portfolio: the MLE δ̂ = ∞. Use a Firth or Jeffreys-penalised likelihood, maximising Σᵢ [yᵢ ln σ(ℓᵢ + δ) + (1 − yᵢ) ln(1 − σ(ℓᵢ + δ))] + ½ ln Σᵢ σ(1 − σ), which gives a finite δ. A Bayesian posterior on δ also works.
- **PSI thresholds for our sample sizes** (derivation from Yurdakul's formula):

| Comparison | 95% threshold | 99.9% threshold |
|---|---|---|
| 33k validation vs 800 test, B = 10 | 16.92 × 0.00128 ≈ 0.022 | 27.88 × 0.00128 ≈ 0.036 |
| vs 400 public only | 0.043 | 0.071 |

  - The 0.10 rule of thumb is therefore 3–5× above the noise floor here and would miss moderate drifts.
  - Diagnostic use: compute the PSI of the model's A-vs-1D margin distribution, validation vs test. A significant PSI means composition drift, so label-shift / EM adaptation is appropriate. A flat PSI plus a decoded calibration gap means label-convention lag (p(y|x) changed), which only probes can measure. This mirrors the EM argument in the sibling note.
  - CSI (per-feature PSI) can show which inputs drifted, but it cannot detect a pure labelling lag.
- **Reject-inference analogy (unsourced analogy).** Outcomes are observed only for questions whose prediction changed (the "accepted" applicants), and those are selected by model margin. Extrapolating the observed success rate to deeper pushes (r > 2) or other margins is the reject-inference problem: outcomes are missing not at random. It cannot be solved without a structural assumption, here the logistic shift δ plus a depth limit from Section 2. A dedicated probe is the only assumption-free check.
- **Applicability.** High for the per-boundary binomial/Jeffreys tests, prudent bounds, the scaled-LR correction and PSI thresholds. All are zero upload cost and a few lines of code on stored posteriors.

### Gaps
- I did not retrieve the BCBS WP14 body text or the ECB 2019 validation-reporting instructions directly. Test definitions come from the secondary sources cited.
- I did not search reject-inference literature (e.g. Hand & Henley), so the analogy is unsourced.
- I found no credit-risk source on per-grade tests with fewer than ~10 observations beyond the GARP Jeffreys example and the Pluto–Tasche low-default work.

---

## 4. Regime-change and structural-break detection (CUSUM, Chow, Andrews sup-F, Bai–Perron) for a localised change

### Takeaway
The useful lesson is the price of not knowing *where* the break is.
- A break at a pre-specified location is tested at nominal size (Chow).
- A searched location must clear a sup-type critical value (Andrews / Bai–Perron), roughly Bonferroni over the candidate locations.
- Pre-registering the kyu→dan gate from a mechanism (Section 2) is worth about a 9-fold smaller p-value requirement compared with scanning all 9 boundaries.
- Time-series break tests themselves are of little direct use: there is no time axis and only a handful of observations.

### Cited Findings
- Andrews (1993) and Andrews & Ploberger (1994) treat testing for structural change when the change point is unknown, using sup-Wald / sup-F statistics with tabulated critical values for a one-time break. Bai & Perron (Econometrica 66(1), 1998, pp. 47–78) extend this to multiple breaks with sup-F tests and tabulated critical values; theirs is the most widely used multiple-break strategy for linear models — [arXiv 2507.22204, empirical guide to multiple change-point testing](https://arxiv.org/pdf/2507.22204); [RePEc Bai–Perron](https://ideas.repec.org/a/ecm/emetrp/v66y1998i1p47-78.html).
- Rating processes show duration dependence and previous-move dependence (Lando & Skødeberg 2002), i.e. state-dependent rather than stationary dynamics — [CBS paper](https://research-api.cbs.dk/ws/files/59164378/Confidence_sets_for_continuous_time_rating.pdf).

### Inferences
- **Translation.** Take the ordered boundaries b = D|C … 5D|6D (9 locations) as the "time" index and the per-boundary calibration offset δ_b as the regression parameter. A localised grade lag is a break in δ at one b.
  - Chow test (known location): the per-boundary binomial z at the gate, compared with the nominal critical value.
  - Andrews sup test (unknown location): max_b z_b. Under the null with independent boundaries, E[max of 9 z's] ≈ 1.52 (DSR bracket). The Bonferroni-equivalent level is α/9 ≈ 0.0056 at α = 0.05.
  - Same logic as DSR/MinBTL: specifying the gate in advance from a promotion-rule mechanism removes the search penalty.
- **Bai–Perron framing.** Lags may exist at several gates (e.g. a band boundary within kyu, or high-dan boundaries). Each failed push at a different boundary is evidence of "no break" there. The relevant model is "breaks only at gates with special rules", not a free search.
- **CUSUM-style evidence accumulation across uploads (idea; CUSUM and SPRT sources not retrieved).** Repeated small probes at the same boundary can be pooled as a running sum of (Oₜ − Eₜ) with a pre-set stopping boundary. That is preferable to judging each 5-question probe alone, because pooled n reaches the probe-size requirement of Section 1C.
- **Applicability.** Conceptually high (pre-specification discount), practically low (no time series, tiny n). Zero cost.

### Gaps
- I did not retrieve sources for Page's CUSUM, the Chow test or Wald's SPRT. The CUSUM/SPRT suggestion above is my adaptation, not a sourced recommendation.
- No source found applies break tests across ordinal category boundaries rather than over time.

---

## 5. Decision and portfolio view of the final submission: pushes as bets with uncertain edge; Kelly, mean–variance and "shrink the edge" heuristics; combining kept adjustments

### Takeaway
The final submission is a single, additive, one-shot bet.
- With a risk-neutral objective, the optimum is per-question Bayes: flip a question iff its posterior probability of the target class exceeds the break-even 0.5 (for A/1D), with the posterior integrating over uncertainty in the boundary shift δ.
- The "fractional Kelly" lesson (estimation error ⇒ bet less) corresponds to shrinking δ̂ toward 0 with a prior, not to an arbitrary fraction. It also means not extending pushes beyond the range actually probed.
- Pushes acting on the same boundary are the same factor exposure, so their evidence must not be double-counted.
- For the confirmed push, the expected private gain is positive under every haircut considered. The probability of a negative private outcome ranges from ≈ 0 to ≈ 18%, depending on the prior.

### Cited Findings
- MacLean, Thorp & Ziemba (Quantitative Finance 2010, "Good and bad properties of the Kelly criterion"): estimation errors can be moderated with fractional Kelly, i.e. wagering a fraction of the Kelly bet (half or quarter Kelly) — [Berkeley copy](https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf).
- Fractional Kelly gives a complete growth-versus-security trade-off — [MacLean, Ziemba & Blazenko, Management Science 38(11) 1992](https://ideas.repec.org/a/inm/ormnsc/v38y1992i11p1562-1585.html).
- Betting above Kelly raises risk and lowers long-run growth, eventually making it negative — [Berkeley copy](https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf).
- Estimation error can cause over-betting and ruin. Rising & Wyner shrink the estimated mean return toward the risk-free rate — [arXiv 2503.17927](https://www.arxiv.org/pdf/2503.17927).
- Practitioner convention (lower-authority source): overbetting costs more than underbetting, hence half or quarter Kelly — [LuxAlgo](https://www.luxalgo.com/library/concept/kelly-criterion.md).
- The flat 50% Sharpe haircut is criticised as a serious mistake, because the correct haircut depends on significance and number of tests — [Harvey & Liu 2015](https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF).
- Empirically, live Sharpe ratios of complex bank strategies were ~62–73% below backtests, and 50% derating "is not conservative enough" — [CXO Advisory](https://www.cxoadvisory.com/big-ideas/live-performance-of-alternative-beta-products/); [Alpha Architect](https://alphaarchitect.com/?p=36892).
- The Hansen–Lunde–Nason Model Confidence Set, which identifies the set of models not significantly worse than the best, is implemented in `arch` — [arch docs](https://arch.readthedocs.io/en/latest/multiple-comparison/multiple-comparison-reference.html).

### Inferences
- **Per-question Bayes rule** (derivation). Given a posterior π(δ) for the A|1D log-odds shift, flip question i iff E_δ[σ(ℓᵢ + δ)] > 0.5, where ℓᵢ = ln(P(A)/P(1D)) from the ensemble. With partial credit and other neighbours, pick the label maximising E[score]. With a symmetric posterior on δ, the flip boundary sits at ℓᵢ = −median(δ). Uncertainty alone does not move it; what shrinks the push is the prior pulling δ̂ toward 0.
  - That shrinkage is the Kelly / Rising–Wyner "shrink the mean" analogue.
  - Since the 8/8 separates, δ̂_MLE = ∞, so a penalised or Bayesian δ (Section 3) is required, not optional.
- **Mean–variance of the private gain of one push** (derivation). G = g·Σᵢ(2Yᵢ − 1)/400, with E[G] = g·n·(2p̃ − 1)/400.
  - Var[G] = (g/400)²·[n(1 − e²) + 4n²Var(p)]. The second term is parameter uncertainty shared by all flips of the push, which acts as a factor exposure.
  - With n_priv = 8:

| Posterior on p | E[G] | SD | P(G < 0) |
|---|---|---|---|
| Uniform prior, Beta(9,1) | +0.0101 | 0.0035 | ≈ 0.2% |
| Skeptical prior, Beta(12.2, 5.8) | +0.0045 | 0.0050 | ≈ 18% |

  - Either way, keeping the confirmed push is the expected-value-maximising choice. The residual risk is mostly parameter uncertainty, not sampling noise.
- **Combining kept adjustments.**
  - Pushes at different boundaries on disjoint questions: gains add and variances add. Include each iff its posterior expected gain > 0 (risk-neutral).
  - Pushes at the *same* boundary (the earlier A/1D composition bet and the r = 2 push) load on one factor δ_{A|1D}. Combine them through one posterior on δ, with total exposure = the union of private flips at that boundary. Do not treat their public results as independent confirmations, and avoid concentration by not stacking a second bet on the same factor without new evidence.
- **"Shrink the edge" heuristics translated.**
  - Flat 50%: equivalent to a model-calibrated prior worth ~6 questions (Section 1E). It is reasonable as a planning value for the 8/8 push but too harsh in Harvey–Liu's sense for very strong results.
  - Harvey–Liu nonlinear haircut: 20% (N = 3) to 46% (N = 16) for the 8/8; 100% for a marginal 10/13 when N = 16, where the adjusted p ≈ 0.74 and the edge is zeroed.
  - Suhonen's 62–73%: an upper benchmark for heavily searched, complex adjustments. It is a good default for the 13 weak-prior candidates if they are chosen by browsing the public decode.
- **Final-file selection among full candidate submissions.** A Model-Confidence-Set-style step keeps all full submissions not significantly worse than the best on the public decode, then chooses by mechanism and prior. With only a few changed questions per comparison the set will usually contain all candidates, so the decision falls back on the Bayes/EV computation above.
- **Rank objective (not finance-sourced).** If the goal is a ranking or prize cutoff rather than expected score, variance matters: reduce variance when expected score is above the cutoff, add it when below. That changes "include iff E > 0" into "include iff it raises P(score > cutoff)".

**Applicability scorecard** (my assessment):

| Method | Translation | Benefit | Upload cost | Risk |
|---|---|---|---|---|
| Exact binomial + Bonferroni/Holm (HLZ, StepM logic) | keep thresholds in table 1B | High: prevents claiming lags from 5–8-question luck | 0 | Over-conservative for bets |
| MinBTL analogue | n > (E[max_N]/e*)² before probing | High: tells which candidates are worth an upload | 0 | Needs honest N |
| Bayes / EV keep rule + EVOI | Section 1F–G | High: right objective for the final file | 0 (saves uploads) | Prior π, p1 misspecified |
| DSR | Φ(z − E[max_N]) | Medium: quick deflation | 0 | Degenerate at perfect records |
| PBO/CSCV | r-grid on validation OOF | Low–medium | 0 | Measures the wrong shift |
| Rating-lag priors (Altman–Rijken, MIR, cliff) | direction, gate, depth limit, heterogeneity | High: shrinks 13 candidates to the few with mechanism | 0 | Analogy may not hold for Go servers |
| Per-boundary binomial/Jeffreys/HL + prudent bounds | Section 3 | High | 0 | Prior-dominated at n ≤ 8 |
| Scaled-LR recalibration (Tasche) | intercept shift δ_b, Firth-penalised | High: principled push form | 0 | Extrapolation beyond probed depth |
| PSI with χ² threshold | composition vs label-convention drift | Medium: diagnosis | 0 | Silent for pure label lag |
| Break tests | pre-specification discount | Medium (concept) | 0 | No time axis |
| Fractional Kelly / mean–variance | shrink δ; factor-exposure accounting | Medium–high | 0 | Rank objective may differ |

### Gaps
- No finance source addresses a one-shot, additive, ranked-tournament payoff. The rank-objective point is my inference.
- MacLean et al.'s quantitative estimation-error results for half Kelly were not retrieved, only the qualitative claim.
- I did not verify Hansen's SPA recentering rule or the Model Confidence Set algorithm beyond the `arch` documentation, so the MCS step is described only qualitatively.
- These derivations need inputs not available to me: the actual n_priv of the kept push, the per-question margins ℓᵢ of its flips, and the counts of the two failed probes. With those, the haircut table (1E) and the mean–variance numbers can be recomputed exactly.

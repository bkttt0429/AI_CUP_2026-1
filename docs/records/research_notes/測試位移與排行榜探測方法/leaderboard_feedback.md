# Extracting information from public-leaderboard feedback, and carrying it to the private half

Scope note: the competition setting (10 ordered classes, 400 public / 400 private, score = 1 exact / e^-1 adjacent / 0 otherwise, last upload counts) is taken from the assignment brief. Every number under "Inferences" that is not followed by a source link is my own derivation from that brief; the assumptions are stated next to it. Notation used throughout: c = e^-1 = 0.3678794; n_e, n_a = number of public questions scored exact / adjacent; X = Δn_e (net one-level corrections), Y = Δn_e + Δn_a (= minus the change in the number of zero-score questions), so 400·ΔS = (1-c)·X + c·Y = 0.63212·X + 0.36788·Y, as in the brief.

## Q1. Leaderboard-probing literature and practice: what has been done, and did it help or hurt the private rank?

### Takeaway
Exact or high-precision public scores leak test labels quickly (Whitehill recovered all 512 test labels of a Kaggle stage in fewer than 100 queries). This only pays off on the private side when the probed fact is a property shared by the public and private halves, for example a class-prior shift (Quora) or a group that shares one label (Red Hat, West Nile). Probing that fits public-only noise or individual public labels moved teams down the private board (Whitehill's #4 became 225th; LANL had a large shake-up).

### Cited Findings
- Whitehill (2017/2018) shows that a log-loss oracle lets a contestant infer ground-truth test labels batch by batch. On stage 1 of Intel & MobileODT Cervical Cancer Screening he reached a perfect score and rank #4 of 848 without supervised learning on the images. — [arXiv 1707.01825](https://arxiv.org/pdf/1707.01825); [AAAI-18 workshop version](https://cdn.aaai.org/ocs/ws/ws0421/16161-75972-1-PB.pdf)
- Mechanics of Whitehill's attack:
  - Each round fixes already-inferred examples to their labels, sets m probed examples to rows of a probe matrix G_m, and sets the rest to the uniform 1/c (c = number of classes), so the unprobed rows add a known constant.
  - The probed loss is ℓ_m = [(n−m−k)·log c − n·ℓ_n]/m, and he brute-forces all c^m labelings.
  - Probe entries are a·10^b with b ∈ {−14…0}, chosen by Monte Carlo to maximise the minimum loss gap Q between labelings: Q(G4) = 0.019248 and Q(G6) = 0.001526; no G7 with Q > 1e-4 was found.
  - Queries: 2 with m=1, 30 with m=4, then m=6; 98 rounds in total; all 512 labels recovered.

  — [ar5iv 1707.01825](https://ar5iv.labs.arxiv.org/html/1707.01825)
- Whitehill's variant for an oracle that scores an unknown fixed subset (the public split) adds 0/1 membership indicators z per probed example and solves for labels and z jointly. In simulation (n=2048, s=512, 3 classes, m=4), mean label accuracy depended on the oracle's displayed digits p: 38.8% (p=1), 47.8% (2), 59.4% (3), 78.7% (4), 93.6% (5). — [ar5iv 1707.01825](https://ar5iv.labs.arxiv.org/html/1707.01825)
- Whitehill fell to 225th of 848 once the final ranking used the full test set. He recommends ranking contestants only on examples the oracle never scored. — [ar5iv 1707.01825](https://ar5iv.labs.arxiv.org/html/1707.01825)
- Blum & Hardt's "boosting attack" submits k random label vectors, keeps those scoring better than chance, and takes a coordinate-wise majority vote. When observed losses are accurate to n^-1/2, this reaches public loss 1/2 − Ω(√(k/n)) (Theorem 5.1). Their experiment used N=12,000 labels with n=4,000 public; Kaggle's 1e-5 rounding is far below the 1/√n ≈ 0.0158 threshold, so the attack succeeds. — [Blum & Hardt, The Ladder, ar5iv 1502.04585](https://ar5iv.labs.arxiv.org/html/1502.04585)
- Zheng (2015): with full-score feedback, boosting on a 1,000-sample validation set "climbs from 0.5 to 0.8" within about 10^3 submissions. With exact ranks plus two accounts, flipping one prediction at a time eventually reaches score 1. Multiple accounts also defeat the Ladder. Recommendations: release only the best score so far, use coarse precision, use a private test set, limit daily submissions, and ban multiple accounts. — [Zheng, Toward a Better Understanding of Leaderboard, ar5iv 1510.03349](https://ar5iv.labs.arxiv.org/html/1510.03349)
- Coursera "How to Win a Data Science Competition" lecture on leaderboard probing:
  - Extracting public ground truth by small targeted changes is "usually pretty harmless" because it does not touch the private half.
  - Probing "consistent categories", meaning groups whose label is the same in the public and private parts, was "a key for winning" in Red Hat and West Nile.
  - In Quora Question Pairs, constant-prediction submissions under log loss gave the test positive ratio; rebalancing to it "gave a huge boost in leaderboard score". Public and private share the label distribution, so this was a population-level fact that transferred.

  — [lecture transcript](https://zfelleg.useribm.hu/videos/courses/specializations/advanced_machine_learning/competitive-data-science/06_data-leakages/01_data-leakages/02_leaderboard-probing-and-examples-of-rare-data-leaks.en.txt)
- The GLUE paper notes that the QQP test set has a different label distribution from the training set. This is consistent with the probed prior shift being real. — [GLUE, arXiv 1804.07461](https://arxiv.org/pdf/1804.07461)
- LANL Earthquake Prediction (2019): a team reports finishing 1st on the public LB among ~4,500 teams "by using LB probing strategy". A solutions summary calls it "a competition with a high leaderboard shakeup because ppl overfit the public LB". — [TDS "How to LB probe on Kaggle" (fetch blocked, search snippet only)](https://towardsdatascience.com/how-to-lb-probe-on-kaggle-c0aa21458bfe); [kaggle-solutions summary](https://kaggle-solutions-quartz.pages.dev/competitions/LANL-Earthquake-Prediction)
- Roelofs et al. (NeurIPS 2019) meta-analysed 120 Kaggle competitions; 34 accuracy competitions with at least 1,000 submissions were analysed in detail.
  - Public-vs-private fits lie close to the diagonal in all but one competition (12598, a non-i.i.d. split).
  - Mean accuracy differences between −2% and +2% are roughly symmetric around zero, and typical overfitting effects are "less than 1% classification accuracy".
  - Outliers came from non-i.i.d. splits, a test set derived from only 7 subjects, a 90-example public set, and 100/209-example private sets.
  - "The top 10% of submissions always exhibit the smallest p-values", meaning the most irregularity is among the best submissions. In competition 3788 the top 10% overestimate private accuracy by about 1%.
  - The authors suggest "at least 10,000 examples is a reasonable minimum test set size to protect against adaptive overfitting", and that non-i.i.d. splits are a bigger danger than test reuse.

  — [Roelofs et al. text](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/NeurIPS/2019/a_metaanalysis_of_overfitting_in_machine_learning__682ee801.txt); [NeurIPS page](https://papers.nips.cc/paper/9117-a-meta-analysis-of-overfitting-in-machine-learning)

### Inferences
- **The 7-decimal readout always decodes to (n_e, n_a).** The best rational approximation of c with denominator ≤ 400 is 71/193, with |193c − 71| = 0.000732. By the best-approximation property of continued-fraction convergents (1/e = [0;2,1,2,1,1,4,1,1,6,…], convergents …32/87, 39/106, 71/193, 465/1264), no |Δn_a| ≤ 400 does better. So two different (n_e, n_a) pairs differ in public score by at least 0.000732/400 = 1.8e-6, which is about 18 units of the 1e-7 display.
- **The leaderboard therefore leaks far more than Blum & Hardt's threshold.** Each upload reveals the exact pair (n_e, n_a) on the public half, which is two integers. The 1e-7 precision is orders of magnitude finer than 1/√400 = 0.05. This explains why exact decoding works. It also means that public-only gains are cheap and say nothing about the private half.
- **Taxonomy for our pushes:**
  - "Which public questions are right" is Whitehill-type information and has zero private value.
  - "Is there a boundary shift between classes p and p+1 in this test population" is Quora/consistent-category information. It transfers only if the public/private split is random with respect to that population property.
  - The 1D→A result (8/8) is the second kind.
- **Main transfer risk.** Following Roelofs, the main risk is a non-random public/private split, not adaptivity per se. If AI CUP split by time, player batch or ID order, a shift seen in public may be absent in private. A cheap check is to compare the model's predicted-class histogram and feature means on the public questions against the private ones, if membership is known.

### Gaps
- I could not retrieve the LANL probing team's final private rank (the TDS page returned 403), so "probing team dropped by X" is not sourced.
- I found no write-up of the Mercedes-Benz Greener Manufacturing probing or shake-up in accessible sources.
- I did not find Alek Trott's post on efficient low-submission probing, which the Coursera lecture cites but which did not appear in search.
- AI CUP 2026's specific rule text on probing, multiple accounts and the public/private split method was not researched; it is out of scope and needs checking.

## Q2. Group testing and combinatorial search with additive (sum) queries: can several pushes share one upload and still be decoded?

### Takeaway
Yes, up to two pushes per upload, decoded exactly. Each upload yields two integers (X, Y), and the move types available give each push a signature in (X, Y) space that depends on whether the push is moved one level forward, two levels forward, or one level backward. Any two different signatures are linearly independent, so two pushes are exactly identifiable (apart from small off-boundary contamination, which the existing DP absorbs). That halves the cost: 13 pushes take 7 uploads instead of 13. Three or more pushes per upload are not identifiable without priors. With realistic priors they stay ambiguous (worked example below: about 30% wrong-attribution risk). Dorfman-style pooling of several pushes in the same signature does not beat 7 uploads at a prior signal rate around 0.2.

### Cited Findings
- **Quantitative group testing.** Quantitative group testing (QGT) is the coin-weighing problem with a spring scale, introduced by Shapiro in 1960: each test returns how many defectives are in the tested group. In the noiseless case it cuts the number of tests from O(k log(n/k)) to O(k log(n/k)/log k). — [Aldridge, Johnson & Scarlett, Group Testing: An Information-Theory Perspective, arXiv 1902.06002](https://arxiv.org/pdf/1902.06002); [Simple and efficient strategy for coin weighing with a spring scale, arXiv 1805.02977](https://www.arxiv.org/abs/1805.02977)
- **Coin weighing with d counterfeit coins.** Bshouty gives an information-theoretic lower bound and algorithms within a factor of 2 of it. — [Bshouty, COLT 2009](https://www.learningtheory.org/colt2009/papers/004.pdf)
- **Nested adaptive plans.** In a nested plan, each test after a positive must be a proper subset of the positive group; the optimal nested plan's performance is characterised in the paper. — [Wang, Zhao & Chuah, arXiv 1407.2283](https://arxiv.org/pdf/1407.2283)
- **Unknown integer weights.** A generalised version with unknown integer coin weights (closer to our case, where a "null" push contributes a random integer) is studied by Karimi et al., who report about 31.75% fewer weighings than the optimal nested strategy asymptotically. — [arXiv 1805.02977](https://www.arxiv.org/abs/1805.02977)
- **Non-adaptive QGT.** An order-optimal, polynomial-time non-adaptive QGT algorithm is given by Soleymani (2024). — [PMLR v247](https://proceedings.mlr.press/v247/soleymani24a/soleymani24a.pdf)
- **Whitehill's per-batch encoding is the closest leaderboard analogue.** He gives each probed example a distinct real weight (a·10^b) so the scalar loss decodes to a unique labeling. It hit a resolution wall at m=7 (no G7 with minimum gap above 1e-4). — [ar5iv 1707.01825](https://ar5iv.labs.arxiv.org/html/1707.01825)

### Inferences
All derived from the scoring rule in the brief.

**1. Per-question contribution (X, Y) by move type.** The current prediction is p and the true class is t.

| move | t=p | t=p+1 | t=p+2 | t=p+3 | t=p−1 | t=p−2 |
|---|---|---|---|---|---|---|
| forward p→p+1 | (−1,0) | (+1,0) | (0,+1) | 0 | (0,−1) | 0 |
| two-level p→p+2 | (−1,−1) | 0 | (+1,+1) | (0,+1) | (0,−1) | 0 |
| backward p→p−1 | (−1,0) | (0,−1) | 0 | 0 | (+1,0) | (0,+1) |

Check against the brief: 5D→6D with 1 right, 5 wrong and 1 two-off gives X = −4, Y = ±1, which matches.

**2. Signatures.** For a push of n public questions whose truths are mostly a at p and b at p+1 (a + b ≈ n), the readout (X, Y) changes per unit of b by:
- forward: (2, 0), i.e. X = 2b − n, Y = 0 plus contamination;
- two-level: (1, 1), i.e. X = Y = b − n plus contamination;
- backward: (1, −1), i.e. X = −a, Y = −b plus contamination.

**3. Same-signature pushes cannot be separated.** Two forward pushes in one upload give only X_A + X_B, and Y holds only the two-off contamination. This is the problem the brief anticipated. Forward-only probing also leaves the Y dimension unused.

**4. Mixed-signature designs are exactly identifiable.** The signature pairs have determinants 2 (forward and two-level), −2 (forward and backward) and −2 (two-level and backward), so each pair can be solved.
- Design "A forward + B two-level": X = (2b_A − n_A) + (b_B − n_B) and Y = b_B − n_B. This gives b_B = n_B + Y and b_A = (X − Y + n_A)/2.
- The parity of X − Y + n_A is a free error check: an odd value means off-boundary truths contaminated the readout.
- Worked example: n_A = 8 with truth 7 at p+1; n_B = 6 with truth 2 at p+1. Then X = 6 − 4 = 2 and Y = −4, which decode back to b_B = 2 and b_A = (2 + 4 + 8)/2 = 7.
- Design "A forward + B backward": Y = −b_B and b_A = (X + Y + n_A + n_B)/2. Here B's p+1 and p counts are read separately, which is extra information about B.
- Cost: B's public score drops in that upload. This does not matter as long as the probe is never the last upload.
- Contamination: forward X is clean. Two-off truths (t = p+2, p−1, p+3, p−2) enter Y, or X for backward, with weight ±1. At the observed two-off rate of about 1 in 7 to 1 in 13, the expected error is ≤ 1 count per push. The existing forward-backward DP handles this if it is given the per-question tables above.

**5. Three pushes per upload are not identifiable.** For forward A, two-level B and backward C, the kernel is (−1, +1, +1): the configurations (b_A, b_B, b_C) and (b_A − 1, b_B + 1, b_C + 1) give the same readout.

Worked example. Assumptions: n = 8 each; prior P(signal) π = 0.2; signal accuracy θ1 = 0.9; null accuracy θ0 = 0.35; truth (8, 3, 3). The readout is consistent with (8,3,3), (7,4,4), (6,5,5), (5,6,6), (4,7,7) and (3,8,8), with posteriors 0.47, 0.20, 0.03, 0.02, 0.10 and 0.18. So P("A is the signal") ≈ 0.67 against P("B and C are both signals") ≈ 0.29, which is unusable for keep or drop decisions.

**6. Information bound.**
- Each push is a binary unknown with π = 0.2, so 13 pushes hold H = 13·H(0.2) = 13 × 0.722 = 9.4 bits.
- A single exact probe of an n = 8 push leaves about 0.10 bits of residual uncertainty: posterior P(signal | b) is 0.998 at b = 8, 0.966 at b = 7, 0.63 at b = 6, 0.093 at b = 5, and ≈ 0 at b ≤ 4. It therefore yields about 0.62 bits.
- The two-slot design yields about 1.24 bits per upload, so 7 uploads give about 8.7 of the 9.4 bits.
- A noiseless sum oracle on binary items could do far better (lower bound about m/log2(m+1) = 13/3.8 ≈ 3.4 queries). That bound does not apply here, because a null push's random contribution (variance ≈ 4θ0(1−θ0) ≈ 0.91 per question) is noise of the same size as the signal gap.

**7. Dorfman-style pooling inside one slot.**
- Two n = 8 forward pushes in a pool give a pooled X with these normal approximations:
  - both null: mean −4.8, sd 3.8;
  - one signal: mean 4.0, sd 3.2.
- With the rule "flag the pool if X ≥ 0", false-alarm and miss rates are each about 10%.
- At π = 0.2: 4 first-stage uploads (two pools per upload) plus about 3 follow-ups (0.36 × 0.89 + 0.64 × 0.10 ≈ 38% of 7 pools flagged) is about 7 uploads, the same as the exact two-slot design but with about 10% misses.
- At π = 0.1: about 5.5 uploads against 7, still with about 11% misses.
- Conclusion: use the exact two-slot design. Pool only if the upload budget is very tight and π is small.

**8. Known-label "calibration" questions.** For example, the 8 now-known A's. They add a known constant and cost no identifiability, so they can safely be included in probe uploads, for instance to sanity-check the decoder. They cannot create a third independent channel, because every question contributes to the same two integers.

### Gaps
- I found no paper that treats exactly "two-integer readout, items with ordinal move signatures". The design above is derived here and not taken from a source. It should be validated by simulation in the existing DP before use: sample truths from the model posteriors, simulate the readout, decode, and measure the misclassification rate.
- I did not fetch the full text of the QGT papers, so asymptotic constants are not quoted. They are of little relevance at m = 13.

## Q3. Bayesian sequential design and bandits: which push to probe next, and when to stop

### Takeaway
The best-arm and thresholding-bandit framing fits only loosely. One upload of a push reveals its whole public part exactly, so repeating a probe gives nothing new. The problem therefore reduces to a budgeted value-of-information (VOI) knapsack: give each push a VOI ≈ π_i · n'_i · 0.00126 (n'_i = private questions flipped; derivation below), probe in descending VOI at 0.5 upload per push (two-slot design), stop when the remaining VOI is below the value of the next-best use of an upload, and always reserve the final slot for re-uploading the chosen submission. The highest-VOI probe is likely a "dose" (radius) extension at the already-confirmed 1D→A boundary, not a new weak-prior push.

### Cited Findings
- The thresholding bandit problem is a fixed-budget pure-exploration task: find the arms whose means exceed a threshold, to a given precision, within a fixed horizon. Locatelli, Gutzeit & Carpentier's APT algorithm is parameter-free and optimal, with matching upper and lower bounds. — [ICML 2016 / arXiv 1605.08671](https://ar5iv.arxiv.org/html/1605.08671); [PMLR page](https://proceedings.mlr.press/v48/locatelli16.html)
- The Ladder's parameter-free rule accepts a new submission only if its score beats the previous best by s/√n, where s is the standard deviation of the paired per-sample loss difference. The authors call this a heuristic one-sided paired t-test at about the 0.15 level, with assumptions broken by adaptivity. — [Blum & Hardt, ar5iv 1502.04585](https://ar5iv.labs.arxiv.org/html/1502.04585)

### Inferences
**1. Per-question private value of applying a one-level push** with true neighbour rate θ, ignoring two-off terms: g(θ) = (2θ − 1)(1 − c)/400 = (2θ − 1) × 0.00158.
- Signal (θ1 = 0.9): +0.00126 per private question.
- Null (θ0 = 0.35): −0.00047. With θ0 = 0.4: −0.00032.
- Two-off truths add ±c/400 = ±0.00092 each.

**2. Default (unprobed) decision.** Apply the push iff π·0.00126 > (1 − π)·0.00047, i.e. π > π* ≈ 0.27. Pushes with "weak priors" (π ≈ 0.1–0.2) are therefore not applied blind.

**3. VOI of a probe** (perfect-information upper bound): VOI_i ≈ π_i · n'_i · 0.00126.
- n' = 8, π = 0.2 gives 0.0020; π = 0.1 gives 0.0010; n' = 4, π = 0.1 gives 0.0005.
- An exact public probe with n_pub = 8 recovers about 86% of perfect information (0.62 of 0.72 bits, Q2).
- Across 13 pushes with mean n' = 8 and π = 0.15, probing everything is worth about +0.020 private in expectation. This is the value of about 2 true signals at about +0.010 each.

**4. Ranking and budget.**
- Rewards are additive across pushes, unlike "pick the best" problems such as Weitzman's Pandora's box. So the optimal policy with a budget of B uploads is to probe the 2B pushes with the highest VOI, paired into mixed-signature uploads.
- Adaptivity helps only in two ways: through hierarchical updating (a confirmed shift at one boundary raises π for related boundaries), and through dose extensions.

**5. Dose (radius) probing at a confirmed boundary.**
- 8/8 at r = 2 suggests the 1D/A shift may be larger.
- The increment r = 2 → 3 (the next questions in posterior order) has a prior well above the weak-prior pushes; π ≈ 0.5 is a guess. That gives VOI ≈ 0.5 × n' × 0.00126 ≈ 0.005 for n' = 8, 2–5× any weak-prior push.
- Successive increments have decreasing accuracy θ(r). Locating r* where θ(r) crosses 0.5 is a one-dimensional threshold search: bisection or the thresholding bandit above, with one exact readout per step.

**6. Stopping rule.**
- Stop when the next probe's VOI is below the expected value of an alternative use of the upload slot, or when uploads left = 1.
- With an unknown daily cap d and about 16 days, there are at most 16d − 1 probe uploads, because the final submission must be re-uploaded last (per the brief, after ct14/ct15).
- Schedule probes early. An unexpected end-of-period cap change should not leave a probe as the last upload.

**7. Preposterior check for each candidate design.**
- Before spending an upload, simulate truths from the DP's current per-question posteriors, compute the (X, Y) readout for the design, and decode.
- Record the probability that the keep/drop decision changes. If it is under about 5%, the probe is not worth an upload.
- This is standard Bayesian experimental design and uses the DP the team already has.

### Gaps
- I did not fetch best-arm-identification surveys (Kaufmann–Cappé–Garivier), Weitzman (1979) or information-directed sampling. They are named here from background knowledge only, and none of the numbers above depend on them.
- The real values of π, θ1 and θ0 for the 13 candidates are unknown. The figures above are illustrative, and the brief's observed failures (6/13 ≈ 0.46 and 1/7 ≈ 0.14) suggest θ0 varies by boundary.
- The daily upload cap is unknown (stated in the brief).

## Q4. Public→private transfer: shrinkage, winner's curse, overfitting control and final-submission choice

### Takeaway
The public readout of a push measures its public questions exactly. The private effect is a different binomial draw from the same θ, so it should be estimated with a hierarchical (empirical-Bayes) model across all probed pushes, never by the raw public delta. Keeping only pushes that look good on public inflates naive estimates (winner's curse). Concrete case: a null push (θ0 = 0.4) that passes "b ≥ 6 of 8" looks like θ̂ ≈ 0.77 and a +0.0068 private gain, but its true expected private effect is −0.0025. Adaptive-overfitting theory says a 400-question public half gives essentially no protection against public-only gains. Protection has to come from only acting on population-level hypotheses that have a private counterpart.

### Cited Findings
- **Selection bias and Tweedie's formula.** Tweedie's formula (Robbins 1956, developed by Efron) is an empirical-Bayes correction for selection bias: "the largest few of the z_i's are likely to substantially overestimate their corresponding μ_i's" (winner's curse / regression to the mean). It links to James–Stein shrinkage. — [Efron 2011, JASA, PubMed 22505788](https://pubmed.ncbi.nlm.nih.gov/22505788/); [paper copy](https://www.scribd.com/document/193214425/2011TweediesFormula-1)
- **Limits under dependence.** Empirical-Bayes and Tweedie corrections can be biased under strong dependence between units; a convolution-based repair is proposed. — [Chen et al., bioRxiv 2023.09.22.558978](https://www.biorxiv.org/content/10.1101/2023.09.22.558978.full.pdf)
- **Ladder guarantee.**
  - Releasing a new best only when it improves by more than η gives leaderboard error O(log^{1/3}(kn)/n^{1/3}), against roughly √k growth for Kaggle-style full feedback.
  - Any estimator incurs Ω(√(log k/n)) in the worst case.
  - Under the boosting attack, the Ladder's bias is O(√(log k/n)).

  — [Blum & Hardt, ar5iv 1502.04585](https://ar5iv.labs.arxiv.org/html/1502.04585)
- **Multiple classes reduce worst-case overfitting.** With k queries, n points and m classes, worst-case bias is Õ(max{√(k log n/(mn)), k/n}). — [Feldman, Frostig & Hardt, arXiv 1905.10360](https://arxiv.org/pdf/1905.10360); matching efficient attacks: [arXiv 1908.03156](https://web3.arxiv.org/abs/1908.03156)
- **Evidence from Kaggle.** Public→private gaps are mostly small and symmetric, and the largest irregularities sit in the top 10% of submissions. Small test sets (90–209 examples) and non-i.i.d. splits produced the outliers, and ≥ 10,000 test examples is suggested as a safe size. — [Roelofs et al. 2019](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/NeurIPS/2019/a_metaanalysis_of_overfitting_in_machine_learning__682ee801.txt)
- "The final winner of the competition is not the 'winner' on the leaderboard", owing to overfitting or hacking. — [Zheng, ar5iv 1510.03349](https://ar5iv.labs.arxiv.org/html/1510.03349)
- The Dwork et al. reusable holdout (Thresholdout, Science 2015) answers holdout queries with added noise to keep adaptive reuse valid. Not fetched in this session. — [Science 10.1126/science.aaa9375](https://doi.org/10.1126/science.aaa9375)

### Inferences
**1. Order of magnitude for worst-case public overfitting at our scale.**
- Plugging in k = 16, n = 400, m = 10 gives √(16 · ln 400 / 4000) ≈ 0.15, with constants hidden.
- Roelofs's 10,000 guideline is 25× our public half.
- Implication: any decision driven by the public score of a whole model swap (for example ct9 vs ct10) should be discounted heavily. A targeted push probe is safer, because it has one degree of freedom and is a population-level hypothesis.

**2. Shrinkage model** (two-group or Beta mixture, fitted by empirical Bayes):
- θ_i ~ π·Beta(signal) + (1 − π)·Beta(null).
- Fit π and the Beta parameters by marginal likelihood over all probed pushes: the 3 so far (8/8, ≈6/13, 1/7) plus future probes. Optionally include CV-derived priors per boundary.
- Private expected gain = n'_i · (2E[θ_i | b_i] − 1) · 0.00158, plus two-off terms.
- The model accounts for selection automatically when the prior is right (Efron). Do not plug in b/n.

**3. Worked shrinkage for the 8/8 push** with n' = 8:
- Observed public gain: 8 × 0.00158 = +0.0126.
- Uniform Beta(1,1) prior: E[θ] = 0.9, so private ≈ +0.0101.
- Skeptical Beta(1,3) prior: E[θ] = 0.75, so +0.0063.
- Mixture prior (π = 0.2, θ1 = 0.9, θ0 = 0.35): P(signal | 8/8) = 0.998, so ≈ +0.0101.
- The raw public delta therefore overstates the private effect by about 25–100%. P(θ < 0.5 | 8/8, uniform prior) = 0.5^9 ≈ 0.002, so the sign is safe.

**4. False-keep rate and winner's curse** for the rule "keep iff P(signal | b) > 0.27". With π = 0.2, θ1 = 0.9 and θ0 = 0.35, this rule becomes "keep iff b ≥ 6 of 8".
- P(false keep) for each null push: 0.050 at θ0 = 0.4, 0.011 at θ0 = 0.3.
- Over about 11 null pushes: expected false keeps 0.55, so P(≥ 1) ≈ 0.43 at θ0 = 0.4; or 0.12 and 0.12 at θ0 = 0.3.
- Miss rate for a true signal: P(b ≤ 5 | θ = 0.9) = 0.038.
- Each false keep costs about −0.0025 private (n' = 8, θ0 = 0.4), against about +0.0101 for a true keep. The expected false-keep cost (about −0.0014) is small next to the expected true-keep gain.
- Winner's curse among null pushes that pass the rule: E[b | b ≥ 6] ≈ 6.18, so naive θ̂ ≈ 0.77 and a naive private estimate of +0.0068, while the truth is −0.0025.

**5. Risk attitude for the final upload.** The objective is rank, not expected score, so a push with intermediate posterior (P between 0.27 and about 0.6) is a variance choice.
- If the team needs to move up, positive-EV variance is acceptable.
- If ahead, prefer only high-posterior pushes.
- This mirrors the user's existing ct9/ct10/ct13 framing (memory: "P(ct10 > ct9 on Private) 0.5–0.85").

**6. Final-selection discipline adapted from Kaggle practice and the Ladder.**
- Every change in the final submission must be justified by a model or probe whose private-side expectation is positive under the hierarchical model.
- Changes justified only by a public-score bump of a whole-model swap get a Ladder-like margin: require the decoded improvement to exceed about 2 sd of its sampling noise under random public/private assignment.
- Never include per-question corrections that apply only to public questions. They are harmless to private but are pure public overfit and distort comparisons.

### Gaps
- I did not find a quantitative study of public-vs-private rank correlation for top submissions. Roelofs reports score-level agreement, and the rank correlation itself was not extracted.
- I did not fetch Kaggle forum posts on "trust CV vs trust public LB" final selection (Medium 403, Kaggle not searched), so no sourced community rule of thumb is given.
- How AI CUP splits public and private (random or not) is unknown. It is the single biggest determinant of whether a public-confirmed shift transfers (per Roelofs, non-i.i.d. splits caused the outliers).

## Q5. Decoding individual test labels from leaderboard scores: methods, ethics and rules

### Takeaway
Decoding individual public labels is well documented (Whitehill; Coursera's "extract public ground truth"). It is generally seen as harmless to the final ranking but against the spirit of benchmarks, and hosts and researchers recommend scoring only on never-queried data. Using aggregate decoded counts of one's own changed questions to test population-level hypotheses is the "consistent category / consistent distribution" type of probing, which has decided Kaggle wins. Multiple accounts are the line that is clearly prohibited.

### Cited Findings
- Whitehill frames his work as a warning about leaderboard integrity and argues that competitions should rank contestants only on examples the oracle never examined. In his case the first stage was mostly informational, but he notes that a temporary high rank could still carry value, for example to recruiters. — [ar5iv 1707.01825](https://ar5iv.labs.arxiv.org/html/1707.01825)
- The Coursera lecture calls extraction of public ground truth "usually pretty harmless". It calls consistent-category probing "probably the most annoying type of data leak", one that was "a key for winning" in Red Hat and West Nile. — [lecture transcript](https://zfelleg.useribm.hu/videos/courses/specializations/advanced_machine_learning/competitive-data-science/06_data-leakages/01_data-leakages/02_leaderboard-probing-and-examples-of-rare-data-leaks.en.txt)
- Zheng recommends daily submission limits, coarse score precision, and disqualifying participants who use multiple accounts, because multiple accounts break the Ladder's guarantees. — [ar5iv 1510.03349](https://ar5iv.labs.arxiv.org/html/1510.03349)

### Inferences
- The team's practice:
  - It uses one account.
  - It decodes only aggregate (X, Y) for its own changed questions.
  - It acts only on population-level boundary hypotheses.
  - It does not use decoded public labels on the private half, apart from as evidence for a shared shift.

  This is in the category the literature treats as legitimate within the rules, unless the competition rules forbid probing explicitly.
- The two-slot design in Q2 deliberately uploads submissions known to be worse (two-level or backward moves). That is still ordinary probing within the upload limit, but it is worth confirming that the AI CUP rules do not penalise or flag deliberately degraded uploads.

### Gaps
- I did not retrieve or check the AI CUP 2026 rule text on leaderboard probing, deliberately degraded submissions, or multiple accounts.
- I found no formal ethics paper beyond Whitehill's discussion and the platform-design papers (Blum & Hardt; Zheng).

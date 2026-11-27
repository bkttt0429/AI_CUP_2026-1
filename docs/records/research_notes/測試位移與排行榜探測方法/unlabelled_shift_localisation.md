# Detecting and localising class-conditional / concept shift from classifier posteriors (unlabelled test, 10-class ordinal, n ≈ 800)

Legend: sources marked **[fetched]** were opened or confirmed in this session (abstract page, full-text HTML or a search-result summary of the primary page). Sources marked **[mem]** are cited from prior knowledge with their canonical URL. They were not re-fetched, so treat their exact numbers and wording as unverified. Everything under "Inferences" is my own reasoning applied to the AI CUP Go setting: 10 ordered classes D, C, B, A, 1D–6D; about 33k validation questions with out-of-fold (OOF) posteriors; 800 test questions; 8/8 probe labels at the A|1D boundary.

---

## Q1. Label-shift estimators (BBSE, RLLS, MLLS/EM + bias-corrected calibration, Garg's unified view) and how they fail when label shift is violated

### Takeaway
All four estimators assume p(x|y) is unchanged, and they estimate only the class composition. When one class's conditional moves toward a neighbour (test A players looking 1D-like), they absorb the move into the composition. The composition estimate then shifts in the opposite direction to what you need: more 1D, less A. Applying that prior correction would push decisions further toward 1D. These estimators are therefore unsuitable for localisation. Their useful role is as a null model: fit the best label-shift explanation, then look for where the test posterior distribution still does not fit.

### Cited Findings
- **BBSE** (Lipton, Wang & Smola, ICML 2018). It estimates importance weights w = q(y)/p(y) by solving C·w = μ̂_test, where C is the joint confusion matrix of a black-box predictor on held-out source data and μ̂_test is the predicted-label frequency on the test set. It needs label shift and an invertible C. Its error bound grows as 1/σ_min(C) and shrinks as 1/√n_test. — [Lipton et al. 2018](https://arxiv.org/abs/1802.03916) [mem]
- **RLLS** (Azizzadenesheli et al., ICLR 2019). It regularises the BBSE solution toward w = 1, meaning no shift, with a strength set by the sample sizes. It is aimed at the small-n_test regime where BBSE's inverted confusion matrix gives high-variance weights. — [Azizzadenesheli et al. 2019](https://arxiv.org/abs/1903.09734) [mem]
- **Saerens–Latinne–Decaestecker EM (MLLS)**. It iterates a re-estimate of the test prior and a Bayes-rule re-weighting of the posteriors. It assumes the posteriors are calibrated for the source prior. — [Saerens et al. 2002, Neural Computation](https://doi.org/10.1162/089976602753284446) [mem]
- **Alexandari, Kundaje & Shrikumar (ICML 2020)**:
  - They note that EM assumes calibrated probabilities, "which is not true of modern neural networks".
  - Maximum likelihood with bias-corrected calibration "outperforms both BBSL and RLLS" across datasets and shifts.
  - The ML objective is concave.
  - They give "a principled way to estimate source-domain priors" that makes the method more robust to poor calibration.
  - They call ML + calibration "a formidable and efficient baseline".
  - Code: https://github.com/kundajelab/labelshiftexperiments
  — [Alexandari et al. 2020](https://arxiv.org/abs/1901.06852) [fetched]. The specific calibrators in the full paper are BCTS (temperature scaling plus class-wise bias terms) and vector scaling. — [same paper](https://arxiv.org/abs/1901.06852) [mem; the abstract does not name them]
- **Garg, Wu, Balakrishnan & Lipton (NeurIPS 2020), unified view**:
  - BBSE is "a moment-matching approach based on confusion matrices", provably consistent with interpretable error bounds.
  - MLLS "dominates empirically".
  - MLLS is consistent under "calibration of the classifier and a confusion matrix invertibility condition that BBSE also requires".
  - BBSE is "roughly equivalent to MLLS for a particular choice of calibration method". Its lower efficiency comes from "a loss of information due to coarse calibration".
  - MLLS's finite-sample error splits into "terms reflecting miscalibration and estimation error".
  — [Garg et al. 2020](https://arxiv.org/abs/2003.07554) [fetched]
- **Behaviour outside label shift, as a detector**: BBSD (shift detection with a classifier's outputs) "works surprisingly well under a broad set of shifts, even when the label shift assumption is not met". The formal justification, however, requires label shift and an invertible confusion matrix. — [Rabanser, Günnemann & Lipton, NeurIPS 2019](https://ar5iv.labs.arxiv.org/html/1810.11953) [fetched]
- **Behaviour outside label shift, as a corrector (RLSbench, ICML 2023; more than 500 shift pairs from 14 datasets, with label-proportion shift and conditional shift at the same time)**:
  - Popular domain-adaptation methods often fall below a source-only classifier when target label proportions shift severely.
  - A meta-algorithm improves results by 2–10 accuracy points under large proportion shifts and has "<0.5%" effect when proportions do not shift. It resamples or pseudo-balances, then re-weights the final classifier with an estimated target label marginal.
  - Methods designed for relaxed label shift (IW-CDANN, SENTRY) "often fail to improve over a source-only classifier".
  — [Garg et al. 2023, RLSbench](https://arxiv.org/abs/2302.03020) [fetched]

### Inferences
- **Why plain EM failed even on validation folds.** Garg 2020 and Alexandari 2020 make calibration a precondition for MLLS. Our per-question posteriors break that precondition in two ways:
  - They come from a model trained class-balanced, so the implied source prior is uniform, not the validation composition. EM must start from the training prior, and Alexandari's source-prior estimation addresses exactly this mismatch.
  - They pool 5–20 games per player and are likely over-confident.
  The recommended repair has three steps:
  1. Fit BCTS or vector scaling on OOF folds, at question level and against the training prior.
  2. Check calibration per boundary, i.e. reliability of s = log P(j)/P(i).
  3. Only then run MLLS.
  BBSE's unstable shares are what a 10×10 confusion matrix with heavy adjacent-class confusion gives you: a small σ_min amplifies noise. RLLS shrinkage, per-boundary 2×2 problems, or an ordinal smoothness penalty (Q5, Q7) are the standard remedies.
- **Direction of the bias under our shift.** Suppose test A players move toward 1D in s = log P(A)/P(1D). A label-shift estimator sees too much mass on the 1D side and reports a higher 1D share. Correcting with q(y)/p(y) then raises 1D posteriors, which is the opposite of the probe evidence (8/8 truly A). Any automatic label-shift correction at this boundary should be considered harmful unless δ is modelled explicitly.
- **Constructive use: the label-shift residual scan.**
  1. Fit MLLS on the calibrated posteriors to get the best composition-only explanation of the test.
  2. Re-weight the validation questions by ŵ(y), which is possible because validation labels are known.
  3. Compare the re-weighted validation distribution of each boundary statistic s_k (or of the full posterior vector) with the test distribution, region by region.
  Residual mismatch is evidence against label shift, localised to the region where it appears. This is the batch analogue of Vovk's label/concept decomposition (Q3), and it generalises the existing per-boundary scan, because composition is fitted once, globally, with all 10 classes constraining it.

### Gaps
- The full-text numbers of Alexandari 2020 were not fetched: sample sizes at which BCTS+EM beats BBSE and how far it degrades under conditional shift. Neither were the exact finite-sample bounds of Garg 2020. No paper found gives label-shift estimator error as a function of *how much* class-conditional shift is present.

---

## Q2. Generalized label shift, conditional shift, sparse joint shift: what is identifiable?

### Takeaway
With only unlabelled test data, a change in the class composition and a change in a class-conditional distribution cannot be separated in general. Separation needs a structural restriction: location-scale conditionals (Zhang 2013), invariant representations (GLS), or sparsity in which features shift jointly with y (SJS). Even then, Tasche shows that conditioning on a sufficient classifier output does not restore identifiability under SJS. The per-boundary model π·f_A(s − δ) + (1 − π)·f_1D(s) is identified only through the *shape* of the KDEs. The probe labels are what break the π–δ confounding.

### Cited Findings
- Zhang, Schölkopf, Muandet & Wang (ICML 2013) separate three regimes:
  - **target shift**: "the marginal P_Y changes, while the conditional P_X|Y stays the same";
  - **conditional shift**: "the marginal P_Y is fixed, while the conditional P_X|Y changes with certain constraints";
  - **generalized target shift**: both change, the conditional "with constraints".
  They estimate weights or transformations by kernel embedding of marginal and conditional distributions so as to reproduce the test covariate distribution. — [Zhang et al. 2013](http://proceedings.mlr.press/v28/zhang13d.html) [fetched]
- The constraint in Zhang 2013 is a class-wise location-scale transformation of P(X|Y), under which the target joint is identifiable given linear-independence conditions. — [Zhang et al. 2013](http://proceedings.mlr.press/v28/zhang13d.html) [mem; the abstract does not state it]
- Generalized label shift (Tachet des Combes, Zhao, Wang & Gordon, NeurIPS 2020) assumes P_S(Z|Y) = P_T(Z|Y) for a learned representation Z, i.e. label shift holds in representation space. It gives target-error bounds and importance-weighted adversarial and optimal-transport variants of domain adaptation. — [Tachet des Combes et al. 2020](https://arxiv.org/abs/2003.04475) [mem]
- Sparse joint shift (Chen, Zaharia & Zou, NeurIPS 2022) "considers the joint shift of both labels and a few features". It "unifies and generalizes ... label shift and sparse covariate shift". The paper gives "mathematical conditions under which SJS is identifiable", and its SEES framework estimates performance without labels. It reports "up to an order of magnitude" lower shift-estimation error than existing approaches. — [Chen et al. 2022](https://arxiv.org/abs/2209.08436) [fetched]
- Tasche (2023/2024), "Sparse joint shift in multinomial classification":
  - Theorem 4.7 gives a global identifiability condition: a full-rank random matrix built from the source distribution. It generalises the confusion-matrix approach of prior-shift estimation.
  - Chen et al.'s condition works only pointwise.
  - "sufficiency of the classifiers does not help achieve identifiability under SJS".
  - The paper gives a conditional posterior-correction formula and critiques inconsistencies in existing SJS estimation algorithms.
  - Published as DOI 10.3934/fods.2024012.
  — [Tasche, arXiv 2303.16971](https://arxiv.org/abs/2303.16971) [fetched]
- Moreno-Torres et al. give the standard taxonomy: covariate shift, prior-probability shift, and concept shift, which is a change in P(Y|X) or P(X|Y) depending on causal direction. — [Moreno-Torres et al. 2012, Pattern Recognition](https://doi.org/10.1016/j.patcog.2011.06.019) [mem]
- Gama et al. separate "real" concept drift (P(y|X) changes, with or without a change in P(X)) from "virtual" drift (only P(X) changes). — [Gama et al. 2014, ACM CSUR](https://doi.org/10.1145/2523813) [mem]

### Inferences
- **A hard limit.** If P_T(X) = P_V(X) and only P(Y|X) changed, the test posterior vectors f(X) have exactly the validation distribution. No unlabelled method can then detect anything, which is the "real drift without virtual drift" case. A significant result from the mixture scan (LR 3.64, p = 0.003) therefore means P(s) itself differs locally in the A|1D region; "no global covariate shift" only reflects low global power (see Q3). The direction is not identified by unlabelled data alone: "A players became 1D-like" versus "a cluster of 1D-like inputs appeared and its labels are A". The probes supply it.
- **Confounding in the 1-D, two-component model.** If f_A and f_1D have similar shapes on the log-odds scale, a joint (π, δ) move toward the other class's mode is nearly flat in likelihood. The profile-LR confidence region for (π, δ) will be banana-shaped. Report it as a joint region, not as a δ estimate at the profiled π̂. Identification comes only from tail and shape differences in the KDEs, and those are fragile with about 150 routed points.
- **Tasche's sufficiency result applies directly.** Our "shifting feature" is the classifier's own log-odds s, a sufficient statistic of the classifier. Tasche's result warns that this does not buy identifiability under SJS. Using additional, non-sufficient features alongside s should help: raw game-level statistics, per-game posterior dispersion within a question, or the number of games. They act as extra "views" that constrain the composition separately from the conditional shift (cf. the rank condition of Theorem 4.7).
- **Our scan as a constrained GeTarS.** The δ-on-one-class model is a Zhang-2013-style generalized target shift restricted to one class and one location parameter on s. A natural next step is location *and* scale (δ, log σ) on s_A, with a likelihood-ratio test of scale = 1. A scale change, such as a broader A population in the test, can mimic a composition change at a boundary.

### Gaps
- I did not obtain the explicit SJS identifiability conditions of Chen et al. 2022 or the reported performance of SEES on small target samples (n < 1000).
- I found no paper that gives identifiability or power results for one-class location shifts in 1-D log-odds mixtures at n ≈ 150.

---

## Q3. Methods that localise *where* the shift is (which classes or boundaries, which direction)

### Takeaway
Global two-sample tests on 800 points are badly underpowered for class-prior-like shifts. Rabanser et al. find knock-out shifts detected 40% of the time at n = 1000. Localisation therefore needs targeted, region-restricted statistics. The best candidates are:
- (i) the label-shift residual scan of Q1;
- (ii) a cross-fitted domain (validation-vs-test) classifier on posterior features, with its outputs aggregated by predicted class or boundary;
- (iii) conditional-distribution score tests (Kulinski et al.) on the posterior vector.

Conformal martingales (Vovk) and harmful-shift monitors (Podkopaev & Ramdas) need streaming *labels*, so they apply only to the 8 probe labels.

### Cited Findings
- **Rabanser et al., "Failing Loudly"**:
  - The best univariate detector is BBSDs: Kolmogorov–Smirnov tests on each softmax dimension with Bonferroni correction.
  - Detection accuracy, averaged over shifts, at n = 10/20/50/100/200/500/1000/10000 is 0.19/0.28/0.47/0.47/0.51/0.65/0.70/0.79.
  - By shift size at n = 100/500/1000/10000: small shifts 0.14/0.36/0.40/0.54, medium 0.38/0.57/0.66/0.76, large 0.82/0.92/0.96/1.00.
  - Class-prior knock-out shifts (part of class 0 removed) are detected 0.07 at n = 100, 0.33 at n = 500, 0.40 at n = 1000 and 0.70 at n = 10000. They are "hard to detect even with many samples".
  - Chi-squared on hard labels (BBSDh) reaches 0.46 at n = 1000.
  - MMD on an untrained autoencoder reaches 0.61 at n = 1000.
  - For localisation, they use the domain classifier's most anomalous samples. In the original MNIST split, many were 6s, and a KS test on 6s gave p = 2.7·10⁻¹⁰. The domain classifier "performs poorly with 100 or fewer samples".
  — [Rabanser et al. 2019](https://ar5iv.labs.arxiv.org/html/1810.11953) [fetched]
- **Kulinski, Bagchi & Inouye (NeurIPS 2020)**:
  - They frame localisation as "multiple conditional distribution hypothesis tests", in non-parametric and parametric forms.
  - The test statistic uses the density model's score function (gradient with respect to the input), so all dimensions are computed "in a single forward and backward pass".
  - Any density model works, e.g. normalising flows or autoregressive models.
  — [Kulinski et al. 2020](https://arxiv.org/abs/2107.06929) [fetched]
- **Kulinski & Inouye (ICML 2023)**, "Towards Explaining Distribution Shifts". Explains a shift as an interpretable transport map from source to target, e.g. a k-sparse mean shift or a cluster-wise shift, trading interpretability against fidelity. — [Kulinski & Inouye 2023](https://arxiv.org/abs/2210.10275) [mem]
- **Vovk, "Testing for concept shift online" (2020)**:
  - Label-conditional conformal martingales detect concept shift (a change in P_X|Y).
  - The product of a label-conditional conformal martingale and a label conformal martingale (independent randomisation) is an exchangeability martingale (Corollary 3). It "decomposes" the evidence into a concept-shift part and a label-shift part.
  - Both parts need labels.
  — [Vovk 2020](https://arxiv.org/abs/2012.14246) [fetched]
- **Podkopaev & Ramdas (ICLR 2022)**. Sequential tests, by betting, of whether a deployed model's *risk* has increased beyond a tolerance. They alarm only on harmful shifts and need labels on the target stream. — [Podkopaev & Ramdas 2022](https://arxiv.org/abs/2110.06177) [mem]
- **Classifier two-sample tests**. Train a classifier to tell source from target; held-out accuracy above chance is the test statistic. — [Lopez-Paz & Oquab 2017](https://arxiv.org/abs/1610.06545) [mem]
- **Attribution methods (all need target labels)**:
  - DISDE (Cai, Namkoong & Yadlowsky 2023) splits a performance drop into three terms: harder but frequent examples, a change in the X→Y relationship, and rare or unseen inputs. Each term holds one distribution fixed and is estimated by re-weighting. — [Cai et al. 2023](https://arxiv.org/abs/2303.02011) [fetched]
  - Zhang, Singh, Ghassemi & Joshi (ICML 2023) attribute a performance change to shifts in individual causal-mechanism factors, such as P(Y) versus P(X|Y), using Shapley values. — [Zhang et al. 2023](https://arxiv.org/abs/2210.10769) [mem]
  - Budhathoki et al. (AISTATS 2021) attribute a change in a joint distribution to causal mechanisms with Shapley values. — [Budhathoki et al. 2021](https://arxiv.org/abs/2102.13384) [mem]

### Inferences
- **Rabanser's numbers support our "no global shift" finding.** Our shift is concentrated in the roughly 150 A|1D-routed questions out of 800. In Rabanser's terms that is a small perturbed fraction (about 20%) of a class-prior-like kind, and those are detected under 50% of the time even at n = 1000. A targeted scan (ours) is the right instrument. Global BBSD on the 10-dim softmax is not.
- **The cross-fitted domain classifier is the most direct "where" tool we lack.**
  - Inputs: posterior features (the 9 adjacent log-odds, entropy, number of games, within-question dispersion).
  - Labels: domain, validation (re-weighted by the MLLS-estimated ŵ(y)) versus test.
  - Use K-fold cross-fitting so every test question gets an out-of-fold "test-ness" probability.
  - Aggregate by predicted class and by boundary region. Then test each region's mean test-ness against a permutation null that respects the 800-vs-33k imbalance, e.g. by subsampling validation to 800.
  The ŵ(y) re-weighting turns it from a covariate-shift detector into a detector of non-label shift (the Q1 residual). Rabanser's warning applies: weak with ≤ 100 points per region. Use it to rank regions, then confirm with the mixture LR.
- **Kulinski-style score tests on the posterior vector.** Fit a density model (a Gaussian copula or small flow on the 9 log-odds) to re-weighted validation. Test, coordinate by coordinate, P_T(s_k | s_−k) = P_V(s_k | s_−k). This localises a shift to the s_A|1D coordinate *conditional on the other boundaries*. It separates "the A|1D log-odds moved" from "questions moved along the whole ordinal axis".
- **Multiplicity.** Nine adjacent boundaries were scanned. Bonferroni on the A|1D p = 0.003 gives 0.027, still significant. Three probe rules were tried (B→A, 1D→A, 5D→6D). The 8/8 result, binomial p = 0.47⁸ ≈ 0.0024, gives about 0.007 after a ×3 correction.
- **Vovk and Podkopaev–Ramdas need labels and so do not help unlabelled detection.** They do give a principled way to combine the 8 probe labels with future probes as an anytime-valid e-value, without optional-stopping bias.

### Gaps
- No quantitative small-sample (n ≈ 100–200) results for Kulinski's score tests or Kulinski & Inouye's explanation maps were retrieved.
- I found no paper applying feature-shift localisation to the coordinates of a classifier's *posterior* vector. That application is my extrapolation.

---

## Q4. Estimating performance without labels (ATC, DoC, agreement-on-the-line, class-specific variants): can they be made boundary-specific?

### Takeaway
These methods estimate *how much* accuracy drops, under assumptions about how confidence transfers. They do not say in which direction to move a decision. ATC's own impossibility theorem says they are blind to the case we care about: concept shift that leaves the posterior distribution nearly unchanged. Class-specific variants (MOVAL) and agreement across model families can be restricted to a boundary region, and are cheap secondary diagnostics. They should not replace the mixture or probe evidence.

### Cited Findings
- **ATC (Garg et al., ICLR 2022)**:
  - It "learns a threshold on the model's confidence" and predicts accuracy as "the fraction of unlabeled examples for which model confidence exceeds that threshold".
  - It is "2–4×" more accurate than prior methods across WILDS, ImageNet, BREEDS, CIFAR and MNIST.
  - The authors prove that in general "identifying the accuracy is just as hard as identifying the optimal predictor", so "the efficacy of any method rests upon (perhaps unstated) assumptions on the nature of the shift".
  — [Garg et al. 2022](https://arxiv.org/abs/2201.04234) [fetched]
- **DoC, difference of confidences (Guillory et al., ICCV 2021)**. Predicts the accuracy drop as the drop in average confidence between source and target. — [Guillory et al. 2021](https://arxiv.org/abs/2107.03315) [mem]
- **Agreement-on-the-line (Baek et al., NeurIPS 2022)**. When accuracy-on-the-line holds across a family of models, in-distribution versus OOD *agreement* between model pairs follows a line with the same slope and intercept, on a probit scale, as accuracy. That gives a label-free estimate of OOD accuracy. It needs many diverse models and fails where accuracy-on-the-line fails. — [Baek et al. 2022](https://arxiv.org/abs/2206.13089) [mem]
- **Class-specific confidence methods (Li, Kamnitsas et al., MICCAI 2022)**:
  - Global calibration only matches overall confidence to overall accuracy and "fail[s] to estimate class-wise accuracy" under imbalance.
  - Class-specific calibration parameters give class-wise TS, DoC and ATC.
  - Estimation error is 18% lower for skin-lesion classification under natural shift, and accuracy estimates are twice as good for segmentation.
  - Code: the MOVAL package, https://github.com/ZerojumpLine/MOVAL
  — [Li et al. 2022](https://arxiv.org/abs/2207.09957) [fetched]
- **COT (Lu et al., NeurIPS 2023)**. Estimates OOD error as the optimal-transport distance between the test softmax outputs and the source label marginal. It is designed to be robust to label shift. — [Lu et al. 2023](https://arxiv.org/abs/2305.15640) [mem]
- **Mandoline (Chen, Goel, Sohoni et al., ICML 2021)**. Re-weights labelled source data with density ratios estimated over user-defined "slicing functions" to estimate target performance. The slices make the estimate interpretable by region. — [Chen et al. 2021](https://arxiv.org/abs/2107.00643) [mem]

### Inferences
- **A boundary-specific ATC/DoC is straightforward but uninformative here.**
  - Restrict to questions routed to boundary k, and take as "confidence" P(argmax) or the decision margin.
  - Under the observed shift, test A questions look 1D-like, so their s-distribution resembles that of true 1D questions. Region-level confidence barely changes, so ATC/DoC would predict roughly validation-level accuracy, while the probes show precision went from 0.47 to 1.0 *in the other direction*.
  - This is the impossibility theorem in action: these tools measure confidence drift, not label drift.
- **Agreement across model families is the one variant that could see something unlabelled.** Several families or seeds already exist (ct3, ct9 and the M-series). Compute pairwise disagreement *per boundary region* on test versus validation. Under pure label shift, region-conditional disagreement should match validation. A concept shift that moves A players onto the 1D side would plausibly change which models flip at that boundary. A region whose disagreement rate deviates from the validation fit is a candidate for shift. This is cheap (posteriors only), but it is my extrapolation of AoL, not a tested method.
- **Mandoline's slicing view** matches the per-boundary design. Slices = routed boundary regions. Density ratios estimated within slices reproduce our KDE-based approach.

### Gaps
- No ATC, DoC or AoL paper evaluated *ordinal adjacent-class* concept shift. None reports how accurate their estimates are with fewer than about 1000 target points per region.

---

## Q5. Mixture-proportion estimation and quantification per class pair: methods, software, power at ~150 points

### Takeaway
Our per-boundary KDE mixture fit is a distribution-matching quantifier (HDy/DyS/KDE family) extended with a shift parameter. Quantification theory gives two levers:
- variance formulas: adjusted classify-and-count variance scales as 1/(n·(tpr − fpr)²), which is about ±0.2 on a composition at n = 150;
- ordinal smoothness regularisation (Bunse et al.), which directly targets BBSE's unstable shares.

Mixture-proportion estimation (MPE) adds the "irreducibility" condition, which is exactly what fails when adjacent ranks overlap. Bayesian quantification (Ziegler & Czyż) gives the cleanest route to honest uncertainty and to adding a shift parameter and probe labels.

### Cited Findings
- **Reviews**. Quantification learning is the estimation of class prevalences in unlabelled samples, with families: classify-and-count, adjusted CC, probabilistic CC, distribution matching (HDy, etc.), EM. — [González, Castaño, Chawla & del Coz 2017, ACM CSUR](https://doi.org/10.1145/3117807) [mem]; book-length treatment in [Esuli, Fabris, Moreo & Sebastiani 2023, "Learning to Quantify" (Springer, open access)](https://doi.org/10.1007/978-3-031-20467-8) [mem]
- **Forman's adjusted classify-and-count (ACC)**: p̂ = (q̂ − fpr)/(tpr − fpr), with "threshold selection" variants (X, Max, T50, median sweep) for when tpr − fpr is small. — [Forman 2008, DMKD](https://doi.org/10.1007/s10618-008-0097-y) [mem]
- **Hopkins & King** estimate category proportions by solving P(S) = P(S|D)·P(D) over word-stem profiles S. Their key assumption is that P(S|D) is equal in the labelled and target sets. — [Hopkins & King 2010, AJPS](https://doi.org/10.1111/j.1540-5907.2009.00428.x) [mem]
- **Elkan & Noto** estimate the PU labelling constant c = P(s=1|y=1) as the mean classifier score on labelled positives. — [Elkan & Noto 2008, KDD](https://doi.org/10.1145/1401890.1401920) [mem]
- **Ratio estimator** (Vaz, Izbicki & Stern, JMLR 2019). A unifying view of ACC-type estimators as ratio estimators under prior-probability shift, with asymptotic variance and consistency results. — [Vaz et al. 2019](https://jmlr.org/papers/v20/18-456.html) [mem]
- **Mixture proportion estimation**:
  - Kernel-embedding MPE (Ramaswamy, Scott & Tewari, ICML 2016) — [arXiv 1603.02501](https://arxiv.org/abs/1603.02501) [mem]
  - "Best Bin Estimation" (BBE) for PU learning from classifier outputs (Garg et al., NeurIPS 2021) — [arXiv 2111.00980](https://arxiv.org/abs/2111.00980) [mem]
  - Both need an *irreducibility* or "top bin" condition: some region where the component of interest has (nearly) no mass from the other. — same sources [mem]
- **QuaPy** (HLT-ISTI, Pisa) offers:
  - "Classify-&-Count and its variants", "Expectation Maximization", HDy, ACC, "quantification methods based on structured output learning", QuaNet and ensembles;
  - "uncertainty quantification via bootstrap-based and Bayesian methods, including confidence intervals and simplex-aware confidence regions";
  - sampling protocols ("APP, NPP");
  - model selection that minimises quantification losses.
  — [QuaPy GitHub](https://github.com/HLT-ISTI/QuaPy) [fetched]
- **Ordinal quantification** (Bunse, Moreo, Sebastiani & Senz):
  - Three, later a class of, ordinal quantification algorithms that regularise existing methods to prevent "ordinally implausible estimates, assuming that ordinal distributions tend to be smooth in practice".
  - They outperform existing algorithms when that assumption holds. The assumption was "only informally verified".
  - Published in DMKD 2024; the conference version is from 2022.
  — [Bunse et al., arXiv 2310.09210](https://arxiv.org/abs/2310.09210) [fetched]
- **Bayesian quantification with black-box estimators** (Ziegler & Czyż, TMLR 2024):
  - ACC, BBSE and invariant-ratio estimators are related to inference in a probabilistic graphical model that approximates the generative process.
  - They propose a Bayesian estimator that is "competitive, and in some cases superior" to point estimators, with a condition under which the MAP approximately converges to the truth at large n.
  - It replaces P(X|Y) with a low-dimensional black-box-classifier approximation, unlike Storkey 2009.
  - Code: https://github.com/pawel-czyz/labelshift
  — [Ziegler & Czyż 2023/2024](https://arxiv.org/abs/2302.09159) [fetched]

### Inferences
- **Power arithmetic for one boundary.** By the delta method, Var(p̂_ACC) ≈ q(1 − q) / (n·(tpr − fpr)²).
  - n = 150, q = 0.5, tpr − fpr = 0.4: SE ≈ 0.10, so the 95% interval is about ±0.20 on the A-vs-1D composition.
  - At tpr − fpr = 0.6: SE ≈ 0.07.
  - A full-distribution (KDE) likelihood does better than ACC's one threshold. But freeing δ as well multiplies the variance through the π–δ correlation (Q2).
  - Practical check: take the parametric-bootstrap null you already have, add *alternatives* (δ = 0.25, 0.5, 1.0 nats at the probe-implied composition), and report power curves per boundary. Then "p ≥ 0.16 elsewhere" can be read as "shift of size ≥ X excluded" rather than "nothing found".
- **Irreducibility is violated at ordinal boundaries.** The A and 1D classes overlap on both sides in s. MPE-style estimators (BBE, Elkan–Noto) that rely on a "pure" top bin will be biased toward the source composition. KDE matching with a parametric δ is the better choice, which is our existing design.
- **Fix for unstable 10-class BBSE shares.**
  1. Solve per boundary (2×2 ACC/HDy) or apply Bunse's smoothness penalty to the 10-vector.
  2. Use QuaPy's bootstrap or Bayesian confidence regions to report the uncertainty.
  3. Use Ziegler–Czyż's graphical model as the template for a joint Bayesian model with (π_1…π_10, δ_k).
  QuaPy works on pre-computed posteriors via its aggregative-method interface (my assumption; not confirmed from the README; see Gaps).

### Gaps
- The QuaPy README did not name its distribution-matching methods (DyS, SMM, KDEy) or explain how to plug in pre-computed posteriors. I could not verify whether the KDE-based multiclass quantifier "KDEy" (Moreo et al. 2024) is included. If it is, it is the multiclass generalisation of our per-boundary KDE mixture.
- I found no published power table for KDE-mixture quantification at n ≈ 150 with overlapping components.

---

## Q6. Using a handful of known test labels (8 probed A questions) to estimate the class-conditional shift and how much to adjust

### Takeaway
Combine the unlabelled-mixture likelihood and the probe labels in one model. Because the probes were selected by a rule on s, the labelled term P(y | s) is ignorable for selection (missing at random given s). With 8 events, only a one-parameter "recalibration in the large" per boundary is supportable. The clinical-prediction updating literature formalises this with a closed testing procedure that escalates from intercept-only to slope to full revision only when the likelihood ratio demands it. The probe data alone bound the A|1D log-odds shift at ≥ about 1.05 nats (95% one-sided) inside the probed band. They say nothing about outside it.

### Cited Findings
- **Vergouwe et al. (Statistics in Medicine, 2017; DOI 10.1002/sim.7179)**:
  - Update methods in increasing order: "recalibration in the large" (intercept only), recalibration (intercept + slope), model revision (all coefficients).
  - A closed testing procedure of likelihood-ratio tests against the original model keeps type-I error near nominal.
  - In their example it chose intercept-only "without overfitting", whereas full revision was significant but would have overfitted.
  — [Vergouwe et al. 2017](https://research.utwente.nl/en/publications/a-closed-testing-procedure-to-select-an-appropriate-method-for-up/) [fetched]
- Follow-up work in dynamic updating found recalibration strategies "the least sensitive" to update interval and window length. — [Schnellinger, Yang & Kimmel 2021](https://doi.org/10.1186/s41512-021-00110-w) [fetched]
- Bayesian quantification turns black-box label-shift estimators into a graphical model with a posterior over prevalences. — [Ziegler & Czyż](https://arxiv.org/abs/2302.09159) [fetched]
- Label-conditional conformal martingales accumulate concept-shift evidence from labelled observations as anytime-valid evidence. — [Vovk 2020](https://arxiv.org/abs/2012.14246) [fetched]
- Active testing selects which target points to label so as to estimate model performance with few labels, correcting the selection bias with importance weights. — [Kossen et al., ICML 2021](https://arxiv.org/abs/2103.05331) [mem]

### Inferences
- **Probe arithmetic.**
  - Validation precision of the rule (predicted 1D, P(A)/P(1D) ≥ 1/2) is 0.47. Test result: 8/8.
  - Under H0 (precision = 0.47), P(8/8) = 0.47⁸ ≈ 0.0024.
  - With a uniform Beta(1,1) prior, the posterior is Beta(9,1): mean 0.90, 5% quantile 0.05^(1/9) ≈ 0.72.
  - Converted to an odds multiplier on A:1D inside the band: k = p′(1 − p)/(p(1 − p′)). At p′ = 0.72, k ≈ 2.9, i.e. a log-odds shift Δ ≥ about 1.05 nats (95% one-sided). At the posterior mean p′ = 0.90, k ≈ 10, Δ ≈ 2.3.
  - 8/8 gives no upper bound. The estimate is entirely prior-driven at the top, so do not extrapolate Δ to s-values below the probed band (P(A)/P(1D) < 1/2) without the unlabelled mixture.
- **Joint model to fit.** For questions routed to the A|1D boundary:
  - Unlabelled term: L_U(π, δ) = Π_i [π f_A^V(s_i − δ) + (1 − π) f_1D^V(s_i)].
  - Labelled term: L_L(π, δ) = Π_probes P(y = A | s_j; π, δ) = Π_j π f_A^V(s_j − δ) / [π f_A^V(s_j − δ) + (1 − π) f_1D^V(s_j)].
  - Maximise or sample L_U·L_L, with a weak prior on δ and a hierarchical prior shrinking the δ_k of other boundaries toward 0.
  The probe term breaks the π–δ ridge of Q2, because it measures P(A|s) directly where the ridge is flattest.
  Then decide by expected score under the *posterior-averaged* posteriors, rather than plugging in a single δ̂. This shrinks flips for questions far from the probed band.
- **Decision threshold.** With the official score (1 for exact, e⁻¹ for adjacent), E[score | predict A] − E[score | predict 1D] = (1 − e⁻¹)(P(A) − P(1D)) + e⁻¹(P(B) − P(2D)). A log-odds shift Δ on s_A|1D enters only through the first term. The flip boundary therefore moves from s ≈ 0 (adjusted for the B/2D neighbours) to s ≈ −Δ. Write the flip rule in this form so that the neighbour term is not lost.
- **Selection ignorability.** Probe questions were selected on s, a function of the inputs, not on y. So P(y | s, selected) = P(y | s), and the labelled term is valid without modelling the selection. Probing *rules* chosen after looking at public-LB results would add multiplicity, but not bias to P(y|s).
- **Hierarchy of updates (Vergouwe-style).** Candidate models:
  - M0: no update.
  - M1: δ on A|1D only. This is the current choice, like the ct13 bias-only push.
  - M2: δ on A|1D plus a scale change on s_A.
  - M3: δ_k on all boundaries.
  With 8 labels plus about 150 unlabelled routed points, the closed testing logic will almost surely stop at M1. That is consistent with "intercept-only" being the most robust update with few events.

### Gaps
- I found no paper on *combining unlabelled mixture likelihood with a few selection-on-score target labels* for class-conditional shift specifically. The joint model above is assembled from standard pieces, not taken from a published method.
- The few-shot recalibration literature (e.g. for language models) was not retrieved.

---

## Q7. Ordinal-specific considerations: threshold shift in cumulative-link models and threshold re-estimation

### Takeaway
An ordinal view separates two hypotheses:
- (a) a global latent offset: the whole test population is stronger or weaker on the latent rating scale, so every cut-point moves together in latent units;
- (b) a single cut-point shift: the definition or measurement of the A|1D boundary differs.

The evidence so far favours (b): a significant δ only at A|1D, failed probes at B→A and 5D→6D, p ≥ 0.16 elsewhere. A cumulative-link reparameterisation with per-cut-point offsets and shrinkage tests this directly. Ordinal smoothness regularisation also stabilises composition estimates.

### Cited Findings
- **Cumulative-link (proportional-odds) models**: P(Y ≤ k | x) = F(θ_k − η(x)), where cut-points θ_k are separate from the linear predictor η. A population-level change in η moves all cumulative logits equally. A change in one θ_k affects only the classes adjacent to boundary k. — [McCullagh 1980, JRSS-B](https://doi.org/10.1111/j.2517-6161.1980.tb01109.x) [mem]
- **Ordinal quantification**: regularisation against ordinally implausible (non-smooth) prevalence vectors improves estimates when real ordinal distributions are smooth. — [Bunse et al.](https://arxiv.org/abs/2310.09210) [fetched]
- **Intercept-only recalibration** is the minimal, most robust update when few target events are available. — [Vergouwe et al. 2017](https://research.utwente.nl/en/publications/a-closed-testing-procedure-to-select-an-appropriate-method-for-up/) [fetched]

### Inferences
- **Cheap refit: per-boundary offsets on frozen posteriors.**
  1. Map each question's 10-class posterior to 9 cumulative logits ℓ_k = logit P(Y > k).
  2. Model the test as ℓ_k^T = ℓ_k^V + c + δ_k, with c a global offset and δ_k sparse, e.g. a horseshoe or Laplace prior.
  3. Fit by the unlabelled mixture likelihood plus probe labels (Q6).
  Then test H_a: δ = 0 (global offset only) against H_b: c = 0 with δ_A|1D ≠ 0.
  - Under (a), the narrow dan bands (about 1000 rating points each, versus about 3000 for kyu bands) should show shift *everywhere* in the dan range, especially 5D→6D. The failed 5D→6D probe and null scans argue against (a).
  - Under (b), the shift stays at A|1D.
- **Where the band widths change.** A|1D is the only boundary where the class width changes from about 3000 to about 1000 points. Any mismatch between how validation and test players were assigned to "A" versus "1D" lives exactly here: a different source-server conversion, or a test sampling rule that truncates within the A band. That gives (b) a plausible mechanism. This is a hypothesis to check against the dataset documentation, not a finding.
- **Smoothness as a prior.** In the joint model of Q6, put a Bunse-style smoothness penalty on the 10-class test composition, e.g. second differences of log π. This removes the high-variance zig-zag in BBSE's 10-class shares. A one-boundary δ then cannot be mimicked by an implausible spike in composition.

### Gaps
- I found no paper on *unsupervised re-estimation of cumulative-link thresholds under shift* from posteriors alone. Domain adaptation for ordinal regression appears thinly covered in what was searched.

---

## Q8. Synthesis: which methods could do better than the per-boundary mixture scan, ranked for our setting (posteriors only, about 800 test points, 8 labels)

### Takeaway
The current scan (validation KDEs, free composition, a δ on one class, profile LR, parametric-bootstrap null) is close to the best available unlabelled tool for a 1-D log-odds. Generic detectors (BBSD, MMD, ATC/DoC) are weaker at this n. The expected gains come from six changes:
1. Fit composition *globally* first (calibrated MLLS with ordinal smoothness), so each boundary scan has fewer free parameters.
2. Use multi-dimensional posterior features and non-sufficient side features to fight π–δ confounding (Tasche's warning).
3. Add a cross-fitted domain classifier on the label-shift-reweighted validation set to rank regions.
4. Model the probe labels jointly, with posterior-averaged decisions.
5. Report power curves, not only p-values.
6. Test "global ordinal offset" against "single cut-point" explicitly.

### Cited Findings
- Global BBSD-style tests have low power for class-prior-like shifts at n ≈ 1000: 0.40 detection for knock-out shifts. — [Rabanser et al. 2019](https://ar5iv.labs.arxiv.org/html/1810.11953) [fetched]
- MLLS dominates BBSE but needs calibration plus invertibility. — [Garg et al. 2020](https://arxiv.org/abs/2003.07554) [fetched]; ML plus bias-corrected calibration beats BBSL and RLLS — [Alexandari et al. 2020](https://arxiv.org/abs/1901.06852) [fetched]
- Under SJS, sufficiency of the classifier does not give identifiability. — [Tasche](https://arxiv.org/abs/2303.16971) [fetched]
- Accuracy estimation without labels is impossible in general without shift assumptions. — [Garg et al. 2022](https://arxiv.org/abs/2201.04234) [fetched]
- Ordinal regularisation helps quantification when distributions are smooth. — [Bunse et al.](https://arxiv.org/abs/2310.09210) [fetched]
- Bayesian quantification provides posterior uncertainty for prevalences from black-box classifiers. — [Ziegler & Czyż](https://arxiv.org/abs/2302.09159) [fetched]
- Closed-testing escalation from intercept-only updating avoids overfitting with few events. — [Vergouwe et al. 2017](https://research.utwente.nl/en/publications/a-closed-testing-procedure-to-select-an-appropriate-method-for-up/) [fetched]

### Inferences
Concrete, ordered plan. All steps are posterior-only or cheap refits, run via sbatch on compute nodes.
1. **Calibrate.** Fit BCTS or vector scaling on OOF question-level posteriors against the *training* (balanced) prior. Check per-boundary reliability of s_k.
2. **Global composition.** Run MLLS on test with a Bunse smoothness penalty and RLLS-style shrinkage toward the validation composition. Get bootstrap or Bayesian intervals (QuaPy or Ziegler–Czyż code).
3. **Label-shift residual scan.** With the composition fixed (or profiled globally), rerun the per-boundary δ scans. This has fewer nuisance parameters per boundary than the current scan, so power should rise. Report power curves for δ ∈ {0.25, 0.5, 1.0}, so that the null boundaries carry a minimum-detectable-shift statement.
4. **Ranking regions with a domain classifier.** Use cross-fitted test-vs-reweighted-validation classification on the 9 cumulative logits plus side features (number of games, within-question spread of per-game posteriors). Aggregate test-ness by boundary region and use permutation p-values. This tells you *where* to look next, including non-adjacent patterns the 1-D scan cannot see.
5. **Conditional localisation.** Run Kulinski-style per-coordinate conditional tests on the cumulative-logit vector to separate "A|1D moved" from "everything moved along the axis".
6. **Joint labelled + unlabelled fit at A|1D.** Model (π, δ, optionally a scale term) with the 8 probe labels as a P(y|s) likelihood. Decide by posterior-averaged expected score using the (1 − e⁻¹)Δ_A,1D + e⁻¹Δ_B,2D form. Restrict flips to the s-range the probes cover, or let posterior uncertainty shrink them outside it.
7. **Ordinal structure test.** Fit a global offset c against sparse δ_k on the cumulative logits. If c ≈ 0 and only δ_A|1D ≠ 0, keep the single-boundary adjustment. Do not propagate a "test population is stronger" correction to other boundaries.

### Gaps
- None of the surveyed methods has published evidence at exactly our scale (about 150 routed points per boundary, a 10-class ordinal target, posteriors pooled over 5–20 games). Gains from steps 3–6 must be checked by simulation: inject δ shifts into held-out validation folds of size 800 and measure detection and localisation rates against the current scan.
- Not retrieved: Factorizable joint shift (He et al.); KDEy (Moreo et al.); Dussap et al.'s distribution-feature-matching estimator with robustness guarantees under contamination; LeQua 2024's ordinal task. Each may be relevant but was not verified in this session.

# Statistics methods for leaderboard probing, boundary-effect selection and target-boundary correction

Scope note: this note covers statistics methods outside standard ML domain adaptation. The sibling note `ordinal_decision_adjustment.md` covers cumulative-link threshold shifts, Saerens EM, quantification and quota assignment, so those are not repeated here.

Notation used in the derivations:
- A "push" (adjustment) flips a set of n questions across one boundary, for example 1D→A.
- θ is the target precision of a flip, meaning the fraction of flipped questions whose true label is on the new side.
- Score per flip under an A↔1D-only model: +G or −G, with G = 1 − e⁻¹ ≈ 0.632 (exact 1 vs adjacent e⁻¹). The test score is the mean over 400 questions, so Δscore = G·(2k − n)/400.
- Flips whose true label is B or 2D move the score by ±e⁻¹ ≈ 0.368 instead. The numbers below ignore them, which is approximate.
- Validation flip precision p₀ ≈ 0.47 at A|1D.

All worked numbers marked **[computed]** come from a stdlib Python script run as Slurm job 724423 on nano4 (25a-hgpn002) on 2026-11-08. They are my derivations, not taken from sources.

---

## 1. Empirical Bayes and selection-bias correction (James–Stein / Fay–Herriot shrinkage, local fdr, Tweedie, conditional inference on winners, keep rules, private-half prediction)

### Takeaway
The right frame is a hierarchical (empirical-Bayes) model over the ~16 boundary effects. Each boundary's test-vs-validation precision deviation d_j is drawn from a common prior N(μ, τ²), and each probe observes d_j with binomial noise. Two things follow from this frame.
- A correctly specified Bayes/EB posterior is not biased by "we kept the best-looking probe". The raw public precision of the selected probe is biased: with 15 probes of n≈8 it overstates by ≈0.27–0.30 in precision **[computed]**.
- The private-half prediction for a kept push is the posterior predictive under that prior. For A|1D 8/8 the answer swings from "near-certain" (P(private net > 0) ≈ 0.94–0.99) to "coin flip" (≈0.45–0.53), depending on how much prior weight goes to "test = validation". That weight (τ, or π₀) is the decision-relevant unknown. It can only be estimated from the other probes.

### Cited Findings
- Efron's "Tweedie's Formula and Selection Bias" (JASA 2011, 106(496):1602–1614) uses Tweedie's formula, first reported by Robbins (1956), as an empirical-Bayes correction for selection bias: the largest of many estimates tend to overstate their true parameters — [PubMed 22505788](https://pubmed.ncbi.nlm.nih.gov/22505788/); [SUFE newsletter entry](https://academicnewsletter.sufe.edu.cn/info/445115).
- Naive estimates of the most extreme effect sizes among many can be severely biased even after a multiplicity correction. An oracle estimator dominates the naive MLE, and a resampling (bootstrap) approximation to the oracle performs well; the authors connect this to empirical Bayes — ["On Estimating Many Means, Selection Bias, and the Bootstrap", arXiv 1311.3709](https://arxiv.org/abs/1311.3709). (The author list was not visible on the abstract page I fetched, so I do not name the authors.)
- Andrews, Kitagawa & McCloskey, "Inference on Winners" (QJE 2024, 139(1):305–358; NBER w25456, 2019):
  - Choosing the best option from estimates biases its estimate upward, so conventional confidence intervals are unreliable for the winner.
  - They give confidence sets and median-unbiased estimators that are valid *conditional on the selected target*.
  - They also give hybrid conditional/projection procedures for validity on average over possible selections.
  - In their applications, the corrections change conclusions materially but still allow informative inference.
  - Sources: [RePEc QJE record](https://econpapers.repec.org/RePEc:oup:qjecon:v:139:y:2024:i:1:p:305-358.); [NBER w25456](https://www.nber.org/system/files/working_papers/w25456/revisions/w25456.rev1.pdf); [Harvard workshop page](https://appliedstatsworkshopgov3009.hsites.harvard.edu/presentations/isaiah-andrews-presents-inference-winners).
- Efron's two-groups model treats test statistics as a mixture of null and non-null densities. The local false discovery rate is the null share of the density at z, fdr(z) = π₀f₀(z)/f(z) — [arXiv 1305.6340](https://arxiv.org/pdf/1305.6340). In his rejoinder, Efron notes that a Bayesian framing centres on fdr(z) rather than tail-area Fdr — [arXiv 0808.0603](https://web3.arxiv.org/pdf/0808.0603).
- Efron separates a theoretical null from an empirical null whose parameters are estimated jointly with the rest of the model — [arXiv 1411.6144](https://arxiv.org/pdf/1411.6144).
- Criticisms of fdr estimation:
  - Rice & Spiegelhalter note that Efron's fdr estimation relies on a flexible seven-parameter smoothed density — [arXiv 0808.0599](https://pdf.arxiv.org/pdf/0808.0599).
  - Common fdr estimators under the "zero assumption" overestimate the local null rate — [arXiv 2402.08792](https://arxiv.org/html/2402.08792v3).
- Fay–Herriot (area-level small-area) model:
  - Each direct estimate y_j has sampling variance D_j, usually treated as known.
  - The BLUP is a weighted mix of the direct estimate and a regression prediction.
  - The shrinkage factor is B_j = D_j/(A + D_j), so the weight on the direct estimate is A/(A + D_j). Noisier areas are pulled harder toward the model.
  - A (the between-area variance) is estimated by moments or (RE)ML. Common estimators of A can over-shrink.
  - Sources: [arXiv 1203.5233](https://arxiv.org/pdf/1203.5233); [Hirose & Lahiri, arXiv 1701.04176](https://arxiv.org/pdf/1701.04176); [Statistics Canada Survey Methodology 2021](https://www150.statcan.gc.ca/n1/pub/12-001-x/2021002/article/00007-eng.htm).
- One preprint criticises Fay–Herriot: the shrinkage fraction is the same for every area with a given sampling variance, so the model cannot separate a truly deviant area from a noisy one. It proposes heavy-tailed (horseshoe) priors. This is an unrefereed 2026 preprint — [arXiv 2606.30659](https://arxiv.org/pdf/2606.30659).
- Online experimentation has the same winner's curse:
  - Airbnb (Lee & Shen, Airbnb Tech Blog, 29 May 2017) proposed a bias correction for the summed effect of experiments launched because they were significant — [Airbnb engineering post](https://medium.com/airbnb-engineering/selection-bias-in-online-experimentation-c3d67795cceb). The page returned 403 to me; the content is from the search snippet only.
  - A secondary summary notes that a global holdout group sidesteps the problem — [Towards Data Science summary](https://towardsdatascience.com/how-to-measure-your-teams-a-b-testing-impact-f74c6f2b4660/).
  - Deng et al. (KDD 2021) study empirical Bayes for post-selection inference in A/B tests — [Microsoft Research PDF](https://microsoft.com/en-us/research/uploads/prod/2021/06/PostSelectionKDD2021.pdf).
  - Meta's 2026 "Bayesian Hybrid Shrinkage" uses an experiment-specific "local" shrinkage factor — [arXiv 2603.12867](https://arxiv.org/abs/2603.12867).

### Inferences
**(a) Model.** Let d_j = θ_j − p₀,j be boundary j's test-minus-validation precision deviation.
- Prior: d_j ~ N(μ, τ²).
- Observation: d̂_j = k_j/n_j − p₀,j, with D_j ≈ θ(1−θ)/n_j ≈ 0.25/n_j.
- Fay–Herriot / James–Stein estimate: d̃_j = μ̂ + (1 − B_j)(d̂_j − μ̂), with B_j = D_j/(τ̂² + D_j).
- Untested boundaries get d̃ = μ̂, i.e. validation precision plus the common shift.
- This is exactly the Fay–Herriot form with "area" = boundary and "direct estimate" = probe precision. **[derivation]**

**(b) Shrinkage factors B_j [computed].** Larger B means more shrinkage toward μ̂.

| τ (between-boundary SD) | n = 5 | n = 8 | n = 10 | n = 15 |
|---|---|---|---|---|
| 0.05 | 0.95 | 0.93 | 0.91 | 0.87 |
| 0.10 | 0.83 | 0.76 | 0.71 | 0.62 |
| 0.20 | 0.56 | 0.44 | 0.38 | 0.29 |

A|1D at 8/8 (d̂ = +0.53, taking μ = 0) shrinks to a precision of 0.51 (τ = 0.05), 0.60 (τ = 0.10) or 0.77 (τ = 0.20). So unless boundaries genuinely differ by ≳0.1 in precision, 8/8 on 8 questions does not prove a large effect.

**(c) Estimating τ from your own probes (illustrative) [computed].** The two failures here are hypothetical stand-ins (2/6 and 3/8 at p₀ = 0.5). Use the real decoded counts.
- Moment estimate: τ̂² = mean(d̂² − D) = 0.073, so τ̂ ≈ 0.27, and μ̂ = +0.09.
- EB precisions: 8/8 → 0.87; 2/6 → 0.43; 3/8 → 0.44. Untested boundaries get validation +0.09.
- But SE(μ̂) = (Σ 1/(τ̂² + D_j))^(−1/2) ≈ 0.19. With three probes μ̂ is not distinguishable from 0. Do **not** use μ̂ to adopt untested pushes blindly.
- The large τ̂ is driven almost entirely by the one 8/8. With 3 points, τ̂² has huge variance. Report τ under a range (0.05–0.27), not a point.

**(d) Winner's-curse size [computed, Monte Carlo 20k reps].** True d_j ~ N(0, τ²), each measured with n flips; keep the argmax. The overstatement of the kept probe's precision is:

| K probes | τ | n | Overstatement |
|---|---|---|---|
| 3 | 0.05 | 8 | +0.14 |
| 3 | 0.10 | 8 | +0.13 |
| 15 | 0.10 | 8 | +0.27 |
| 15 | 0.10 | 15 | +0.18 |

With the correct prior, the Tweedie/James–Stein posterior mean has ≈0 selection bias (−0.000 to +0.001) in every case. This shows that Bayes estimates are immune to selection *if the prior is right*. The residual risk is prior misspecification, i.e. τ and the tail shape.

**(e) Tweedie / local fdr translation.**
- Tweedie's formula: E[μ | z] = z + σ²·(d/dz) log f(z), with f the marginal density of the observed z's. This is standard and consistent with the Efron abstract; I recalled the formula, it is not quoted from a fetched page.
- With 15–16 effects, f cannot be estimated nonparametrically: Efron's machinery is meant for hundreds to thousands of z's. The practical version is a *parametric two-groups* model:
  - The null component is "test = validation" (θ ≈ p₀ with validation-scale uncertainty, e.g. Beta(47,53)).
  - The alternative is "boundary shifted" (θ ~ U(0,1)).
  - π₀ is the prior null fraction.
- Then 1 − fdr_j is the posterior probability that boundary j is really shifted **[derivation]**.
- For A|1D 8/8 [computed]:
  - Marginal likelihoods: null 0.0032, alternative 1/9. Bayes factor = 34.8.
  - P(shift | data) = 0.97 (π₀ = 0.5), 0.90 (π₀ = 0.8), 0.65 (π₀ = 0.95).
  - With 2 failures in 3 probes, π₀ ≈ 0.6–0.8 is a defensible empirical range. The failures look more like validation than like a uniform alternative, but this should be checked with the actual counts.

**(f) Posterior predictive for the private half of A|1D (8/8 on public) [computed].** Columns give P(private net > 0) and E[Δscore] for n_priv private flips. The model assumes θ_public = θ_private (random public/private split) and A↔1D-only flips.

| Prior on θ | P(θ > 0.5) | n_priv = 8 | n_priv = 19 | n_priv = 22 |
|---|---|---|---|---|
| Beta(1,1) flat | 0.998 | 0.97 / +0.010 | 0.99 / +0.024 | 0.99 / +0.028 |
| Beta(9.4,10.6) (validation, m = 20) | 0.91 | 0.63 / +0.003 | 0.80 / +0.007 | 0.77 / +0.008 |
| Beta(23.5,26.5) (m = 50) | 0.75 | 0.47 / +0.001 | 0.63 / +0.003 | 0.57 / +0.003 |
| Beta(47,53) (m = 100, "validation is right") | 0.58 | 0.39 / +0.0002 | 0.53 / +0.0006 | 0.46 / +0.0006 |
| Two-groups π₀ = 0.8 | 0.96 | 0.91 / +0.009 | 0.95 / +0.022 | 0.94 / +0.025 |
| Two-groups π₀ = 0.95 | 0.85 | 0.77 / +0.007 | 0.83 / +0.016 | 0.80 / +0.018 |

Under every prior, P(private net < 0) stays ≤ 0.47. Expected Δscore is positive under all of them, from +0.0002 to +0.028. Keeping the A|1D push has positive expected value whatever the prior; only its size is uncertain. This is consistent with the earlier internal 0.5–0.85 range for a private A|1D bet, and it makes explicit that the spread comes from the prior on τ (or π₀). Plug in the actual private flip count of whichever submission is being judged.

**(g) Keep rule (selection-safe).** Keep a push if its posterior expected private Δ is > 0 under the hierarchical prior, with τ (or π₀) estimated from *all* probes, including failures and unpromising ones.
- Report sensitivity over τ ∈ [0.05, 0.27].
- Never compute τ from the kept probes only. That is the selection bias re-entering through the hyperparameter.
- The frequentist alternative is the Andrews–Kitagawa–McCloskey conditional median-unbiased estimate of the winner's θ. As I understand it, this inverts a truncated-normal CDF given that the winner beat the runner-up; I did not verify the details in the paper text. It is more conservative and gives very wide intervals when the winner barely wins. Here the winner (8/8) wins by a large margin, so the correction is small, and this method adds little over EB.
- Expected gain: decision clarity rather than score. Cost: 0 uploads, minutes of CPU. Risk: low, as long as τ is not fitted on survivors only.

**(h) Using the forward–backward DP posteriors.** The joint decoding gives per-question posteriors on the public set. These are "soft target labels". Feed them in as the direct estimates (k_j = Σ posterior-correct over flipped questions, with D_j from the posterior variance plus the binomial term) rather than hard counts. Questions that appear in several uploads then contribute properly weighted evidence. **[derivation]**

### Gaps
- I could not open the full text of Efron (2011) or Andrews–Kitagawa–McCloskey (the PDF was binary and no local PDF tools are installed). The exact Tweedie statement and the exact AKM conditional-distribution construction are from my background knowledge, not quoted.
- The Airbnb bias-correction formula could not be fetched (HTTP 403).
- I found no source that applies EB to "boundary-specific" leaderboard probes. The translation is my own.
- The public/private θ equality assumes a random split and similar margin composition of the public and private flip sets. If the private flip set has a different margin distribution, θ_priv differs; see §3 on post-stratification.

---

## 2. Sequential testing and anytime-valid inference under a query budget (SPRT, e-values / test martingales, optimal stopping, value of information, Gittins / knowledge gradient)

### Takeaway
Two tools matter.
- **Decision-theoretic value of information** (knowledge-gradient / EVSI): which candidate to probe next. A probe is worth ≥1e-3 of expected private score only when the candidate's prior precision is both uncertain (effective prior strength m ≲ 20, i.e. τ ≳ 0.1) and centred near 0.5. With a confident validation prior at 0.47 (m = 100), EVSI ≈ 0 and the upload is wasted.
- **E-values / test martingales**: a type-I-safe evidence ledger. They let you probe a candidate in stages and stop or continue on the data without inflating error. 8/8 gives e ≈ 57 (anytime-valid p ≤ 0.018); 6/8 gives e ≈ 1.9 (no evidence).

Gittins indices target cumulative discounted reward, which is the wrong objective. Here the problem is pure exploration with one terminal decision, which is what KG / ranking-and-selection address.

### Cited Findings
- An e-variable for a null family 𝐏 is a nonnegative E with E_P[E] ≤ 1 for all P ∈ 𝐏. An e-process is upper-bounded by a test martingale for every P, so E_τ is an e-value at any stopping time τ — [Ramdas, Grünwald, Vovk & Shafer, Statistical Science 38(4) 2023, arXiv 2210.01948 §2.1, §2.4](https://arxiv.org/html/2210.01948v2).
- Ville's inequality: sup_P P(∃t: E_t ≥ 1/α) ≤ α. The anytime-valid p-value is 1/max_{s≤t} M_s. A level-α sequential test rejects the first time the process reaches 1/α — [arXiv 2210.01948 §2.5–2.7](https://arxiv.org/html/2210.01948v2).
- Combining e-values:
  - Independent e-values multiply.
  - More generally, ∏ B_k with E[B_k | past] ≤ 1 is an e-value (a supermartingale). This licenses chaining probes.
  - Averages of e-values are e-values even under dependence, if the weights are fixed in advance.
  - Source: [arXiv 2210.01948 §2.9–2.10](https://arxiv.org/html/2210.01948v2).
- For simple null vs simple alternative, the likelihood ratio is growth-rate optimal (GRO), linked to Kelly betting. Wald's SPRT uses this LR but differs: the null can be accepted and the stopping rule is fixed in advance. Composite alternatives use plug-in, mixture or Bayes-factor constructions. Universal inference gives an e-process q̄(Xᵗ)/sup_{p∈𝐏} p(Xᵗ) — [arXiv 2210.01948 §2.2, §2.6, §3.1–3.3, §4.1](https://arxiv.org/html/2210.01948v2).
- Confidence sequences satisfy P(∀t: φ(P) ∈ C_t) ≥ 1 − α and come from inverting test-martingale families — [arXiv 2210.01948 §2.8](https://arxiv.org/html/2210.01948v2).
- The knowledge-gradient policy (Frazier, Powell & Dayanik, SIAM J. Control Optim. 2008) myopically maximises the expected one-step increase in the value of information, measured by the terminal utility. It is:
  - optimal for a single measurement and as the horizon goes to infinity;
  - optimal for any horizon in some special cases, with bounded suboptimality otherwise.
  - Sources: [Frazier & Powell PDF](https://optimallearning.princeton.edu/Papers/Frazier%20Powell%20-%20Knowledge%20gradient%20policy.pdf); [Princeton record](https://collaborate.princeton.edu/en/publications/a-knowledge-gradient-policy-for-sequential-information-collection/).
- The correlated-belief KG (INFORMS J. Computing 2009) exploits dependence between alternatives under a multivariate normal prior — [Frazier, Powell & Dayanik JOC PDF](https://optimallearning.princeton.edu/Papers/Frazier%20Powell%20Dayanik%20CorrelatedKnowledgeGradientJOC.pdf). Under independence with a common known variance, the KG step has a closed form — [ranking-and-selection review, arXiv 2008.00249](https://arxiv.org/pdf/2008.00249).
- Blum & Hardt's "Ladder" (ICML 2015) shows that repeated leaderboard evaluation lets participants overfit the holdout. It updates the displayed best score only when a submission is significantly better, and it resisted a boosting attack that exploits leaderboard feedback — [arXiv 1502.04585](https://arxiv.org/pdf/1502.04585); [PMLR v37](https://proceedings.mlr.press/v37/blum15.html).

### Inferences
**(a) E-values for one push [computed].** H₀: θ ≤ 0.5 (the push does not help). Two e-values are used: E_LR = LR against θ₁ = 0.75, and E_mix = mixture over θ ~ U(0.5,1).

| k/n | E_LR (θ₁ = 0.75) | E_mix | Anytime p (1/E_mix) |
|---|---|---|---|
| 8/8 | 25.6 | 56.8 | 0.018 |
| 4/4 | 5.1 | 6.2 | 0.16 |
| 6/8 | 2.9 | 1.85 | 0.54 |
| 7/10 | 2.1 | 1.4 | 0.73 |
| 10/15 | 1.8 | 1.2 | 0.82 |

- Wald SPRT boundaries for α = 0.05, β = 0.2 are 16 (accept H₁) and 0.21 (accept H₀).
- 8/8 crosses 16. Pushes with ~65–70% public precision on 8–15 flips never reach strong evidence. A *positive-expectation* push can still be worth keeping without "significance" (§1g), so e-values are a safety ledger, not the keep rule.

**(b) Staged probing of one candidate (group-sequential / e-process).**
- Order a candidate's public flips by posterior margin (most confident first).
- Upload the top half first. Continue to the full set only if the running e-value > 1 (or the posterior mean > 0.5).
- Validity holds at any stopping time because the product of conditional e-values is an e-process.
- Cost: +1 upload per staged candidate. Benefit: aborts bad candidates after ~half their flips and localises the useful margin range. With ~16 days and limited uploads, this is worth it only for the 2–3 largest candidates (n ≥ 12).

**(c) Value of information per probe (EVSI ≈ one-step KG) [computed].** Setup:
- Prior θ ~ Beta(μ₀m, (1−μ₀)m).
- Probe n_pub public flips. Adopt for private iff posterior mean > 0.5. Private gain = G·n_priv·(2θ̂ − 1)/400.
- EVSI in units of 10⁻³ score:

| μ₀ | m | n = 5 | n = 10 | n = 15 |
|---|---|---|---|---|
| 0.40 | 5 | 0.34 | 0.97 | 1.59 |
| 0.40 | 20 | 0.01 | 0.04 | 0.13 |
| 0.47 | 5 | 0.75 | 1.75 | 2.83 |
| 0.47 | 20 | 0.13 | 0.43 | 0.77 |
| 0.47 | 100 | 0.00 | 0.00 | 0.02 |
| 0.50 | 5 | 0.97 | 2.17 | 3.50 |
| 0.50 | 20 | 0.32 | 0.79 | 1.38 |
| 0.50 | 100 | 0.07 | 0.19 | 0.35 |

- Mapping τ to m (Beta variance μ(1−μ)/(m+1) = τ²): τ = 0.1 ↔ m ≈ 24; τ = 0.2 ↔ m ≈ 5; τ = 0.27 ↔ m ≈ 2.4.
- So if your probes support τ ≈ 0.2+ (one 8/8 plus two failures does), probing a 10–15-flip candidate is worth ≈ +1–3 × 10⁻³ expected private score per upload.
- If you believe validation (τ ≈ 0.05), every probe is worth ≈ 0.
- Ranking rule: probe in order of EVSI, which grows with n_priv × prior SD and is maximal for μ₀ ≈ 0.5. Candidates with validation precision ≤ 0.40 are low priority unless something specific suggests a shift (e.g. the unlabelled scan). **[derivation]**

**(d) Correlated KG.** If boundaries share a cause (e.g. the test player pool is stronger overall), a probe on one boundary informs its neighbours. The correlated KG (multivariate normal prior over the 16 d_j with neighbour correlation ρ) values probes that teach about several boundaries. It also tells you when a single A|1D-adjacent probe (e.g. B|A or 1D|2D) updates the whole kyu/dan split. Cost: tiny compute. Risk: ρ is a guess, so show results for ρ = 0 and ρ = 0.5. **[derivation]**

**(e) Optimal stopping / budget.**
- The last upload counts, so the final submission costs 1 reserved upload (more if the decision is uncertain until late).
- Stop probing when max_j EVSI_j < the value of an alternative use of the upload, or when EVSI < ~0.3 × 10⁻³ (below the noise of everything else).
- Gittins indices solve discounted-reward bandits with independent arms, which is a different objective. I recall this from background knowledge and did not check it against a fetched source; KG / ranking-and-selection is the apt formalism.
- Ladder-style lesson: every public-informed decision increases optimism of the final public score. The private half is the only untouched holdout, so all predictions for it must be shrunk (§1).

### Gaps
- I did not fetch e-BH (Wang & Ramdas) for multiple simultaneous boundaries. I know it exists, but it is not cited here.
- Exact upload limits per day were not provided, so the budget analysis is generic.
- EVSI uses the A↔1D-only ±G scoring and θ_pub = θ_priv. Flips with true B/2D labels lower |G| to 0.368.

---

## 3. Survey statistics and transportability (post-stratification, raking / calibration, small-area estimation, Pearl–Bareinboim transport of a confusion matrix)

### Takeaway
Calibration and raking reweight the 33k validation questions so that their observable margins (predicted class × confidence bins) match the 800 test questions. That transports the validation flip precision correctly only if the shift lives in p(x) (covariate or label shift with an invariant confusion structure). In selection-diagram terms, A|1D's 8/8 vs 47% is a shift *in the mechanism generating features given the label* (an S-node into X given Y) or in labelling. Such a shift is not transportable from validation plus unlabelled test alone, and only the probe labels identify it.

So use raking as a **zero-upload diagnostic**:
- If raked-validation precision stays ≈0.47 while public says 8/8, the effect is conditional (concept) shift. That justifies the hierarchical probe model and argues against inferring other boundaries from unlabelled margins.
- Fay–Herriot (§1) is the small-area tool for per-boundary effects with few labels.

### Cited Findings
- Deville & Särndal, "Calibration Estimators in Survey Sampling" (JASA 1992, 87(418):376–382):
  - Calibrated weights w_i minimise a distance D(d, w) to the design weights, subject to matching known auxiliary totals.
  - Raking and regression (GREG) estimators are special cases of the distance choice.
  - Different distance functions give asymptotically equivalent estimators.
  - Sources: [Statbel working paper summary](https://doc.statbel.fgov.be/publications/A000.02/A000.02E_WP_2001_3.pdf); [Southampton eprint](https://eprints.soton.ac.uk/34683/1/013s.pdf); [ISI Survey Statistician 2023](https://isi-iass.org/home/wp-content/uploads/Survey_Statistician_2023_July_N88_017.pdf).
- Fay–Herriot EBLUP and shrinkage factor B_j = D_j/(A + D_j); see the sources in §1 — [arXiv 1203.5233](https://arxiv.org/pdf/1203.5233); [arXiv 1701.04176](https://arxiv.org/pdf/1701.04176). Smoothing the sampling variances improves the model-based estimator — [Statistics Canada 2021](https://www150.statcan.gc.ca/n1/pub/12-001-x/2021002/article/00007-eng.htm).
- Pearl & Bareinboim's selection diagrams add S-nodes marking mechanisms that differ between source and target, reducing transportability to do-calculus manipulations — [UCLA tech report r372-a](https://ftp.cs.ucla.edu/pub/stat_ser/r372-a.pdf); [r404](https://ftp.cs.ucla.edu/pub/stat_ser/r404-reprint.pdf); [arXiv 1503.01603](https://arxiv.org/abs/1503.01603v1).
- Transporting P(y|x) (a classifier) needs only a subset of the selection diagram's variables to be observed. Correa & Bareinboim developed a complete algorithm. Jalaldoust, Bellot & Bareinboim (NeurIPS 2024) treat covariate shift and label shift as diagram-encoded assumptions and study a classifier's average target loss. A 2025 follow-up bounds functionals such as a classifier's target generalisation error. This summary is from a search across these papers — [causalai.net r99](https://causalai.net/r99.pdf); [arXiv 2503.23605](https://ar5iv.labs.arxiv.org/html/2503.23605).

### Inferences
**(a) Selection-diagram reading of the three candidate mechanisms [derivation].**
1. S → Y (pure label shift).
   - The confusion matrix P(ŷ | y) is invariant, only the class mix changes.
   - The flipped-set precision transforms by an odds factor: odds′ = r · odds₀, where r = q(A)/q(1D) ÷ p(A)/p(1D) among borderline questions.
   - With p₀ = 0.47 (odds 0.887), a test precision of 0.90 needs r ≈ 10.1, 0.77 needs r ≈ 3.8, and 0.60 needs r ≈ 1.7.
   - A tenfold local prevalence shift is implausible. The more plausible readings are a moderate prevalence shift (r ≈ 2–4) combined with luck (8 questions), or (2)/(3) below.
2. S → X with X ← Y (class-conditional feature shift at one boundary, e.g. test "A" players play more dan-like than validation "A" players).
   - P(ŷ | y) changes for y ∈ {A, 1D}.
   - It is not identifiable from unlabelled test data plus validation without parametric assumptions. The probes are necessary.
3. S → labelling (organisers' rank labels defined differently in test, e.g. a date cut-off or server-rank conversion).
   - Same identifiability as (2).
   - It would make *all* borderline 1D-predicted test players with a particular feature signature more likely to be A. A probe that varies the margin (§2b) can separate this from (1).

**(b) Raking diagnostic, 0 uploads, minutes of CPU.**
- Calibrate the 33k validation weights so that the weighted distribution of (argmax class) × (top-2 margin decile) × (perhaps HD bucket) equals the 800-test distribution.
- Two standard choices: chi-square distance (w_i = d_i(1 + x_iᵀλ), the GREG form) or raking ratio (w_i = d_i exp(x_iᵀλ)). These forms are from my knowledge of Deville–Särndal and were not quoted from a fetched page.
- Recompute each boundary's flip precision under these weights.
- Expected outcome: a small change (≤ ±0.05) for most boundaries. If A|1D stays ≈0.47, mechanism (1) through observable margins is ruled out as the explanation.
- Risk: none for the score. Tail weights can explode with 10×10 cells against 800 targets, so trim weights or collapse cells.

**(c) Post-stratifying the public→private prediction.**
- Public and private flip sets come from the same rule but can differ in margin composition. Split the flips into 2 margin strata (near-threshold vs deep).
- Estimate θ_s per stratum from public (with shrinkage) and weight by private stratum counts: θ̂_priv = Σ_s (n_s^priv / n^priv) θ̃_s.
- With 8 public flips, use at most 2 strata, sharing strength through a common logistic slope in margin.
- This matters when, for example, all 8 public flips were near the threshold but many of the 19–22 private flips are deeper. **[derivation]**

**(d) Small-area framing.** "Area" = boundary (or boundary × margin stratum). "Direct estimate" = decoded probe precision, or the forward–backward posterior expected precision. "Synthetic estimate" = raked-validation precision. Fay–Herriot then blends the two with B_j = D_j/(Â + D_j), and untested areas get the synthetic estimate plus the common shift μ̂. This is §1 with the regression prior made explicit: use the raked validation precision as the covariate x_jᵀβ rather than a constant. **[derivation]**

### Gaps
- I did not fetch the full Deville–Särndal text or the Bareinboim confusion-matrix results. No source gives an explicit "transport a confusion matrix across a boundary-specific shift" estimator; the search agent also found none.
- Whether the test labels were produced by a different rank-conversion rule (mechanism 3) is a data-provenance question, not a statistical one.

---

## 4. Misclassification / latent-class models (Dawid–Skene, latent class analysis, multiclass Rogan–Gladen) and identifiability of a one-boundary change

### Takeaway
- Under label shift, the multiclass Rogan–Gladen correction is black-box shift estimation: ŵ = Ĉ⁻¹μ̂, using the validation confusion matrix C and the test predicted-class frequencies μ̂. It identifies the target class priors from unlabelled data if C is full-rank.
- It does **not** identify a change in the confusion matrix itself at one boundary. That needs either target labels (the probes) or several conditionally independent "views" (Dawid–Skene / latent class identifiability), which ensemble members are not.
- With ~150 test questions per boundary, the Rogan–Gladen-corrected local prevalence has SD ≈ 0.06–0.14 [computed], so only large local prevalence shifts are detectable. This matches the scan finding only A|1D.

### Cited Findings
- BBSE (Lipton et al.), as used by Podkopaev & Ramdas:
  - μ = C_P(f)·w, where μ_i = P_Q(f(X) = i), C is the source confusion matrix and w = q(y)/p(y). The estimator is ŵ = Ĉ⁻¹μ̂.
  - It requires q(y) > 0 ⇒ p(y) > 0 and C_P(f) full-rank.
  - Source: [Podkopaev & Ramdas, UAI 2021, arXiv 2103.03323, Appendix A](https://ar5iv.labs.arxiv.org/html/2103.03323).
- The basic difficulty in misclassification problems is identifiability. Without information beyond the raw data, misclassification cannot be fully accounted for. Permutation-type non-identifiability is handled by problem-structure constraints — [Swartz et al., Canadian J. Statistics 2004](https://www.sfu.ca/~tswartz/papers/yoel.pdf).
- Dawid–Skene estimates annotators' confusion matrices in a first stage. Identifiability under incomplete labelling and finite samples is described as not established in general — [search summary of crowd-sourcing EM papers, e.g. arXiv 2306.03288](https://arxiv.org/pdf/2306.03288v1.pdf).
- Sufficient conditions for identifiability under multiclass label noise, with algorithms for infinite and finite samples — [arXiv 1710.01167](https://arxiv.org/pdf/1710.01167v1).

### Inferences
**(a) Multiclass Rogan–Gladen at one boundary (2 × 2 restriction) [derivation + computed].**
- Restricted to borderline questions predicted A or 1D, Rogan–Gladen gives π̂_A = (p̂_obs + Sp − 1)/(Se + Sp − 1), with Se and Sp the validation sensitivity and specificity of "predict A" among true {A, 1D}.
- Its SD is √(p(1−p)/n)/J, with J = Se + Sp − 1 (Youden).
- With n = 150 and p_obs = 0.5: SD(π̂) = 0.136 (J = 0.3), 0.082 (J = 0.5), 0.058 (J = 0.7).
- Borderline A/1D discrimination is weak (J ≈ 0.3–0.5), so a local prevalence change < ~0.2 is invisible at 2σ. A change large enough to turn 47% into 90% precision (r ≈ 10, §3a) *would* be visible. The scan's p = 0.003 at A|1D is therefore consistent with a real local prevalence shift of moderate size, plus a lucky 8/8.

**(b) Identifiability of a one-boundary *conditional* change.** Model the target confusion matrix as C′ = C + Δ, with Δ nonzero only in the A/1D rows and columns. Unlabelled test data identify only the product C′w. One boundary contributes 2 unknowns in Δ plus 1 in w, but only 1 new moment (the predicted A:1D ratio), so Δ is not identified. **[derivation]**

Three ways out:
1. Probe labels: the k/n from public uploads plus the forward–backward posteriors give direct target information on Δ.
2. Parametric restriction: e.g. Δ is a pure location shift of the latent ordinal score at threshold A|1D, which is the cumulative-link intercept δ of the sibling note. Then 1 parameter is identified by 1 moment.
3. Multiple conditionally independent views: Dawid–Skene-type identifiability (the Kruskal / Allman–Matias–Rhodes 3-view condition, from memory, not fetched) needs ≥3 views independent given the true class. Ensemble members trained on the same features violate this, so do not treat them as independent annotators.

**(c) Latent-class model on test.** A latent class model with 10 ordered classes and per-boundary class-conditional emission shifts, fitted to the 800 test posteriors with the public decoded labels as partial supervision (semi-supervised EM), is the principled version of "unlabelled scan + few labels". It is identifiable only under restriction (2). Expected gain: it formalises what the scan already does. Cost: hours to implement, 0 uploads. Risk: over-fitting Δ to 8 questions, so put a hierarchical prior on δ_j (§1).

### Gaps
- I did not find a published "multiclass Rogan–Gladen" under that name. BBSE is the de facto equivalent.
- The 3-view identifiability theorem (Allman, Matias & Rhodes 2009) is cited from memory, not verified in this session.
- No source treats a confusion-matrix change localised to one ordinal boundary.

---

## 5. Change-point and two-sample methods for localising where test differs from validation (classifier two-sample tests, local two-sample tests, scan statistics, MMD witness; power at ~150 points per boundary)

### Takeaway
These methods detect differences in the *unlabelled* distribution p(x) (here, the posterior vectors) between validation and test. They are the right tool for the per-boundary scan, but blind to pure P(y|x) shifts. Their power at n ≈ 150 test points per boundary is modest:
- A 1-D shift of 0.3 SD is found with ≈0.82 power after a 9-boundary Bonferroni correction.
- A 0.2 SD shift is found only ≈37% of the time.
- A 0.1 SD shift is essentially undetectable.

"Nothing elsewhere" is therefore weak evidence of no shift for effects ≲0.2 SD.

### Cited Findings
- Classifier two-sample tests (C2ST): if two samples come from the same distribution, a classifier trained to tell them apart should be at chance on held-out data. The test statistic has interpretable units and a simple null, and the classifier's predictive uncertainty can explain *where* the samples differ — [Lopez-Paz & Oquab, ICLR 2017, arXiv 1610.06545](https://arxiv.org/abs/1610.06545v4); [ICLR record](https://mlanthology.org/iclr/2017/lopezpaz2017iclr-revisiting).
- Kim, Lee & Lei, "Global and local two-sample tests via regression" (Electronic J. Statistics 2019):
  - A regression framing of two-sample testing, extending classification-accuracy tests.
  - A local testing framework that identifies *where* two multivariate distributions differ.
  - Uses kernel, nearest-neighbour or random-forest regression with permutation tests. Earlier local work (Duong 2013) used kernel density estimates, which do poorly in multiple dimensions.
  - Sources: [Project Euclid](https://www.projecteuclid.org/euclid.ejs/1576573369); [CMU page](https://www.stat.cmu.edu/~annlee/publication/kim19-regtest).

### Inferences
**(a) Power at n_test ≈ 150 vs a large validation set [computed].** Two-sided z-test of a mean shift d (in SD units) of a 1-D boundary statistic, e.g. the logit P(≤A) − P(>A) margin among questions near that boundary.

| d | Power at α = 0.05 | Power with Bonferroni over 9 boundaries |
|---|---|---|
| 0.1 | 0.23 | 0.06 |
| 0.2 | 0.69 | 0.37 |
| 0.3 | 0.96 | 0.82 |
| 0.4 | 1.00 | 0.98 |

The validation side contributes negligible variance at 33k.

**(b) Recommended localised test.** Fit a C2ST / Kim–Lee–Lei-style regression of 1{test} on the 10-dim posterior (or on the boundary margin and the HD features), with validation reweighted to the same size. Then report the *witness*: the fitted P(test | x) − ½ as a function of the boundary margin.
- A positive witness concentrated on the 1D side of A|1D says test has excess mass where the push acts.
- Permutation p-values per boundary window form the scan. Scan statistics, i.e. the max over windows of a standardised local excess calibrated by permutation, give a selection-adjusted p for "the most different boundary".
- Cost: 0 uploads, minutes. Expected gain: diagnostic only (which candidates to probe first, feeding the EVSI prior in §2c).

**(c) The kernel MMD witness function**, f(x) = mean_k(x, test) − mean_k(x, val), is the kernel analogue. With 800 points, use a 1-D kernel on the boundary margin. Multi-dimensional MMD loses power at this n. (Background knowledge, not fetched.)

**(d) Blind spot.** A shift that changes P(y|x) but not p(x) (mechanisms 2–3 in §3a) produces zero two-sample signal. The A|1D detection (p = 0.003) shows that *some* p(x) change exists there. It does not show that the 8/8 is explained by it. The §3 calculation shows the required odds ratio (~10 for 0.9 precision) would be large; a moderate r plus noise is the parsimonious reading.

### Gaps
- I did not fetch scan-statistic or MMD-witness primary papers in this session, so those descriptions are background knowledge.
- No source gives power for local tests at n ≈ 150 in a 10-dim simplex. The table above is a 1-D approximation.

---

## 6. Conformal prediction under label shift / conditional shift as a decision tool for borderline questions

### Takeaway
Label-shift-weighted conformal (Podkopaev & Ramdas 2021) and class-conditional (Mondrian) conformal produce prediction *sets* with coverage guarantees. They do not maximise the exact/adjacent score, so they are not a decision rule for which label to submit. Their usable role here is to **define the candidate flip set objectively**: the questions whose label-conditional conformal set contains both A and 1D. The keep/flip decision within that set then uses the hierarchical estimate (§1).

Expected score gain: ~0 directly. Cost: 0 uploads. Risk: low.

### Cited Findings
- Podkopaev & Ramdas (UAI 2021) adapt split conformal and post-hoc calibration to label shift by reweighting with w(y) = q(y)/p(y) estimated from unlabelled target data (BBSE). Label shift harms coverage and calibration — [arXiv 2103.03323](https://arxiv.org/abs/2103.03323v3); [PMLR v161](https://proceedings.mlr.press/v161/podkopaev21a.html).
- Weighted threshold (their eqs. 8–9): τ*_w(y) = Q_{1−α}(Σᵢ p̃ᵢʷ(y) δ_{rᵢ} + p̃ʷ_{n+1}(y) δ₁), with p̃ᵢʷ(y) = w(Yᵢ)/(Σⱼ w(Yⱼ) + w(y)).
  - With known w, coverage ≥ 1 − α in finite samples (Theorem 2).
  - With estimated ŵ, coverage holds only asymptotically (Corollary 1), and a plug-in quantile can fail at a jump. No finite-sample correction for estimated weights is given.
  - Source: [ar5iv 2103.03323](https://ar5iv.labs.arxiv.org/html/2103.03323).
- Label-conditional conformal: τ*_c(y) = Q_{1−α_y}({rᵢ : Yᵢ = y} ∪ {1}).
  - Guarantee: P(Y ∉ set | Y = y) ≤ α_y for every y, exactly in finite samples, and robust to label shift without estimating weights.
  - Costs: the calibration data are split into K groups, and sets get larger when classes overlap.
  - Source: [ar5iv 2103.03323](https://ar5iv.labs.arxiv.org/html/2103.03323).
- In experiments, uncorrected conformal sets under-cover under label shift. Weighted sets reach the target with oracle or BBSE weights. Estimated weights are slightly worse than oracle (wine-quality data, p = (0.1, 0.4, 0.5) → q = (0.4, 0.5, 0.1)) — [ar5iv 2103.03323](https://ar5iv.labs.arxiv.org/html/2103.03323).

### Inferences
- **Label-conditional coverage is invariant to label shift but not to class-conditional shift.** If test "A" players look different from validation "A" players (mechanism 2), the per-class score quantiles are wrong for test, and coverage for A degrades. Conformal offers no protection against the very shift that makes A|1D interesting. **[derivation]**
- **Using conformal sets as a flip-set definition.** Questions whose set is {A, 1D} (or contains both) form the ambiguous pool. The current "multiply A by r = 2" rule is a heuristic version of this. Defining the pool by a calibrated α and ranking by nonconformity gives a reproducible pool. Its size (and so n_pub, n_priv) for each α feeds the EVSI table in §2c.
- **Weighted conformal with BBSE weights** requires a full-rank 10 × 10 confusion matrix and many target points per class. With 800 test questions and weak discrimination between adjacent ranks, ŵ is noisy. Expect only asymptotic validity, which is not reliable at this n.
- **Rating:** applicability low to moderate (organising tool, not a scorer). Gain ≈ 0 directly, possibly positive by stopping ad-hoc flip-set choices. Cost: 0 uploads, < 1 h. Risk: low.

### Gaps
- I found no conformal method designed for conditional (P(x|y)) shift localised at one class boundary.
- I did not verify conformal methods for ordinal sets (contiguous intervals) in this session.

---

### Cross-method rating summary (my judgement, not from sources)

| Method | Applicability | Expected gain (score) | Uploads | Compute | Main risk |
|---|---|---|---|---|---|
| Hierarchical EB / Fay–Herriot over boundary effects, keep rule by posterior expected private Δ (§1) | High | Decision quality; keeps A\|1D (E[Δ] +0.0002 to +0.028) and prevents chasing noisy wins | 0 | Minutes | Fitting τ on survivors only; tiny K |
| Two-groups (parametric local-fdr) posterior P(shift) per boundary (§1e) | High | Same as above; explicit π₀ sensitivity | 0 | Minutes | π₀ subjective |
| AKM conditional inference on winners (§1g) | Medium | Small here (large winning margin) | 0 | Minutes | Very wide intervals when the margin is small |
| EVSI / knowledge-gradient probe ordering (§2c–d) | High | ≈1–3e-3 per well-chosen probe if τ ≳ 0.2; ≈0 if τ ≈ 0.05 | Uses the budget optimally | Minutes | Prior on τ and ρ |
| E-values / staged probing (§2a–b) | Medium | Saves uploads on bad candidates; not a keep rule | +1 per staged candidate | Trivial | Over-conservatism if used as the keep rule |
| Raking / calibration diagnostic (§3b) | Medium | 0 directly; discriminates label shift vs conditional shift | 0 | Minutes | Extreme weights |
| Post-stratified public→private transfer (§3c) | Medium | Corrects θ_priv when flip margins differ | 0 | Minutes | Too few points per stratum |
| BBSE / Rogan–Gladen local prevalence (§4a) | Medium | Diagnostic; SD 0.06–0.14 at n = 150 | 0 | Minutes | Low J inflates variance |
| Semi-supervised latent-class model with a one-parameter boundary shift (§4c) | Medium | Formalises scan + probes | 0 | Hours | Over-fit to 8 labels |
| C2ST / local two-sample / scan with witness (§5) | Medium | Probe prioritisation | 0 | Minutes | Blind to P(y\|x) shift; power ≤ 0.4 for d ≤ 0.2 |
| Weighted / label-conditional conformal (§6) | Low–medium | ≈0; defines flip pools | 0 | < 1 h | Not valid under conditional shift |

# Cross-domain analogues for a threshold-specific label-vs-ability gap (test players labelled A but playing at 1D strength), and how other fields detect and correct such gaps

Research date: 2026-11-08. Scope: published rating rules (Yugen no Ma, other Go servers, chess, one esports ladder) and general methods from psychometrics, epidemiology/diagnostics and deployed-ML monitoring. No attempt was made to identify any player; everything below is at the level of published rules and general methods. Statements marked "(inference)" are my own reasoning or arithmetic, not from a source. Competition label mapping used below, from the sibling note `/home/u2507115/BK_AICUP_2026-1/research_notes/類似任務他人解法/aicup_context.md` (which cites the official tutorial repo): D = 10–12k, C = 7–9k, B = 4–6k, **A = 1–3k**, then 1D … 6D — [AILAB-NDHU/AICup-2026-Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial).

New relative to the earlier note `/home/u2507115/BK_AICUP_2026-1/research_notes/玩家層偏差的新資訊來源/評分系統與心理計量.md` (which could not read the Yugen PDFs): the official Yugen no Ma rating page (HTML, manual_G03) is readable and gives the exact update formula, the rank-to-RP table and the per-rank step sizes. This closes most of that note's Q2 gaps.

---

## Q1. Rank rules: does Yugen no Ma (or any comparable server or chess body) put a gate, lag or rule change at the kyu→dan (1級→初段) boundary?

### Takeaway
Yugen no Ma's published rules contain **no special requirement for 初段**. A single Elo-type formula, RP(new) = RP(old) + C(S − P), runs without a break from 18級 to 8段. Every rank is 1,000 RP wide, and the step size falls smoothly across the boundary (1級 116 → 初段 105 RP per game). The documented breaks in the rules sit elsewhere: at 18/19級 (the 8,000 floor and a separate beginner system), at 6段 (wins weighted by recent win rate) and at 8段 (no streak bonus). The one break that exists only at A|1D is the competition's own grouping: A covers three ranks (3,000 RP) while 1D covers one (1,000 RP). Other systems do have explicit kyu/dan or title gates: Tygem (per forum reports) uses different promotion windows and thresholds for kyu and dan, and FIDE titles combine a rating threshold, performance norms and a lifetime title (hysteresis).

### Cited Findings
**Yugen no Ma (official rating page and FAQ)**
- Two tiers. Beginners (25級–19級) use a separate points system, starting at 25級 with 1,000 points; the first 10 games add 50 points each whatever the result, then only wins add points, with no deductions until 18級. General players (18級–8段) use a formula-based system — [幽玄の間 レーティング](https://u-gen.nihonkiin.or.jp/static/manual_G03.asp)
- General formula: RP(new) = RP(old) + C(S − P), where S = 1/0/0.5 for a win, loss or draw, P is the win probability computed from the two players' point difference, and "「C」は各棋力別に値の違う常数です" (C is a constant that differs by rank). Neither the P formula/scale nor the C values are published — [幽玄の間 レーティング](https://u-gen.nihonkiin.or.jp/static/manual_G03.asp)
- Each rank spans 1,000 RP. Examples from the table: 18級 8,000–8,999; 3級 23,000–23,999; 2級 24,000–24,999; 1級 25,000–25,999 (base 25,500); 初段 26,000–26,999 (base 26,500); 2段 27,000–27,999; 6段 31,000–31,999; 7段 32,000–32,999; 8段 33,000+ — [幽玄の間 レーティング](https://u-gen.nihonkiin.or.jp/static/manual_G03.asp)
- "1局当り平均変動RP（目安）" (average RP change per game, guide value): 9級 204, 8級 193, 7級 182, 6級 171, 5級 160, 4級 149, 3級 138, 2級 127, **1級 116, 初段 105**, 2段 95, 3段 84, 4段 74, 5段 64, 6段 54, 7段 44, 8段 35 — [幽玄の間 レーティング](https://u-gen.nihonkiin.or.jp/static/manual_G03.asp)
- Streak multipliers: "3連勝は25％、4連勝は50％、5連勝は75％、6連勝は100％が基本獲得点数に加算されます", applied symmetrically to losing streaks; 8段 is excluded — [幽玄の間 レーティング](https://u-gen.nihonkiin.or.jp/static/manual_G03.asp)
- 6段–8段 only: "勝ったときの獲得ポイントは、両対局者の勝率（直近20局）を加味して加算されます" (points for a win also depend on both players' win rates over their last 20 games); losses are not adjusted. The wording for the 8段 example (a 70% average win rate gives 100% of the base gain) is ambiguous — [幽玄の間 レーティング](https://u-gen.nihonkiin.or.jp/static/manual_G03.asp)
- Speed examples: "18級の場合は4連勝程度で17級に昇級できます" (about 4 straight wins); "7段から8段になるためには30局程度の勝ち越しが必要になります" (about 30 net wins) — [幽玄の間 レーティング](https://u-gen.nihonkiin.or.jp/static/manual_G03.asp)
- Floor: "8000点が最低ポイントになり、19級への降級はありません" (8,000 is the minimum; no demotion to 19級) — [幽玄の間 レーティング](https://u-gen.nihonkiin.or.jp/static/manual_G03.asp)
- Only rating games change RP. Games with 4 or more handicap stones automatically become free games (so rating games have at most 3 stones). A game counts once at least one move has been played. The graph and personal record update the next day, while the waiting-room RP updates in real time — [幽玄の間 レーティング](https://u-gen.nihonkiin.or.jp/static/manual_G03.asp)
- The page states no extra condition for reaching 初段 or any dan rank; nothing about promotion matches, minimum game counts or a separate dan scale (absence noted when reading the full page) — [幽玄の間 レーティング](https://u-gen.nihonkiin.or.jp/static/manual_G03.asp)
- The self-declared rank (棋力) cannot normally be changed after registration: "棋力の変更は基本的に認めておりません。但し、ご登録の際に誤った棋力を選択してしまった場合のみ、1回に限り変更" (only a one-time correction of a wrongly selected rank) — [幽玄の間 Q&A No.29](https://u-gen.nihonkiin.or.jp/faq/faq_view.asp?faq_div=1&num=29)
- The table is consistent with the 2017 AI 8-dan exam held on Yugen: the run started at 7段 = 32,500 RP, passed at ≥33,000 (reached at 37–14, RP 33,023) and finished at 101 games, 68–33, RP 33,385 (the article elsewhere says 102 games, 69 wins) — [4Gamer](https://www.4gamer.net/games/362/G036268/20171016080/)
- The Yugen portal has a 免状申請 (official certificate application) page under My Page; its conditions were not readable — [幽玄の間 top](https://u-gen.nihonkiin.or.jp/), [免状申請](https://u-gen.nihonkiin.or.jp/mypage/menjo_step1.asp)

**Other Go servers**
- KGS: the rank is recomputed continuously from ranked games in the last 180 days (19×19, at most 6 handicap stones). A "?" marks an uncertain rank that "may be very, very far from the user's actual rank" and usually clears after about two wins and two losses against solidly ranked players. The page names sandbagging (deliberately forfeiting won games to keep the rank low); players who forfeit too many games while ahead may be deranked. It gives no kyu/dan difference in rank width or speed — [KGS rank help](https://www.gokgs.com/help/rank.html)
- OGS (Glicko-2): rank = ln(rating/525) × 23.15, a log mapping, so a rank is wider in rating points at higher ranks. The 2021 retune aligned the dan range between AGA and EGF and widened bands. Calibration target: a one-rank gap equals one stone of handicap, with black's 1-stone win rate around 43.3% (EGF ≈ 42–43%). Cells with fewer than 100 games were excluded. A `*` vs `**` bug had made volatility about 30× too large, and windowed systems had been added to compensate — [OGS forum: 2021 rating and rank adjustments](https://forums.online-go.com/t/2021-rating-and-rank-adjustments/33389)
- Tygem (forum reports, not official, details conflict): promotion uses a rolling window of recent games. Dan uses 20 games (≥70% to promote at 1d–4d, ≥75% at 5d–9d); kyu uses 15 games (≥65%, double promotion at ≥90%); demotion at ≤30%. On these reports the window and threshold change exactly at the kyu→dan boundary — [L19 thread p=90205](https://www.lifein19x19.com/viewtopic.php?p=90205), [L19 thread p=57399](https://www.lifein19x19.com/viewtopic.php?p=57399), [L19 "ELO width of Tygem ranks"](https://www.lifein19x19.com/viewtopic.php?p=259452)
- Pandanet: a self-declared rank is shown as provisional (「仮」) until 20 rated games. This comes from a search-engine summary of the manual and was not verified first-hand — [Pandanet manual](https://www.pandanet.co.jp/manual/man17.pdf)

**Chess and esports**
- FIDE titles are a gate with hysteresis. A rating of at least 2500 (GM) or 2400 (IM) must have been reached once, plus title norms (performance in events of at least 9 games, normally three norms), and titles are kept for life. The IM norm performance is ≥2450 against opponents averaging ≥2230 — [FIDE Title Regulations (2021)](https://handbook.fide.com/chapter/B012022), [FIDE Title Regulations (2024)](https://handbook.fide.com/chapter/B012024), [Wikipedia: FIDE titles](https://en.wikipedia.org/wiki/FIDE_titles)
- Lichess (Glicko-2) marks a rating provisional ("?") when the rating deviation exceeds 110. RD grows during inactivity, so an inactive player becomes provisional again. This is from forum and Q&A posts, not the FAQ text — [Lichess forum](https://lichess.org/forum/general-chess-discussion/what-does-a-question-mark-after-a-rating-mean)
- League of Legends: a hidden MMR drives matchmaking and LP gains. Visible rank and MMR can diverge (a negative LP differential means MMR is below the visible rank), and promotion series were removed so 100 LP now promotes automatically. The removal date differs between sources — [Riot: What's next for ranked](https://www.leagueoflegends.com/en-us/news/game-updates/what-s-next-for-ranked/), [r/summonerschool ranked FAQ mirror](https://r.datuan.dev/r/summonerschool/wiki/rankedfaq)

### Inferences
- (inference) **No gate at shodan on Yugen.** From the table, the net wins needed to cross one rank, at about 1,000/(average step) and ignoring streak bonuses, are: 3級 ≈ 7.2, 2級 ≈ 7.9, 1級 ≈ 8.6, **初段 ≈ 9.5**, 2段 ≈ 10.5. The step shrinks by about 11 RP per rank throughout, with no jump at 1級→初段, and streak bonuses (up to +100%) shorten climbs further. A 1D-strength player starting anywhere in A would cross into 初段 within a few dozen rating games. A server-side "promotion lag" at shodan therefore cannot explain a persistent A label, unless the player plays few rating games or plays mostly free games, which do not move RP.
- (inference) **The only break that exists exclusively at A|1D is the competition's band width.** In RP, the A band is 3,000 wide (23,000–25,999) and 1D is 1,000 wide (26,000–26,999). B|A joins two 3,000-wide bands and 5D|6D two 1,000-wide bands, so A|1D is the only boundary where width changes. This matches the observation that only A|1D shows the effect while B→A and 5D→6D adjustments failed. How the width matters depends on the undisclosed "representative value":
  - *Band-level mode (time-share argmax over the 10 labels):* wide bands capture the mode. Example: 40% of the period in 1–3級, 35% in 初段 and 25% in 2段 gives label A, although 60% of the time was at dan level.
  - *Median or mean RP:* band width does not bias the label itself, but A's conditional strength distribution has a long upper tail (1級) next to 1D. That is a "spectrum" issue (Q4): errors at A|1D depend on what share of A players sit near the top of the band.
  - *Start-of-period or early-snapshot rank:* improvers lag. A linear climber needs about 24 net wins to cross A but about 9.5 to cross 1D, so improvers are disproportionately labelled A.
- (inference) Any of these mechanisms is present in training labels too. It produces a **train→test discrepancy** only if the test sample differs from training in composition: more improvers, games drawn from a later part of the 2021–2023 window than the label time, or A players concentrated near 1級. The leaderboard result (8/8 probed questions truly A) cannot tell these apart. It shows only that, for the probed slice, P(label = A | games look like 1D) is higher in test than in validation.
- (inference) Registration lock plus low activity can yield stale labels without sandbagging. A player who declared 1–3級 at sign-up, improved offline and plays few *rating* games would show 1D-like moves under an A rank. KGS explicitly documents deliberate sandbagging; Yugen documents nothing on it.
- (inference) The only documented change in rating dynamics in the dan range is at 6段 (win-rate-weighted gains), so a server-rule effect would be expected at 5D|6D rather than at A|1D. The failed 5D→6D adjustment suggests that rule matters little for the labels. This is weak evidence, since the rule's effective date is unknown.

### Gaps
- Not public: how the organisers computed the per-player "representative value" (mode, median, snapshot, maximum), and whether test games come from the same time window as the label. This is the decisive unknown for choosing among the mechanisms above.
- Yugen's C constants, P formula and scale, and the date the 6段–8段 rule took effect are not published. The manual page is undated, so it is unconfirmed that it matches 2021–2023.
- The 免状申請 conditions (e.g., whether reaching 初段 on Yugen qualifies for a paid 初段 certificate, which would create a shodan-specific incentive) could not be read. Unknown.
- Fox (野狐) promotion rules: no official or reliable source found (two searches). Tygem's numbers rest on forum posts only. KGS's rating math page (senseis.xmp.net) returned 403.
- No source found on a psychological "shodan barrier" (players avoiding rating games near 1級). Not researched beyond rule pages.

---

## Q2. What does the rating-system literature say about lag, floors, deflation/inflation, provisional ratings and sandbagging, and how were the gaps diagnosed and corrected?

### Takeaway
The documented mechanisms are:
- update lag for fast-improving players (chess juniors underrated by up to about 50 Elo);
- scale distortion from floors and from low initial ratings of many improving entrants (USCF 1990s deflation; FIDE 2024 compression);
- provisional or uncertain ratings (Glicko RD, KGS "?", Pandanet 仮);
- deliberate sandbagging.

The chess diagnostic is to compare **actual vs expected score as a function of rating difference, over time**. Corrections include a one-time rescaling (FIDE compression), bonus points (USCF), higher K for juniors, a raised floor, and seeded initial ratings.

### Cited Findings
- FIDE 2024 (Sonas): players 400 points higher scored 87% in 2008–2012 but 79% in 2021–2023, with the curve flattening each year from 2013. Sonas concluded that strength across 1000–2400 Elo spans about 1000 points, not 1400. He attributed this to young, newer players getting very low initial ratings during rapid improvement and then taking points from established players; more than 200,000 new players joined in six years. Fixes: initial ratings stabilised with two hypothetical draws against 1800-rated opponents, and initial ratings capped at 2200 — [Chess.com news](https://www.chess.com/news/view/fide-mathematician-proposes-changes-to-improve-rating-accuracy), [Sonas supplemental report](https://www.fide.com/docs/presentations/Sonas%20Supplemental%20Report.pdf)
- FIDE implemented a one-time boost of 0.40 × (2000 − R) for players below 2000 (rank order preserved, spacing compressed), raised the minimum rating from 1000 to 1400, and applied the changes on 1 March 2024. Critic Otto Milvang said this "will reset the current deflation … however it will not stop the deflation for the future" — [Chess.com news](https://www.chess.com/news/view/fide-adds-rating-points-to-more-than-300-000-players), [FIDE call for discussion](https://www.fide.com/fide-seeks-public-discussion-about-proposed-rating-changes/)
- USCF Ratings Committee: "Overall rating levels have deflated from the mid-1990s through 2000 … when rating floors were decreased by 100 points without a counteracting inflationary mechanism." The 2001 system re-inflated ratings. The bonus threshold was lowered from B=10 to B=6 "to accelerate the re-inflation of the rating pool", then raised to B=8 to slow increases. For Glicko-2, "a bonus-type mechanism would be an essential feature" — [Glickman RC report (c.2012)](https://www.glicko.net/ratings/report12.txt)
- USCF later reported ratings "uncomfortably higher relative to where they were at the end of 1997" and proposed B=12. The floor algorithm is "subtracting 200 from the highest integer-rounded rating" — [Glickman RC report (c.2014)](https://www.glicko.net/ratings/report15.txt)
- Juniors: on average juniors are statistically underrated by up to 50 Elo against adults, and the bias grows with the age gap. The explanation is that ratings trail fast improvement, which is why FIDE and USCF give young players higher K. The author calls the analysis incomplete (blog-style, not peer reviewed) — [ChessBase, part 1](https://en.chessbase.com/post/elo-rating-system-how-underrated-are-the-kids), [ChessBase, part 2](https://en.chessbase.com/post/elo-ratings-how-underrated-are-the-kids-part-two)
- Provisional-uncertainty mechanisms: Lichess RD > 110 means "?" and RD grows with inactivity ([Lichess forum](https://lichess.org/forum/general-chess-discussion/what-does-a-question-mark-after-a-rating-mean)); KGS "?" ranks ([KGS](https://www.gokgs.com/help/rank.html)).
- Sandbagging, meaning deliberately losing or forfeiting to keep a low rank, is named on KGS — [KGS](https://www.gokgs.com/help/rank.html). Floors are commonly justified as anti-sandbagging devices, but the only quantitative statement found links floors to deflation (USCF above). Forum discussion disagrees — [US Chess forum](https://forum.uschess.org/t/uscf-rating-deflation/383)

### Inferences
- (inference) The Go-server analogue of "juniors" is any fast-improving player. If test questions over-represent improvers relative to training, or sample games later than the label snapshot, labels lag strength. With A three ranks wide, the lag is largest exactly where an improver leaves A.
- (inference) The FIDE diagnostic transfers directly. Bin questions by the model's continuous ability score and compare the observed label mix (from the few known labels) with the training-calibrated expectation, per boundary. This is a DIF/DSF test (Q3) with the model score as the matching variable.
- (inference) The FIDE correction (rescale one part of the scale, keep order) corresponds to moving one decision threshold (A|1D) while keeping the others. This is a partial-invariance correction, not a global prior shift.

### Gaps
- No quantitative study of rank lag or rating inertia on Go servers was found. Glickman's peer-reviewed papers (Glicko, dynamic paired comparison) were not fetched.
- No study found measuring whether rating floors create gaps between label and ability at specific thresholds, beyond the USCF deflation statement.

---

## Q3. Psychometrics: which DIF, measurement-invariance and equating methods translate to "is the mapping from ability evidence to label different between two samples at one threshold?"

### Takeaway
The closest analogue is **differential step functioning (DSF)** for polytomous (ordered) items. It tests whether one step of an ordered response (here A|1D) behaves differently between groups (train vs test) after conditioning on a matching variable (the model's ability score). It complements item-level DIF, which gives one number per item. In threshold models this is "threshold non-invariance", with partial invariance (other thresholds as anchors). Operational tools include Mantel–Haenszel and logistic-regression DIF (ordinal LR in `lordif`), purification (iteratively excluding flagged items from the matching score), and anchor-based equating (NEAT) with checks for item parameter drift. Asymptotic tests need focal-group sizes far above the handful of known test labels. With fewer than 10 labelled focal cases, use exact conditional or Bayesian versions.

### Cited Findings
- DSF: traditional polytomous DIF yields a single item-level index that does not show which score levels drive the effect. DSF tests invariance at each step of the response — [NCME ITEMS Module 29 (Penfield, Gattamorta & Childs)](https://ncme.org/wp-content/uploads/2025/10/Module-29-Differential-Functioning-III-Step-Functi-2.pdf)
- DSF estimators: Penfield (2007) common odds ratio estimator, J. Educ. Meas., doi 10.1111/j.1745-3984.2007.00034.x (as listed in [Kuzu & Gelbal reference list](https://dergipark.org.tr/en/download/article-cite-file/1221823/type/6)); three classes of nonparametric DSF effect estimators ([record](https://www.bohrium.com/paper-details/three-classes-of-nonparametric-differential-step-functioning-effect-estimators/811903583204147201-9216)); adjacent-category vs cumulative DSF estimators (Gattamorta & Penfield 2012, listed in the same reference list); a taxonomy of DSF patterns ([Penfield, Alvarez & Lee 2008](https://scholarship.miami.edu/esploro/outputs/journalArticle/Using-a-Taxonomy-of-Differential-Step/991031576972002976)); crossing DIF explained by DSF effects that change sign across steps ([Penfield 2010 record](https://www.bohrium.com/paper-details/explaining-crossing-dif-in-polytomous-items-using-differential-step-functioning-effects/811751900381708289-9216))
- Model-based recursive partitioning (rating-scale and partial-credit trees) can find which covariate splits produce DIF or DSF, without pre-specifying groups — [Abou El-Komboz, Zeileis & Strobl 2014](https://core.ac.uk/download/19450575.pdf). A 2023 comparison of DIF vs DSF detection found results sensitive to how sparse categories were combined — [Kuzu & Gelbal 2023](https://dergipark.org.tr/en/download/article-file/2843049)
- `lordif` runs an iterative hybrid ordinal-logistic-regression/IRT DIF analysis. It flags items by "Chisqr", "R2" (McFadden by default; Nagelkerke and Cox–Snell available) or "Beta" (change in the group coefficient). The default R² change threshold is 0.02 in current versions and 0.035 in v0.1-1. It iterates until the same items are flagged twice in a row (purification), unless anchor items are given. Reference: Choi, Gibbons & Crane (2011), J. Stat. Softw. 39(8), doi 10.18637/jss.v039.i08 — [lordif reference manual](https://search.r-project.org/CRAN/refmans/lordif/html/lordif.html), [lordif 0.1-1](https://www.rdocumentation.org/packages/lordif/versions/0.1-1/topics/lordif)
- Sample size:
  - MH and SIBTEST lost power and inflated Type I error at a reference:focal ratio of 1:0.10 — [Awuor dissertation, Virginia Tech](https://vtechworks.lib.vt.edu/bitstream/handle/10919/28321/RAA_ETD.pdf)
  - Logistic-regression and likelihood-ratio DIF tests rely on asymptotic χ², and penalised-likelihood or bootstrap alternatives have been proposed for small samples — [PMC5978487](https://pmc.ncbi.nlm.nih.gov/articles/PMC5978487)
  - Simulation studies in the search results reported that conditions inflating Type I error also inflate apparent power, and that MH and LR perform comparably for uniform DIF. These are snippet-level; I could not tie them to a specific paper among [dergipark 2505531](https://dergipark.org.tr/en/download/article-file/2505531) and [Frontiers in Education 2019](https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2019.00047/pdf)
  - No single authoritative minimum focal-group size was found.
- Ordered-categorical invariance: Wu & Estabrook (2016, Psychometrika 81(4):1014–1045) show that identification constraints on the configural baseline change what threshold and loading invariance tests mean, and derive tests that do not depend on the identification choice. Applied work tests threshold equality first — [doi 10.1007/s11336-016-9506-0](https://doi.org/10.1007/s11336-016-9506-0), [Cambridge](https://www.cambridge.org/core/journals/psychometrika/article/identification-of-confirmatory-factor-analysis-models-of-different-levels-of-invariance-for-ordered-categorical-outcomes/7B76C5362BCE2F6380635391AA36A1BC)
- Equating and drift:
  - The NEAT (common-item non-equivalent groups) design assumes anchor items function identically across groups and forms. Drifted anchors distort the equating transformation, and the effect compounds in chain equating — [Michaelides 2010, Frontiers in Psychology](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2010.00167/full)
  - An ACT report compared five methods for detecting unstable anchor parameters, judged by their effect on equated scores, and found detection power depends on sample size, drift magnitude and number of common items — [ACT R2153 (2022)](https://act.org/content/dam/act/unsecured/documents/2022/R2153-Finding-Stability-Detecting-Unstable-Item-Parameters-in-IRT-Equating-2022-03.pdf)
  - A large-scale exam study used MH and LR DIF on anchors and flagged 22 drifting anchors. Snippet-level; URL is the likely source — [J. Psychol. Sci. 2012](https://jps.ecnu.edu.cn/EN/Y2012/V35/I2/446)

### Inferences
- (inference) **Mapping:**
  - group g = train/validation vs test;
  - "item response" = the 10-level label;
  - matching variable s = the model's continuous ability score from the games (the analogue of the rest score or θ);
  - cumulative-logit DSF model: logit P(label ≥ k | s, g) = α_k − βs + δ_k·g.
  - Uniform DIF (all δ_k equal) is a global label or prior shift. **DSF at one step means δ_{A|1D} ≠ 0 with the other δ_k = 0.**
  - A "bias-only" logit push between 1D and A is exactly an estimate of δ_{A|1D} under partial invariance. Boundaries where adjustments failed (B|A, 5D|6D) act as anchors that pin the scale.
- (inference) **Small-sample test.** With only probed test labels, use the exact conditional analogue of MH. Under H0 (δ = 0), each probed question i has a validation-calibrated p_i = P(label = A | s_i), and the number of A labels follows a Poisson-binomial distribution. Illustration only, not the project's numbers: if all eight p_i were ≤ 0.3, P(8/8 A) ≤ 0.3⁸ ≈ 6.6×10⁻⁵; at p_i = 0.5 it is 0.0039. A Bayesian version puts a prior on δ_{A|1D} and gives its posterior. The posterior applies only to the slice of s that was probed.
- (inference) **Selection caveat.** DIF statistics assume focal examinees are sampled independently of the item response. Probed questions were chosen *because* the model predicted 1D (selection on s), so they condition on s correctly, as MH does. They cannot speak to s regions that were never probed, such as A-labelled questions predicted 2D.
- (inference) **Purification analogue.** If the matching score s is itself trained on labels that are non-invariant at A|1D, estimate s with that boundary excluded or reweighted, then retest, as `lordif` does.
- (inference) **Equating analogue.** A NEAT design needs anchors on the label scale observed in both groups. If the per-game handicap (HD) or the opponent's strength reflects the server rank *at game time*, it could serve as a common anchor. Check whether the label-vs-anchor relationship differs between train and test at A|1D, which would also show whether label time and game time are misaligned. This depends on what HD encodes in this dataset (not established here).

### Gaps
- No DSF power or sample-size guidance found for focal n ≈ 5–30. The asymptotic literature does not apply at that size.
- No published method found for DIF/DSF when the focal group's responses are revealed only through an aggregate score (leaderboard feedback) instead of item-level labels.

---

## Q4. Epidemiology and diagnostics: misclassification and prevalence correction, latent-class models, spectrum effects, transportability

### Takeaway
Four tools map onto the problem:
1. **Rogan–Gladen, and its multiclass and ML forms (BBSE, Saerens EM)**, correct predicted class proportions with a confusion matrix estimated elsewhere. Both assume the error rates (p(x|y)) carry over. A boundary-specific label gap violates exactly that assumption, so these methods are better used to *detect* the violation than to correct it.
2. **Latent-class (Hui–Walter)** models estimate error rates without a gold standard from two populations with different prevalence, but they need conditionally independent tests.
3. **Spectrum effect**: accuracy at a threshold depends on the case mix *within* a category, the best epidemiological description of "test A players sit nearer 1D than training A players".
4. **Transportability**: performance in a new population is identifiable only under an untestable conditional-independence assumption. Exponential-tilt sensitivity analysis is the standard way to bound departures, and a one-boundary tilt is the same object as the DSF parameter δ_{A|1D}.

### Cited Findings
- Rogan & Gladen (1978), "Estimating prevalence from the results of a screening test", Am J Epidemiol 107(1):71–76. Formula: P = (P_obs + Sp − 1)/(Se + Sp − 1), the ML estimate when Se and Sp are known. It can fall outside [0,1] and treats Se and Sp as known constants — [PMC7690961](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7690961/), [medRxiv preprint](https://medrxiv.org/content/10.1101/2022.11.24.22282720v4.full.pdf), [delicatessen docs](https://deli.readthedocs.io/en/latest/Reference/generated/delicatessen.estimating_equations.measurement.ee_rogan_gladen.html)
- Rogan–Gladen was the most common correction in COVID-19 seroprevalence studies (57%, 45%–69%) — [Int J Public Health 2025 systematic review](https://www.ssph-journal.org/articles/10.3389/ijph.2025.1608343/full)
- Saerens, Latinne & Decaestecker (2002), Neural Computation 14(1):21–41: EM re-estimation of new priors and rescaling of posteriors without refitting, which maximises the likelihood of the new data. EM beat the confusion-matrix method for prior estimation — [MLAnthology](https://mlanthology.org/neco/2002/saerens2002neco-adjusting), [PDF](https://dipot.ulb.ac.be/dspace/bitstream/2013/68391/1/Decaestecker_NeuralComp02.pdf)
- BBSE (Lipton, Wang & Smola, ICML 2018) estimates test p(y) by solving Ax = b, where A is the predictor's confusion matrix on source data and b is the mean prediction on test. It works for biased or uncalibrated predictors if A is invertible, under the label-shift assumption p(x|y) = q(x|y). It also gives a statistical test for label shift and an importance-weighted correction — [arXiv 1802.03916](https://arxiv.org/abs/1802.03916)
- Hui–Walter: two tests applied in two populations with different prevalence make six parameters (2 Se, 2 Sp, 2 prevalences) identifiable, assuming the tests are conditionally independent given the true state. Bayesian versions use Beta priors and MCMC. A 2024 paper applies the Hui–Walter paradigm to estimating ML classifier performance online from unlabelled data — [arXiv 2401.09376](https://arxiv.org/pdf/2401.09376). An identification caveat (two populations with distinct prevalence are needed, and latent classes need external labelling) comes from a vendor guide — [MetricGate docs](https://metricgate.com/docs/latent-class-no-gold-standard/)
- Spectrum effect:
  - Ransohoff & Feinstein (1978) introduced spectrum problems in diagnostic evaluation — [Citizendium](https://www.citizendium.org/wiki/Spectrum_bias), [Wikipedia](https://en.wikipedia.org/wiki/Spectrum_bias)
  - Mulherin & Miller (2002, Ann Intern Med 137:598–602) argue for "spectrum effect" and recommend sensitivity and specificity *stratified* by subgroup, likelihood ratios and ROC curves — [abstract record](https://www.bohrium.com/paper-details/spectrum-bias-or-spectrum-effect-subgroup-variation-in-diagnostic-test-evaluation/813234661051334656-5322)
  - Goehring et al. (2004) found sensitivity and specificity often move in opposite directions across subgroups, which can leave likelihood ratios stable — [CiNii record](https://cir.nii.ac.jp/crid/1363107370601766400)
- Transportability:
  - Steingrimsson et al. (Am J Epidemiol 2022) give conditions for estimating a model's target-population performance when only covariates are observed in the target, and define "prediction error modifiers" that require re-estimation — [PMC11004796](https://pmc.ncbi.nlm.nih.gov/articles/PMC11004796)
  - Steingrimsson, Robertson, Voter & Dahabreh (Biometrics 2024) use an exponential tilt model to analyse sensitivity to the untestable assumption that the outcome is independent of population given covariates — [arXiv 2306.08084](https://arxiv.org/pdf/2306.08084)

### Inferences
- (inference) **Rogan–Gladen/BBSE as a detector.** Restricted to the A/1D pair, the 8/8 probe result says that among test questions predicted 1D, the share truly A (the analogue of 1 − PPV) is much higher than validation implies. Under pure label shift, BBSE would attribute this to more A players in test. Under a boundary-specific gap, p(x | y = A) itself has moved toward 1D, which violates the BBSE assumption. A practical check is over-identification: fit BBSE with hard predictions, with soft predictions, and with two model families. Disagreement beyond sampling error, or negative or implausible weights, points to concept or spectrum shift rather than label shift. This is consistent with the project's earlier finding that a global Saerens EM prior shift did not help.
- (inference) **Spectrum effect is the best-fitting analogue.** Label A pools three ranks (3,000 RP). If the test's A players are enriched for 1級, the error rate at A|1D rises with no change in the model and no change at B|A. The epidemiological remedy is to report and correct error *stratified by a severity proxy*: here, the within-A position of the model score s. That points to a correction that depends on s near the boundary (shift the A|1D cut only where s is just above it), not a constant shift.
- (inference) **Hui–Walter.** In principle, two predictor families plus two populations with different A/1D mixes (train vs test) could identify test-specific A|1D error rates without test labels. The conditional-independence assumption is likely violated: the project memory records that the two model families share about 97% of player-level bias. Expect biased, overconfident estimates unless the dependence is modelled.
- (inference) **One-boundary exponential tilt.** P_test(y | x) ∝ P_train(y | x)·exp(γ·1[y ≥ 1D]) applied only near the A|1D margin is the transportability form of δ_{A|1D}. γ can be bounded by the probe labels, and the decision is sensitive to γ over a plausible range instead of a point estimate. This links Q3 and Q4: DSF estimation and transport sensitivity analysis are the same computation.

### Gaps
- No ordinal (multi-threshold) Rogan–Gladen variant or threshold-specific misclassification correction with fewer than 10 validation labels was found.
- The full text of Mulherin & Miller and of Steingrimsson 2022 was not read. Statements rest on abstracts and records.

---

## Q5. Post-deployment monitoring of medical and deployed AI: detecting threshold- or subgroup-specific drift with few labels

### Takeaway
The recent position is to frame monitoring as statistically valid hypothesis tests, kept label-efficient:
- check label shift first, then concept drift;
- run label-free CUSUM on prediction rates;
- run subgroup CUSUM where labels exist;
- detect calibration drift with adaptive windows;
- choose labels actively and reweight to stay unbiased (active testing);
- combine many predictions with few labels (prediction-powered inference).

No validated method was found that does subgroup- or threshold-specific detection with fewer than 10 labels. Combining label-free distribution checks with an exact or Bayesian test on the probed slice is the practical frontier.

### Cited Findings
- Dolin, Li, Dasarathy & Berisha (arXiv 2506.05701) argue that post-deployment monitoring of clinical AI "is underdeveloped" and that only 9% of FDA-registered AI health tools include a surveillance plan. Their framework:
  - two-sample hypothesis tests for data shift and performance, checking label shift first (using BBSE) before testing for concept drift;
  - performance-deviation and specification-threshold tests, plus "prediction correctness monitoring" to catch localised degradation while overall accuracy is stable;
  - subgroup discovery as an optimisation with a minimum group size;
  - anytime-valid FDR control (SAFFRON, e-BH);
  - active labelling (Li et al.) and surrogate labels;
  - CUSUM, EWMA and Shewhart charts.
  — [arXiv 2506.05701v3](https://arxiv.org/html/2506.05701v3)
- Subgroup CUSUM on sensitivity of breast-cancer detection AI identifies drifts in underperforming subgroups quickly; false alarms and detection delay are reported via bootstrapped batches — [Kundu et al., arXiv 2502.10562](https://arxiv.org/pdf/2502.10562)
- Label-free CUSUM on the test negativity rate of mammography AI detected performance drift without ground truth — [Univ. of Miami record](https://scholarship.miami.edu/esploro/outputs/journalArticle/Detecting-Performance-Drift-in-AI-Models/991033137828302976)
- Other strands:
  - subgroup-stratified equalised-odds CUSUM (KAISEN; preprint, self-described initial probe) — [arXiv 2607.28608](https://arxiv.org/pdf/2607.28608)
  - score-based CUSUM with dynamic control limits for clinical prediction models, including delayed detection when clinicians over-trust the model — [arXiv 2211.09781](https://pdf.arxiv.org/pdf/2211.09781)
- Calibration drift: Davis, Greevy, Lasko, Walsh & Matheny (J Biomed Inform 2020, 112:103611) maintain dynamic calibration curves by online SGD and use an adaptive sliding window (ADWIN) as a one-sided test for increasing miscalibration. The window it returns, mostly post-drift data, is a candidate for model updating — [PubMed 33157313](https://pubmed.ncbi.nlm.nih.gov/33157313/), [PMC8627243](https://pmc.ncbi.nlm.nih.gov/articles/PMC8627243)
- Active testing (Kossen, Farquhar, Gal & Rainforth, ICML 2021) chooses which test points to label. Active selection introduces bias, which is removed by importance-type weighting that also reduces variance — [PMLR v139](https://proceedings.mlr.press/v139/kossen21a.html), [arXiv 2103.05331](https://arxiv.org/abs/2103.05331)
- Prediction-powered inference (Angelopoulos, Bates, Fannjiang, Jordan & Zrnic, Science 2023, 382:669) combines abundant predictions with a small gold-standard set to give valid confidence intervals and p-values for means, quantiles and regression coefficients. Better predictors give tighter intervals. PPI++ improves efficiency — [arXiv 2301.09633](https://arxiv.org/abs/2301.09633), [PPI++ arXiv 2311.01453](https://arxiv.org/abs/2311.01453)

### Inferences
- (inference) **Label-free first look.** The analogue of the "test negativity rate" is the test distribution of the A-vs-1D margin, logit P(1D) − logit P(A), compared with its distribution on validation questions with the same predicted label mix. Excess mass just on the 1D side of the cut is a label-free signal that something at A|1D differs. It cannot distinguish "test players really are 1D" from "labels lag". Only labels, or the probes, can.
- (inference) **Probes are actively selected labels.** PPI's guarantees assume the labelled subset is a random (or known-probability) sample. Leaderboard probes chosen because the model predicted 1D are not. Either weight each probe by its inverse selection probability (the active-testing idea; this requires writing the selection rule down in advance), or restrict conclusions to the selected slice (the conditional DSF view of Q3).
- (inference) **Multiplicity.** Each probed boundary (B|A, A|1D, 5D|6D, …) is a separate monitor. With several probes per boundary, e-value or FDR control in the style of Dolin et al. guards against adopting a chance success. Given the project's own adoption rule (difference ≥ 0.0025 plus fresh-seed replication), likelihood-ratio or posterior reporting per boundary fits better than p-values.
- (inference) **Prediction correctness monitoring at a slice** (did the error rate move within the A/1D slice while overall accuracy stayed flat?) is the clinical analogue of what the 8/8 probe showed. Their recommendation to localise drift by subgroup search maps onto scanning s-bins or covariates (game count per question, handicap presence) for where the A|1D gap concentrates.

### Gaps
- No clinical-monitoring method found that is designed for fewer than 10 labels, or for labels revealed only through an aggregate score.
- The KAISEN and Miami CUSUM items rest on abstracts and snippets; full methods were not read.

---

## Q6. Synthesis: what each field implies for detecting and correcting the A|1D gap with only model predictions on the unlabelled test set plus a few known labels

### Takeaway
All fields reduce to the same object: a **boundary-specific shift δ_{A|1D}** in the mapping from ability evidence to label, with the other boundaries as anchors. They differ in what they estimate it from:
- label-free distribution checks (monitoring, BBSE consistency) can only *detect* that something differs at A|1D;
- probed labels, analysed with exact or Bayesian conditional tests (DSF), *estimate* δ for the probed slice;
- spectrum and transport sensitivity analysis says how far to *extrapolate* δ beyond that slice.

Yugen's published rules show no server-side shodan gate. The most plausible source of an asymmetry found only at A|1D is the competition's grouping (A = 3 ranks, 1D = 1 rank) interacting with the undisclosed representative-value rule and the composition of the test sample.

### Cited Findings
- Each component is cited in Q1–Q5:
  - no special 初段 rule and smooth step sizes — [幽玄の間 レーティング](https://u-gen.nihonkiin.or.jp/static/manual_G03.asp)
  - step-level DIF (DSF) — [NCME Module 29](https://ncme.org/wp-content/uploads/2025/10/Module-29-Differential-Functioning-III-Step-Functi-2.pdf)
  - label-shift estimators and their assumption — [Lipton et al. 2018](https://arxiv.org/abs/1802.03916)
  - spectrum effect and stratified accuracy — [Mulherin & Miller record](https://www.bohrium.com/paper-details/spectrum-bias-or-spectrum-effect-subgroup-variation-in-diagnostic-test-evaluation/813234661051334656-5322)
  - exponential-tilt transport sensitivity — [Steingrimsson et al.](https://arxiv.org/pdf/2306.08084)
  - active-testing reweighting — [Kossen et al. 2021](https://proceedings.mlr.press/v139/kossen21a.html)
  - prediction-powered inference — [Angelopoulos et al. 2023](https://arxiv.org/abs/2301.09633)
  - FIDE's targeted, order-preserving rescale as a real-world precedent for fixing one part of a scale — [Chess.com](https://www.chess.com/news/view/fide-adds-rating-points-to-more-than-300-000-players)

### Inferences
(inference) Mapping table:

| Field / method | What it estimates | Our analogue | Needs | Can tell us | Cannot tell us |
|---|---|---|---|---|---|
| Rating systems: expected-vs-actual by rating gap (Sonas) | scale deflation in a region | label mix vs model score s, per boundary | labels per bin | where on the scale the mapping bends | why (lag vs composition) |
| Rating systems: Yugen rules | server-side gates | none at 初段; rule breaks only at 18/19級, 6段, 8段 | public rules | rules out a shodan gate | the representative-value rule |
| DSF / partial threshold invariance | δ_k per step, others anchored | δ_{A\|1D} with B\|A and 5D\|6D as anchors | probe labels conditioned on s | size of the boundary shift in the probed s-range | shift outside probed range |
| Exact Poisson-binomial / Bayesian DSF | P(observed probe outcome \| δ = 0) | p-value or posterior for δ_{A\|1D} | calibrated p_i per probe | strength of evidence from very few probes | population-level effect |
| Purification (lordif) | contamination of matching score | retrain or recalibrate s excluding the A\|1D boundary | training data | whether s itself absorbs the gap | — |
| NEAT equating anchors | common-scale link | HD / opponent strength at game time, if it encodes rank | per-game anchor field | timing mismatch between label and games | — (depends on what HD encodes) |
| Rogan–Gladen / BBSE / Saerens | test class proportions under label shift | test A vs 1D share | confusion matrix from validation | label shift alone, if p(x\|y) holds | wrong under boundary-specific concept shift; use as an over-identification check |
| Hui–Walter latent class | error rates without gold standard | two model families × train/test | conditionally independent families | test-specific A\|1D error rates | biased if families share errors (likely here) |
| Spectrum effect | case-mix-dependent accuracy | within-A position of s (1級-like vs 3級-like) | s distribution in test vs train | whether test A is enriched near the top | true labels |
| Transport + exponential tilt | sensitivity of target performance to unobserved shift | tilt γ on 1[y ≥ 1D] near the margin | probes to bound γ | decision robustness over a γ range | point identification |
| Label-free CUSUM / prediction-rate drift | shift in prediction rates | excess test mass near the A\|1D margin | predictions only | that the margin region differs | direction of truth |
| Active testing / PPI | unbiased performance with few selected labels | probe-based estimate of A\|1D error | known probe selection probabilities | population-level error with valid CI | anything if the selection rule is unknown |

(inference) A minimal workflow consistent with all six fields:
1. **Detect.** Compare the test vs validation distribution of the A/1D margin and of s within predicted-A and predicted-1D questions. Run BBSE with two predictors or two prediction types and check for consistency.
2. **Estimate.** Treat probes as conditioned on s. Compute the Poisson-binomial tail, or a δ_{A|1D} posterior, from validation-calibrated p_i, keeping B|A and 5D|6D fixed as anchors.
3. **Correct.** Apply the shift only at A|1D (partial invariance), preferably as an s-dependent shift concentrated just above the cut (spectrum view) rather than a constant one.
4. **Bound.** Report the decision's sensitivity over a γ range (exponential tilt). State that the extrapolation beyond the probed s-slice is assumption-driven.

(inference) Which representative-value rule is in force decides whether a correction should depend on question characteristics:
- a band-mode or early-snapshot label predicts larger gaps for players whose games show a strength *trend* (improvers);
- a median-RP label with a spectrum shift predicts gaps tied to s position, not to trend.

If the games within a question carry any trend signal, a DSF test with an interaction (δ_{A|1D} × trend) on the probes could distinguish the two. Eight probes are likely too few, so this is a design for the next probes, not a conclusion.

### Gaps
- None of the fields offers a validated small-n (fewer than 10 labels) method for estimating a boundary-specific shift. The exact or Bayesian DSF construction above is my synthesis, not a published procedure.
- The organisers' label rule and the timing of test games are the key unknowns; no public source answers them (see also `/home/u2507115/BK_AICUP_2026-1/research_notes/類似任務他人解法/aicup_context.md`, which records that the official pages never state how labels were assigned).

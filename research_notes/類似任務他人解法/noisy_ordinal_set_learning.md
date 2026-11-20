# 從少量雜訊觀測集合預測實體的序數類別：雜訊標籤、集合學習、決策規則、轉導、階層貝葉斯、集成（文獻與競賽實務調查）

調查日期：2026-11-05。對象問題：Go 段位預測，每位玩家 5–20 局，10 個序數類別，metric = 命中 1 / 相鄰 e^-1 / 其他 0；現行系統 CV 0.726，誤差分解為 65% 玩家層級持續偏移（兩個模型家族共享約 97%，屬 label/performance mismatch）+ 35% 抽樣。已知死路：temperature scaling、Saerens EM prior shift、balanced assignment、kernel/soft ordinal targets (K)、expected-score fine-tuning (ES)、label smoothing、stacking、PMA pooling、mixed-player training、per-n standardisation。

每節格式：Takeaway / Cited Findings / Inferences / Gaps。Cited Findings 一律附來源 URL；Inferences 是我依據引文對本問題做的推論，未必有文獻直接支持。

---

## Q1. 雜訊序數標籤的學習（loss correction、co-teaching、noise-transition 估計、label refurbishment）；測試標籤同樣有雜訊時是否仍有益？

### Takeaway
理論與最新 NeurIPS 2024 結果都指向：在只有雜訊標籤可用、且評分也用雜訊標籤時，「直接對雜訊標籤做 ERM」已是（近）minimax 最優；loss correction / 轉移矩陣估計 / 樣本篩選的收益來源都是「對乾淨標籤評分」或「有限樣本下避免記憶翻轉標籤」，在我們的設定（玩家層級、與測試共享的 instance-dependent 偏移）預期增益 ≈ 0，甚至為負。

### Cited Findings
- Patrini et al. 2017（CVPR）backward correction 用 T⁻¹ 加權損失、forward correction 將預測經 Tᵀ 後再算損失；兩者在 class-conditional noise 且 T 非奇異時，「雜訊分佈上的最小化子 = 乾淨分佈上的最小化子」；T 由「每類存在 perfect example」假設估計（取 97 百分位點）。CIFAR-10 ResNet-32 非對稱雜訊 N=0.6：CE 53.6% → forward(真 T) 87.6%、backward(真 T) 74.3%、forward(估計 T) 77.6%；CIFAR-100 N=0.6 CE 17.1% → forward(真 T) 68.4%，但 forward(估計 T) 僅 15.9%（估計誤差吃掉全部增益）。測試一律用標準（乾淨）測試集，論文未討論測試標籤有雜訊的情形 — [Patrini et al. 2017, arXiv 1609.03683](https://ar5iv.labs.arxiv.org/html/1609.03683)
- Garg & Manwani 2020（ACML）首個針對序數回歸的雜訊標籤理論方法：提出 inversely decaying 雜訊模型（誤標到 j 的機率 ∝ ρ/|i−j|，相鄰段最易混淆），用 N⁻¹ 做 unbiased-estimator 損失修正，證明修正後仍 rank-consistent；雜訊率「已知 (KR)」或「用 Patrini 法估計 (EST)」。只測一個雜訊水準 ρ=0.15。Table 2 同時列「Clean Data」與「Noisy data」兩欄的 MAE（欄位意義為 clean/noisy 測試標籤屬推斷，caption 未明寫）：Synthetic l_CE 0.20 → KR 0.06 / EST 0.12；Abalone 0.46 → 0.44 / 0.47（EST 反而變差）；California 0.38 → 0.34 / 0.35；MSLR 0.63 → 0.55 / 0.62。乾淨標籤欄則幾乎不變（如 MSLR 0.55 → 0.55 / 0.53）。 — [Garg & Manwani 2020, arXiv 1912.03488](https://ar5iv.labs.arxiv.org/html/1912.03488)；[PMLR v129](https://proceedings.mlr.press/v129/garg20a.html)
- Zhu, Zhang, Gangrade, Scott, "Label Noise: Ignorance Is Bliss"（NeurIPS 2024）：針對 multi-class instance-dependent noise 定義 relative signal strength (RSS)；RSS>0 的區域即「雜訊 Bayes 分類器 = 乾淨 Bayes 分類器」的區域；對稱雜訊只要雜訊率 <(K−1)/K 即具 noise immunity。證明 Noise-Ignorant ERM（直接忽略雜訊訓練）的 excess risk 上界 ε+Õ((1/κ)√(V/n)) 與任何只用雜訊資料的學習規則之 minimax 下界 ((K−1)/K)ε+Ω((1/κ)√(1/n)) 相匹配，即「沒有任何只用雜訊標籤的方法能在 minimax 意義下顯著勝過 NI-ERM」。實驗：DINOv2 特徵 + 線性分類器 + NI-ERM，CIFAR-10N Aggre 98.69 vs ProMix 97.65，Rand1 98.80 vs 97.39，CIFAR-100N 83.17 vs 73.79；但 CIFAR-10N Worst 95.71 落後 ProMix 96.34。論文未個別分析 loss correction / sample selection 何時能勝出。 — [arXiv 2411.00079](https://arxiv.org/html/2411.00079v1)；[NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/hash/d3696c79d572c995a74eac78037551a8-Abstract-Conference.html)
- Chen et al. 2022 "Deep Learning is Provably Robust to Symmetric Label Noise"：對稱雜訊下，基於雜訊後驗 q_k(x) 的 Bayes 決策與乾淨後驗 p_k(x) 相同，直到資訊理論門檻（二元 1/2，多類 (K−1)/K）。 — [arXiv 2210.15083](https://arxiv.org/pdf/2210.15083)
- Binary classification with instance- and label-dependent noise（2023）：僅憑雜訊樣本、無乾淨樣本或強分佈假設時，學習（乾淨目標）不可能。 — [arXiv 2306.03402](https://ar5iv.labs.arxiv.org/html/2306.03402)
- Northcutt, Athalye, Mueller 2021 "Pervasive Label Errors in Test Sets Destabilize ML Benchmarks"：10 個常用測試集平均至少 3.3% 標籤錯誤（ImageNet val ≥6%）。以修正後標籤評分時，只要原始誤標樣本佔比提高 6%（ImageNet）/5%（CIFAR-10），ResNet-18 就勝過 ResNet-50、VGG-11 勝過 VGG-19 — 即「用原始雜訊測試標籤評分」會系統性偏好高容量、更能擬合雜訊的模型。 — [arXiv 2103.14749](https://arxiv.org/abs/2103.14749)
- Co-teaching（Han et al. 2018）：兩網路同時訓練，各自挑選「可能乾淨」的樣本交給對方更新；實驗在加噪的 MNIST/CIFAR-10/100 上，摘要未提供數字；頁面未說明測試標籤是否乾淨（標準做法為乾淨測試）。 — [arXiv 1804.06872](https://arxiv.org/abs/1804.06872)

### Inferences
- 決策論：若 metric 的報酬矩陣為 R(真類 j, 預測 k)，而評分用的真類是「雜訊標籤 ỹ」，則 Bayes 最優決策為 argmax_k Σ_j q(j|X) R(j,k)，其中 q 是**雜訊標籤**的後驗。直接分類器（class-balanced whole-question CE）估的就是 q；forward/backward correction 估的是乾淨後驗 p，在 class-conditional 假設下 q = Tᵀp，修正後若要對 ỹ 評分還得「再加噪回去」，最多回到原點。這與 Zhu et al. 2024 的 minimax 結果一致：噪聲建模無法在只用雜訊標籤時取得率上的改進。
- Garg & Manwani 的「Noisy data」欄仍見改善並不矛盾：他們的雜訊是合成、argmax-preserving（ρ=0.15、對角佔優），因此「預測乾淨標籤」同時就是「預測雜訊標籤的眾數」；修正只是防止小資料集（Boston ≈500、Abalone）記憶翻轉標籤的正則化作用。我們 65% 的誤差是**玩家層級持續偏移**、兩個模型家族共享 97%，屬 instance-dependent、且**在測試集同樣存在**；它不是訓練資料可被「記憶」掉的隨機翻轉，故此正則化效應不適用。
- Co-teaching / label refurbishment / sample selection 會把「高損失樣本」（= 標籤與棋力不符的玩家）丟棄或改標；但測試集也含同比例這類玩家，而他們的「正確」預測在雜訊評分下就是那個不符的段位 — 這些方法會把模型推離 q 而靠近 p，對我們的 metric 預期為負。Northcutt 2021 的「雜訊測試偏好擬合雜訊的模型」亦佐證：對雜訊測試評分時，不應去噪。
- 唯一可能有用的形式：把雜訊模型當 **structured output head**（例如預測 p 再經可學習的 ordinal 轉移 Tᵀ 得到 q，T 參數化為 Garg 的 inversely-decaying 形式）。這在數學上與 kernel/soft ordinal targets (K) 幾乎等價，而 K 已是死路。預期增益 0 ± 0.001；成本低；與死路 K、label smoothing 高度重疊。
- 推斷結論：**不建議投入**。

### Gaps
- 未找到任何論文明確以「雜訊測試標籤」作為評分設定來比較 noise-robust 方法與直接分類器（Garg & Manwani Table 2 的 Noisy 欄是唯一近似，且欄位意義為推斷）。
- 未找到針對「實體層級持續偏移（同一實體所有實例共享同一標籤偏差）」的噪聲模型文獻；此型態介於 class-conditional 與 instance-dependent 之間，現有 CIFAR-N 類 benchmark 不含此結構。
- Co-teaching 的數值結果未取得（僅摘要）。

---

## Q2. Deep Sets / MIL / Set Transformer 於小集合（5–20 實例）的實體層級預測：mean vs attention vs noisy-OR、set-size conditioning、隨機子集大小訓練

### Takeaway
在平均每袋 5–7 個實例的經典 MIL 基準上，attention pooling 與 mean/max 類方法只是「comparable」而非更佳；attention/Set Transformer 的明確勝出都出現在需要實例間互動（計數相異字元）或偵測單一異常實例的任務。段位預測更接近「估計平均品質」，mean/sum 類 pooling 已足夠，PMA 死路與文獻一致。隨機 5–20 子集訓練已涵蓋測試範圍，無文獻顯示再做 set-size conditioning 有系統性增益。

### Cited Findings
- Ilse, Tomczak, Welling 2018（ICML，Attention-based Deep MIL）：經典 MIL 基準的袋大小（instances÷bags，自算）Musk1 ≈5.2、Tiger ≈6.1、Fox ≈6.5、Elephant ≈7.0、Musk2 ≈64.7。準確率：Attention 0.892/0.858/0.615/0.839/0.868（Musk1/Musk2/Fox/Tiger/Elephant），Gated-Attention 0.900/0.863/0.603/0.845/0.857，而 MI-Net with DS 0.894/0.874/0.630/0.845/0.872 — 作者自述「comparable with the best performing classical MIL methods」，未宣稱勝出。MNIST-bags 上「attention-based deep MIL performs much better in the small sample size regime」，訓練集大時各法收斂到相近結果。注意力權重和為 1 以對袋大小不變。 — [arXiv 1802.04712](https://ar5iv.labs.arxiv.org/html/1802.04712)
- Lee et al. 2019 Set Transformer：計數相異字元（集合大小 6–10）：rFF+Pooling 0.4382、rFFp-mean 0.4617、rFF+PMA 0.4572、SAB+PMA 0.6037；CelebA 8 張一組異常偵測 AUROC：average pooling 0.5643、rFF+Dotprod attention 0.5671、rFF+PMA 0.5756、SAB+PMA 0.5941；max regression 任務 max pooling MAE 0.1355 反而優於 SAB+PMA 0.2085；point cloud：100 點時 ISAB+PMA 0.8454 > rFF+Pooling 0.7951，5000 點時 0.8779 < 0.8933，作者將大集合劣勢歸因於「大集合本身已含足夠資訊」；「PMA outperformed simple pooling in all other experiments」。 — [arXiv 1810.00825](https://ar5iv.labs.arxiv.org/html/1810.00825)
- Wagstaff et al. 2019 "On the Limitations of Representing Functions on Sets"：連續映射（神經網路）下，sum-decomposable 表示要能普遍逼近集合函數，潛在維度必須 ≥ 最大集合大小 M。 — [arXiv 1901.09006](https://arxiv.org/abs/1901.09006)
- Probabilistic Invariant Learning with Randomized Linear Classifiers（2023）：Deep Sets「known to have poor set size generalization」當測試集合比訓練集合大；只有 hidden 維度 = 集合大小時才達 universality。 — [arXiv 2308.04412](https://arxiv.org/pdf/2308.04412)
- 2026 CHIL 多資料集 3D 神經影像 MIL benchmark：以 mean pooling 的簡單 MIL baseline「can often match or outperform recent MIL methods」（含 attention-based MIL）。 — [arXiv 2604.26807](https://arxiv.org/html/2604.26807)
- Oner et al. "Studying the Effect of MIL Pooling Filters"：distribution-based pooling（估計實例特徵的分佈/直方圖）整體優於 mean/max 等 point-estimate pooling。 — [arXiv 2006.01561](https://ar5iv.labs.arxiv.org/html/2006.01561)
- Certainty Pooling（Gildenblat et al. 2020）：mean pooling 下大量負實例會淹沒正實例；max pooling 下單一高分負實例會污染袋級預測；smoothed-max（log-sum-exp）在 τ=0 為 max、τ=∞ 為 mean 之間插值。 — [arXiv 2008.10548](https://arxiv.org/pdf/2008.10548)；[arXiv 2305.08040](https://arxiv.org/pdf/2305.08040)
- 機器人變長輸入研究：mean pooling「able to encode both the presence and the ratios of the different observed instances, and thus performs well」，劣化發生在實例數變多時而非變少。 — [arXiv 1809.07480](https://arxiv.org/pdf/1809.07480)
- 使用集合大小 5 的 FACILE-FSP 研究：三種 set-input 模型「none is consistently better」，Deep Set 在較多任務上平均 F1 最高。 — [arXiv 2311.05659](https://arxiv.org/pdf/2311.05659)

### Inferences
- 我們每問 5–20 局、每局已由預訓練網路轉成 per-move likelihood 特徵，再經小型序列模型；袋大小與 Musk1/Tiger/Fox/Elephant（5–7）同量級，文獻在此量級下 attention 無系統性優勢，與 PMA 死路一致。
- 段位預測屬「每局是對同一潛在棋力的獨立雜訊量測」，最適 pooling 是對 per-game 證據做 **log-linear（product-of-experts）加總**或可靠度加權平均；這在結構上是 Deep Sets（sum pooling in log-posterior space）＋每局標量權重（= 最退化的 attention）。若現行序列模型已隱含做到，再換 pooling 形式的期望增益 <0.002。
- Noisy-OR 式 pooling 適合「袋中任一正實例即為正」的偵測問題（Set Transformer 的 CelebA 異常偵測即此類）；段位不是偵測問題，不適用。
- Oner 的 distribution pooling（per-game 統計量的直方圖/多階矩）是唯一在小袋上有系統性正結果的 pooling 改法，成本低（在現有 per-game 特徵上加 2–4 階矩或分位數），但其基準是病理影像，遷移性未知；期望增益 0–0.002。
- Wagstaff 界（潛在維 ≥ 20）在我們的隱藏維度下必然滿足；set-size 泛化問題只發生在測試集合大於訓練集合，我們以 5–20 隨機子集訓練、測試同範圍，無此問題。per-n standardisation 已是死路，與此一致。

### Gaps
- 未找到任何「系統性掃描袋大小（5–20）× pooling 類型」的受控研究；上述結論由多個固定袋大小的 benchmark 拼接而成。
- 未找到「隨機子集大小訓練 vs 固定大小訓練」的對照文獻；Set Transformer 僅在 6–10 隨機大小下訓練與測試。
- 未找到 set-size conditioning（將 n 作為輸入）之獨立消融。

---

## Q3. 相鄰加權序數 metric 的決策規則：threshold 最適化、argmax E[score]、如何利用已知「各類等量」的測試類別平衡

### Takeaway
QWK 競賽的閾值調整收益（+0.05 QWK 量級）來自回歸輸出與序數 metric 的錯配；當模型已輸出校準後驗並以 argmax E[score] 決策時，這部分收益已被吃完。利用「測試各類等量」的理論（OTTER, 2024）要求 (i) 訓練/測試先驗確實不同、(ii) 後驗已校準；若模型預測邊際已近均勻，最優傳輸指派「什麼也不改」，與 balanced assignment / Saerens EM 死路一致。

### Cited Findings
- DSB2019（QWK，4 類）合成資料實驗：squared-error XGBoost 以直接四捨五入 QWK 0.5806 → 5-fold CV 調閾值 0.6335（+0.053）；LightGBM 0.5797 → 0.6348；但用 QWK-aware 的 extended-QWK 目標訓練時，四捨五入 0.6350 vs 調閾值 0.6351（≈0）。作者註明為合成資料、無超參調整。 — [Zenn: DSB2019 QWK 直接最適化](https://zenn.dev/jackthekaggler/articles/cf988ca341e34ed83034)
- DSB2019 高名次方案（koukyo1994）：NN 以 one-vs-rest BCE 輸出 4 類機率，再與 arange(4) 內積轉成 pseudo-regression 值，作者稱「gives us better result compared to regression」；閾值以「截斷後 CV 最大化」決定；驗證時排除評估次數超過測試 95 百分位的使用者並以 1/(assessment count) 重新加權以模擬測試分佈；「public LB and OOF were not very helpful」。truncated CV 0.5818 / public 0.565 / private 0.557。 — [GitHub koukyo1994/kaggle-dsb2019](https://github.com/koukyo1994/kaggle-dsb2019)
- PetFinder（QWK，5 類）社群方案：以 Nelder-Mead 調 cut-points「increased our performance dramatically」，XGBoost 0.407 / LGBM 0.396；10 名方案的 QWK-optR 以 95 百分位設定最低類別切點後再套學習係數（分佈式閾值）；另一方案指出 OptimizedRounder 的「predicted distribution and the actual test data predictions were too different」。 — [Medium PetFinder write-up](https://medium.com/@meilingxt/petfinder-my-adoption-prediction-8c512c2abcb1)；[GitHub chizhu 10th](https://github.com/chizhu/kaggle-petfinder-adoption-prediction-10th-solution)；[Medium thom42](https://medium.com/@thom42/pet-cat-and-dog-adoption-speed-prediction-kaggle-petfindermy-part-2-model-a46b91a439bd)
- OptimizedRounder 套件化（Optuna 搜尋、預設 quadratic_kappa、可 CV）。 — [PyPI optimized-rounder](https://pypi.org/project/optimized-rounder/)
- Vaughn & Justice "On the Direct Maximization of QWK"：QWK 可改寫為較簡單的等價形式並直接最大化。 — [arXiv 1509.07107](https://arxiv.org/pdf/1509.07107)
- OTTER（Optimal Transport for label-distribution adaptation, 2024）：Theorem 4.2 在 label shift（P_s(X|Y)=P_t(X|Y)）、模型對來源分佈校準、無平手、且已知真實目標邊際 ν* 的條件下，OT 指派恢復目標分佈的 Bayes 分類器；Theorem 4.3 的額外誤差來自目標邊際誤差 Δν 與成本矩陣（校準）誤差 ΔC；**Theorem 4.1：若指定邊際等於模型自身的預測邊際，OTTER 的預測與原預測完全相同**。目標為 ⟨γ, −log scores⟩（非相鄰加權）。零樣本影像/文字任務平均 +4.8% / +15.9%。 — [arXiv 2404.08461](https://ar5iv.labs.arxiv.org/html/2404.08461)

### Inferences
- 我們的 argmax_k [q_k + e^{-1}(q_{k−1}+q_{k+1})] 已是該 metric 在模型後驗下的 Bayes 決策；QWK 競賽裡閾值調整的大收益是「回歸輸出→序數」的錯配修正（Zenn 的實驗顯示 metric-aware 訓練後閾值調整收益 ≈0），對我們不存在。
- 利用等量類別：OTTER 理論說明兩個前提在我們這裡皆不成立或已滿足 — 訓練用 class-balanced CE，先驗本已均勻（無 label shift 可修）；若 12 模型集成的預測邊際已近均勻，硬指派只會把邊際上的隨機抖動強制歸零、以犧牲相鄰命中為代價，與 balanced assignment 死路相符。溫和版（entropic Sinkhorn、或對邊際偏離加 λ 懲罰、成本改為相鄰加權 −log E[score]）在文獻中無證據，期望增益 ≤0.001，且與兩個死路重疊。
- 唯一可考慮的廉價檢查：確認集成的預測邊際與真實等量分佈之偏差是否顯著大於抽樣噪聲（多項式檢定）；若不顯著，即可正式關閉這條線。

### Gaps
- Kaggle 頁面（DSB2019 1st、PetFinder 1st、AES 2.0 1st、Child Mind PIU 1st）為 JS 渲染，WebFetch 只取得標題，無法引用官方第一名寫法；以 GitHub/Medium 轉述與高名次 repo 代替。
- 未找到針對「命中 1 / 相鄰 e^-1 / 其他 0」這類非 QWK 相鄰加權 metric 的決策規則文獻；此類 metric 下 argmax E[score] 為標準 Bayes 決策，無需額外文獻。
- 未找到「已知測試類別完全等量」被成功利用的競賽案例（有的是分佈式閾值，但其收益與回歸錯配混雜）。

---

## Q4. 轉導 / 測試時方法：測試集偽標籤、test-time training、entropy minimisation；訓練與測試同分佈時的風險

### Takeaway
偽標籤在 Kaggle 的收益沒有先驗保證（The Kaggle Book 明言須實測、可能變差）；TTT/entropy minimisation 的收益全部來自分佈偏移，近期工作指出 entropy minimisation 在無偏移時可能劣化並以「在乾淨資料上測試不劣化」作為必要檢查。我們訓練/測試同分佈且標籤同噪，轉導方法的理論增益來源不存在；偽標籤最多提供自蒸餾式的小幅平滑。

### Cited Findings
- The Kaggle Book：無法事先知道偽標籤是否有效，須自行驗證；「Adding pseudo-labels won't change the result; it may even worsen it if there is some added noise in the pseudo-labeled data」；技術首見於 Santander Customer Transaction Prediction 的 Wizardry 隊。 — [Packt, The Kaggle Book, Pseudo-labeling](https://subscription.packtpub.com/book/data/9781801817479/9/ch09lvl1sec52/pseudo-labeling)
- NVIDIA Kaggle Grandmasters Playbook：建議 soft pseudo-labels（「add more signal, reduce noise, and let you filter out low-confidence samples」）、k-fold 下產生 k 組偽標籤避免洩漏、最後只在原始標籤上 fine-tune 以去除偽標籤噪聲；舉 BirdCLEF 2024 為例但**無數字**；同文：100 顆不同 seed 的 XGBoost 平均 MAP@3 ≈0.379 vs 單 seed 平均 ≈0.376（+0.003）；全資料重訓「another leaderboard bump」（無量級）。 — [NVIDIA Developer Blog](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- Arazo et al. 2020 "Pseudo-Labeling and Confirmation Bias"：天真偽標籤會過擬合自己的錯誤偽標籤（confirmation bias）；mixup 與每 mini-batch 最少標註樣本數可緩解。 — [arXiv 1908.02983](https://arxiv.org/abs/1908.02983)
- DIPS 2024：以資料中心方式篩選偽標籤，在 12 個表格資料集上改善多種偽標籤演算法。 — [arXiv 2406.13733](https://arxiv.org/pdf/2406.13733)
- POEM（NeurIPS 2024）：摘要稱其「maintaining accuracy and calibration in their absence [of shift], outperforming leading entropy minimization methods」；alphaXiv 摘要頁進一步稱 entropy minimisation「can lead to overconfident predictions, model collapse, and unnecessary adaptation even when no distribution shift exists」（此強措辭來自摘要頁，非論文原文直接引述）。 — [arXiv 2408.07511](https://arxiv.org/abs/2408.07511)；[alphaXiv overview](https://alphaxiv.org/overview/2408.07511v2)
- LookSharp 2025：「we also perform TTA on clean data to ensure our method maintains performance without distribution shift」— 將「無偏移時不劣化」列為 TTA 方法的必要檢查。 — [arXiv 2511.18925](https://arxiv.org/html/2511.18925v3)
- REALM 2023：entropy minimisation 類策略因噪聲樣本而失效，須跳過不可靠樣本。 — [arXiv 2309.03964](https://arxiv.org/pdf/2309.03964)
- Test-Time Training（Sun et al. 2020）：把單一測試樣本變成自監督問題後更新參數；收益報告於分佈偏移 benchmark；摘要未述無偏移時的得失。 — [arXiv 1909.13231](https://arxiv.org/abs/1909.13231)
- Northcutt 2021（同 Q1）：對雜訊測試標籤評分時，較能擬合雜訊的高容量模型得分較高。 — [arXiv 2103.14749](https://arxiv.org/abs/2103.14749)

### Inferences
- 我們沒有分佈偏移（測試 = 另一組同來源玩家），TTT / entropy minimisation 的機制前提不成立；entropy minimisation 會把後驗推向尖峰，直接削弱 argmax E[score] 對相鄰類的保護，預期為負。
- 偽標籤在同分佈下等價於自蒸餾（soft label 版）或半監督資料擴增；可能的小收益來自 (a) 測試集若顯著大於訓練集，增加特徵分佈覆蓋；(b) 蒸餾的平滑效果。confirmation bias 在我們這裡的具體形式是「把測試玩家的 model 預測當真，強化模型自身的玩家偏移」，對雜訊評分而言方向中性（Northcutt：擬合雜訊不會被雜訊測試懲罰），但也不會修正 65% 的持續偏移。
- 可行性評估：這是死路清單中**尚未出現**、且在競賽中有實務紀錄的唯一轉導手段。建議做法為 NVIDIA 流程（soft labels、k 組偽標籤、最後只用原標籤 fine-tune）；期望增益 0–+0.003（對應 CV 0.726 的 0–0.4%），成本 = 重訓 12 模型一輪。若測試集規模 ≤ 訓練集，期望值取下緣。

### Gaps
- 未找到「同分佈、小規模測試集」下偽標籤收益的量化基準；所有正面案例皆為大量未標註資料或存在偏移（BirdCLEF、AES 2.0 的 Persuade 語料重標）。
- AES 2.0 第一名「以偽標籤修正 Persuade 子集分數」的細節僅見部落格轉述（無法存取 Kaggle 原帖）。 — [The Scandal at Kaggle (Substack)](https://datancoffee.substack.com/p/the-scandal-at-kaggle)

---

## Q5. 校準與跨實體聚合：階層貝葉斯收縮、empirical Bayes；當實體偏移不可由觀測預測時，能否降低？

### Takeaway
確認「否」：與觀測條件獨立的實體層級標籤偏移是 aleatoric（不可消去）成分，任何只以觀測為輸入的函數（包括收縮估計）都無法降低它；Bayes error 估計文獻亦顯示強模型可以「低於」依乾淨標籤估出的 Bayes error，原因正是測試標籤本身的噪聲。收縮只能減少 35% 的抽樣成分，而 class-balanced 訓練 + 集成已隱含均勻先驗，再做 EB 收縮會重複計算先驗。

### Cited Findings
- Kendall & Gal 2017：「Aleatoric uncertainty captures noise inherent in the observations」；epistemic 是「uncertainty which can be explained away given enough data」— 僅後者可用更多資料消去。 — [arXiv 1703.04977](https://arxiv.org/abs/1703.04977)
- Ishida et al. 2023（ICLR notable top-5%）：Bayes error 估計量 β̂ = (1/n) Σ min{c_i, 1−c_i}（軟標籤均值），model-free、無超參；ViT-B/16 在 CIFAR-10/C10F/C10.1 的原始測試誤差**低於**以 CIFAR-10H 軟標籤估出的 Bayes error，作者歸因於標籤差異（C10H 以降採樣影像標註，標註者較不確定，使 β̂ 偏高）；改以 C10H 重抽標籤評分時 ViT 誤差「on par with or slightly higher than the estimated Bayes error」；Fashion-MNIST-H ResNet-18 3.852%±0.041 vs 估計 Bayes error 3.478%±0.079。 — [arXiv 2202.00395](https://ar5iv.labs.arxiv.org/html/2202.00395)
- Northcutt 2021（同 Q1）：雜訊測試標籤使模型排名在誤標率僅 5–6% 時即翻轉。 — [arXiv 2103.14749](https://arxiv.org/abs/2103.14749)
- Assumption-Lean Shrinkage（2026）：每個潛在參數只有單一雜訊估計時，交叉驗證無法確定相對真值的誤差。 — [arXiv 2606.12324](https://arxiv.org/pdf/2606.12324)
- Aurora（Ignatiadis et al.）：每單位多個重複雜訊觀測可在極少分佈假設下估計單位均值，坍縮成單一平均會失去此優勢；Stanford 論文：先驗與似然皆未知時需 K>1 觀測/實體才能近 Bayes 最優。 — [arXiv 1911.05970](https://ar5iv.arxiv.org/html/1911.05970)；[Stanford purl sy052fc9059](https://purl.stanford.edu/sy052fc9059)
- Empirical Bayes 教學：收縮強度隨訊噪比調整，樣本少的單位被拉向總平均最強；先驗估計通常需 ≥20 個單位。 — [metricgate EB shrinkage](https://metricgate.com/blogs/empirical-bayes-shrinkage-estimation/)
- 類比任務 — 西洋棋評分預測：RatingNet（CNN-LSTM，棋步 + 時鐘）單局 MAE 182（無時鐘 239、預測均值 346）；作者提醒結果「need a grain of salt」因訓練資料中較高評分者只贏 53%，並引述評分系統對高評分者過度樂觀的證據；未研究多局聚合。 — [arXiv 2409.11506](https://arxiv.org/html/2409.11506)
- 類比任務 — Kaggle Finding Elo（2015，單局預測雙方 Elo，25k 訓練/25k 測試，主辦方提供 Stockfish 逐手評估）第二名 MAE ≈160：per-move RandomForest（棋步、分數、ply、棋子、與 Stockfish 最佳步比較）+ 局級 OLS→RF 堆疊，特徵含 25 百分位最差步的 centipawn loss、Guid-Bratko 複雜度、開局庫 Elo 統計；時間控制資訊值約 5 分但測試無此資訊；加入 5 萬局選取方式不同的外部資料反而變差。 — [GitHub scchess/KaggleBlunderCheck](https://github.com/scchess/KaggleBlunderCheck)；[GitHub rozim/KaggleFindingElo](https://github.com/rozim/KaggleFindingElo/blob/master/stockfish.py)

### Inferences
- 形式化：令 y_p = f(θ_p) + ε_p，θ_p 為真實棋力、X_p 為該玩家觀測到的棋局集合。若 ε_p ⊥ X_p | θ_p（偏移不可由觀測預測，兩個模型家族共享 97% 即此證據），則 E[y_p | X_p] = E[f(θ_p) | X_p]，ε_p 的變異完整進入不可約誤差；收縮估計、階層先驗、校準都只改變對 θ_p 的估計（35% 抽樣成分），對 65% 無作用。這是 Kendall & Gal 的 aleatoric 定義在實體層級的直接套用。
- 現行 class-balanced CE 已等價於均勻先驗下的後驗；再對集成輸出做 EB 收縮等於二次套先驗，且 Saerens EM / temperature scaling 死路已顯示後驗校準方向無收益。
- 要降低 65% 成分只有兩條路，皆非「觀測棋步」函數：(a) 取得與 ε_p 相關的側資訊（段位是由對局結果與對手段位決定，而非棋步品質；若資料含對手段位/勝負/讓子，結果型特徵是唯一可碰到 ε_p 的通道 — 記憶中 HD 讓子特徵 +0.0071 即屬此類）；(b) 由主辦方端改善標籤。皆不在本調查範圍。
- Finding Elo 的教訓可直接遷移：單局 Elo MAE ≈160 分（約一個段位級距的量級），且「加入選取方式不同的資料會變差」— 與 mixed-player training 死路同向。

### Gaps
- 未找到 Go 段位（或任何伺服器段位制）標籤相對棋力的噪聲量化研究；只有西洋棋 Glicko/Elo 的間接證據。
- 未找到「多局聚合後評分預測誤差隨局數下降曲線」的公開結果（RatingNet 未做、Finding Elo 為單局）。
- Finding Elo 第一名寫法無法取得（Kaggle 頁面不可讀）。

---

## Q6. 小幅集成實務：snapshot ensembles、SWA、checkpoint averaging、跨 fold 平均 vs 全資料重訓

### Takeaway
SWA/snapshot 在影像基準上給單一模型 +0.2–1.9 pp，但 2025 ICML 大規模研究顯示當學習率已完整退火到 0 時 checkpoint averaging 的終點收益「largely disappears」且整體僅 mild；在已有 12 模型集成且 65% 誤差為共享偏移的系統上，這些手段只能再壓縮 35% 中的模型變異部分，期望增益 ≤0.1–0.2%。跨 fold 平均 vs 全資料重訓無一致勝者，須以 CV 實測。

### Cited Findings
- SWA（Izmailov et al. 2018）Table 1/2：CIFAR-100 PreResNet-164 SGD 78.49 → SWA 1 budget 79.77 (+1.28) / 1.5 budgets 80.35 (+1.86)；WRN-28-10 80.82 → 81.46 / 82.15；CIFAR-10 PreResNet-164 95.28 → 95.56 / 95.83 (+0.55)；WRN 96.18 → 96.79 (+0.61)；Shake-Shake 96.93 → 97.16 (+0.23)；ImageNet ResNet-50 76.15 → 76.97 (+0.82)、ResNet-152 78.31 → 78.94 (+0.63)、DenseNet-161 77.65 → 78.44 (+0.79)；「SWA is able to achieve comparable or better performance than FGE ensembles with just one model」；幾乎無額外計算。 — [arXiv 1803.05407](https://ar5iv.labs.arxiv.org/html/1803.05407)
- Snapshot Ensembles（Huang et al. 2017）Table 1 錯誤率：ResNet-110 CIFAR-10 5.52 → 5.32、CIFAR-100 28.02 → 24.19、SVHN 1.96 → 1.66；DenseNet-100 CIFAR-10 3.74 → 3.44、CIFAR-100 19.25 → 17.41；WRN-32 CIFAR-10 5.43 → 4.41；作者稱「comparable with ensembling of 2 independent models」，獨立訓練的 true ensemble 為其下界；較大重啟學習率 (α₀=0.2) 因多樣性較高多數情況更好。 — [arXiv 1704.00109](https://ar5iv.labs.arxiv.org/html/1704.00109)
- "When, Where and Why to Average Weights?"（ICML 2025, AlgoPerf 7 workloads）：LAWA/EMA 平均達標步數為 NadamW 的 ≈82%，GPU 時數省約 10–12%；終點泛化增益「mild」：ViT 0.79048 → 0.79112 (LAWA) / 0.79232 (EMA)，WMT BLEU 31.157 → 31.429 / 31.457，OGBG MAP 0.3012 → 0.3129，Criteo 與 Conformer 無差；**學習率完整退火到 0 時，平均結果收斂到退火模型，終點收益大致消失**；以平均取代 LR decay 通常較差，兩者合用略優；LAWA 視窗約總預算 1% 穩定有效。 — [arXiv 2502.06761](https://arxiv.org/html/2502.06761v2)
- NVIDIA Grandmasters Playbook：100 seed XGBoost 平均 MAP@3 ≈0.379 vs 單 seed ≈0.376（+0.003）；全資料重訓「another leaderboard bump」無量級；hill-climbing 集成贏 Predict Calorie Expenditure、三層 stacking 贏 Podcast Listening Time（無數字）。 — [NVIDIA Developer Blog](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- fast.ai Plant Seedlings 討論：k-fold 機率平均 LB 0.98992 vs 全資料單模 0.98488，作者不確定原因；其他串建議全資料重訓但警告收斂行為會變、無法再判斷過/欠擬合。 — [forums.fast.ai Plant Seedlings](https://forums.fast.ai/t/kaggle-comp-plant-seedlings-classification/8212/180?u=radek)；[forums.fast.ai 100% training data](https://forums.fast.ai/t/using-100-of-the-training-data-for-training/103070)
- FGE 論文：改善優於 Snapshot Ensembles（CIFAR-10/100、ImageNet）。 — [arXiv 1802.10026](https://arxiv.org/pdf/1802.10026v4.pdf)

### Inferences
- 量級換算：SWA/snapshot 的 +0.2–1.9 pp 是對**單一**未集成模型；集成內每個成員做 SWA 的邊際收益會被 12 模型平均大幅稀釋（Snapshot ≈「2 個獨立模型」的效果，而我們已有 12）。對 CV 0.726，合理期望 +0.000–0.002。
- 若各成員訓練已採 cosine/linear 退火到接近 0，依 ICML 2025 結果 EMA/LAWA 幾乎無終點收益；若採 step decay 或 early stopping，SWA/EMA 值得一試（成本 ≈0）。
- 跨 fold 平均 vs 全資料重訓：兩者差異取決於學習曲線斜率（5-fold 用 80% 資料，全資料多 25%）與失去 OOF 驗證的風險；文獻無一致方向。可同時保留：12 個 fold 模型 + 1–2 個全資料模型（以 fold 模型決定的 epoch 數）進集成，是 Kaggle 常見妥協。期望增益 0–0.002。
- 因 65% 誤差變異為跨模型共享偏移，任何集成/平均手段的理論上限是壓縮剩餘 35% 中「模型變異」那一小部分（12 模型後已約為單模的 1/12）。這是 stacking 死路的根本原因，也是 SWA/snapshot 收益必然微小的原因。

### Gaps
- 未找到在表格/序列小模型（而非 CNN/Transformer 影像）上 SWA/snapshot 的量化報告。
- 未找到「fold 平均 vs 全資料重訓」的系統性研究；僅論壇單例。

---

## 綜合評估（相對 CV 0.726 的期望增益、成本、與死路重疊）

### Takeaway
依文獻與理論，沒有任何一條線有「有證據的 ≥+0.5% 增益」；唯二值得低成本實測的是 (1) soft 偽標籤 + 原標籤 fine-tune（0 至 +0.3%，未在死路清單），(2) 成員層 EMA/SWA 或補 1–2 個全資料模型（0 至 +0.2%，成本 ≈0）。其餘（噪聲建模、attention/noisy-OR pooling、等量指派、TTT/entropy、階層收縮）理論上對雜訊評分設定無益或與死路重疊。

### Cited Findings
- 各項依據見 Q1–Q6；此節僅彙整。

### Inferences
| 技術 | 機制 | 文獻證據 | 對本設定（雜訊標籤與測試共享） | 期望增益 | 成本 | 死路重疊 |
|---|---|---|---|---|---|---|
| Forward/backward loss correction、ordinal noise model | 估乾淨後驗 p | Patrini 2017、Garg 2020（乾淨/argmax-preserving 合成雜訊） | 評分對象是 q 非 p；NI-ERM 近 minimax 最優 | 0 ± 0.001 | 低 | K、label smoothing |
| Co-teaching / label refurbishment | 丟棄或改標高損失樣本 | 僅乾淨測試證據 | 丟掉的正是測試中同樣存在的偏移玩家 | ≤0 | 中 | — |
| Attention / Set Transformer / noisy-OR pooling | 學習實例權重或互動 | 小袋 (5–7) 上僅 comparable；勝出於計數/異常偵測 | 段位 = 均值估計問題 | 0–0.002 | 低 | PMA |
| Distribution pooling（per-game 統計量多階矩） | 保留實例分佈 | Oner 2020 病理影像 | 遷移性未知 | 0–0.002 | 低 | — |
| Set-size conditioning / 固定大小訓練 | — | 無對照文獻 | 5–20 隨機子集已涵蓋 | 0 | 低 | per-n standardisation |
| 閾值最適化 | 修正回歸→序數錯配 | +0.05 QWK（回歸模型）、≈0（metric-aware 模型） | 已用 argmax E[score] | 0 | — | ES |
| 等量類別 OT/硬指派 | 修正 label shift | OTTER：邊際已匹配則不變 | 無 label shift、後驗近均勻 | ≤0 | 低 | balanced assignment、Saerens EM |
| 軟邊際約束（Sinkhorn/λ 懲罰、相鄰加權成本） | 同上之溫和版 | 無文獻 | 同上 | 0 ± 0.001 | 低 | 同上 |
| 測試集 soft 偽標籤 + 原標籤 fine-tune | 自蒸餾 / 資料擴增 | Kaggle 實務；無同分佈量化 | 無偏移故機制弱；不會修 65% | 0–+0.003 | 中（重訓 12 模型） | 無 |
| TTT / entropy minimisation | 適應分佈偏移 | 無偏移時可劣化 | 削弱相鄰保護 | <0 | 中 | — |
| 階層貝葉斯 / EB 收縮、校準 | 降低抽樣變異 | EB 理論 | 偏移 ⊥ 觀測 ⇒ 不可約；先驗重複 | 0 | 低 | temperature scaling、Saerens |
| 成員層 SWA / EMA / snapshot | 平坦極小、單模內集成 | 單模 +0.2–1.9 pp；LR 退火到 0 則消失 | 被 12 模集成稀釋 | 0–0.002 | ≈0 | — |
| 全資料重訓補入集成 | 多 20–25% 資料 | 論壇單例互有勝負 | 失去 OOF；學習曲線決定 | 0–0.002 | 低 | — |

### Gaps
- 以上期望增益為依文獻量級與本問題誤差分解的推估，非直接量測；真正的上限由 35% 抽樣成分中的模型變異決定，建議以「12 模集成 vs 24 模集成」的 CV 差做為所有集成類手段的經驗上限（若該差 <0.001，可一併關閉 SWA/全資料重訓線）。

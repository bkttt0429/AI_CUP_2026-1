# 最終流程用到的權重（官方若要求時提供的清單）

2026-11-21。這份清單只列「產生最終提交檔的那條流程」用到、而且檔案還在的權重，共 5.56 GB。其餘實驗的權重、輸出與快取都留在本地備份，不在這份清單。

每一行是：sha256、大小（位元組）、相對路徑。

## 官方要交的東西（競賽頁面的原文）

- 「得獎隊伍須於指定截止日前，提交可重現該模型的原始程式碼與最終報告，以驗證結果。」
- 報告內容：「(1)成員、(2)模型演算法說明、(3)資料前處理原始程式碼說明、(4)模型訓練原始程式碼與說明、(5)各項參數之設定(包括訓練權重)說明、(6)程式執行環境說明，(7)是否使用額外資源並說明來源。」
- 「優勝隊伍提交預測模型說明文件、程式碼，以及自行製作或使用之外部資料集（如有）」
- 繳交時間：2026/11/26 11:00 – 2026/12/06 23:59:59。頁面沒有寫權重要不要另外上傳、用什麼形式、大小上限；上傳的地方 11/26 才開放。

也就是：明文要求的是報告、可重現的原始碼、外部資料集；權重是報告裡要說明的項目。下面的清單是為了「驗證結果」時能立刻拿出來。

## 沒有權重、只剩輸出的三個部分（報告要如實說明）

- M82 的學習式配對器（識別器 5e 的一部分）：權重已刪，輸出在 `BK_AICUP_2026-1/results/m82/`。
- M84 的配對器 P（與 M82 同型、用全部訓練列訓練的學習式配對器，三個成員；識別器 SP、SP2e 的一部分）：權重已刪，輸出在 `AICUP1_MOVE_20261121_work/work/aicup1_moved/m84_cache/var_P.npz`。
- M120 的對稱擴增網路（「像誰」讀數 S4e）：權重已刪，原型與輸出在 `results/m120/out_*.npz` 與 `AICUP1_MOVE_20261121_work/work/aicup1_moved/m84_cache/var_S4e.npz`。

這三項的輸出都在本地備份裡，最後幾層（證據、合併、決策）可以由輸出重算。

## 外部資料集

M101 的認人網路訓練時加入了 Fox Go Dataset（https://github.com/featurecat/go-dataset，GPL-3.0）的公開帳號。選用的棋局清單在 `AICUP1_MOVE_20261121_work/work/ext_fox_pilot/select.csv`（193,700 局；帳號只以加鹽雜湊表示）。由這些棋局算出的逐手特徵（4 GB）在 nano4 的 `/work/u2507115/ext_fox/pilot/ply`，沒有放進備份（可由公開資料重算）。

## 基礎集成：24 個成員（C、T 兩種結構 × 4 折 × 3 個種子）（24 個檔，11 MB；已在 GitHub）

位置：`AICUP1_MOVE_20261121/` 之下

```
5ff85dc1ff0634e416e8170376e65795e01e5a15d849bc19d3960ab6add6f94f      476557  BK_AICUP_2026-1/results/models/ct_cv_m33/val1_C-pc-f0-x40-oh_s0.pt
97d919fdf1a73f0334824e967c5edd01f04b6bd2df775181f0a0948397daadf7      476557  BK_AICUP_2026-1/results/models/ct_cv_m33/val1_C-pc-f0-x40-oh_s1.pt
f0c41c0f4e156d3123908c13f3c96d845226c70cf0e0c104d513525fa6a6334b      476557  BK_AICUP_2026-1/results/models/ct_cv_m33/val1_C-pc-f0-x40-oh_s2.pt
960535b95a4340588f19b20a18bafaf1d931897181dd98fc5ef6a2daf0af1622      400397  BK_AICUP_2026-1/results/models/ct_cv_m33/val1_T-pc-f0-x40-oh_s0.pt
28ca77d2ad8dc8a7161c68067d97fceca6d5ad6a6d1a9366d7f0c7fa86902ae4      400397  BK_AICUP_2026-1/results/models/ct_cv_m33/val1_T-pc-f0-x40-oh_s1.pt
464f199c33160fe505130b8807d79b0af213407f62f345fe2ca96917f47c053e      400397  BK_AICUP_2026-1/results/models/ct_cv_m33/val1_T-pc-f0-x40-oh_s2.pt
9ebff46d9be9520668c1690e4af7609a5db0220ed5824077775cc4c30ffebbe4      476557  BK_AICUP_2026-1/results/models/ct_cv_m33/val2_C-pc-f0-x40-oh_s0.pt
d565e5e4c3239afb72c9b1fe0656cc1c2f208c3a23d0ae5a29484611f1019191      476557  BK_AICUP_2026-1/results/models/ct_cv_m33/val2_C-pc-f0-x40-oh_s1.pt
d6ea5ab4497be99ddf262d8f2e86c68eb852b6d6b1fbf068a367097e30953432      476557  BK_AICUP_2026-1/results/models/ct_cv_m33/val2_C-pc-f0-x40-oh_s2.pt
61300ce962cc98a706fb99c0a285bac44277175cbc41c4b075fdd1f3d1ef3ff4      400397  BK_AICUP_2026-1/results/models/ct_cv_m33/val2_T-pc-f0-x40-oh_s0.pt
18fb585d890afb62ef48454d345ca94a5a81cc380ddea2b0fa029d76af1b04fc      400397  BK_AICUP_2026-1/results/models/ct_cv_m33/val2_T-pc-f0-x40-oh_s1.pt
3d3f86344ede47d6688f3cdbcea3a8f6373bb0dd662604a7fe95e685718262fb      400397  BK_AICUP_2026-1/results/models/ct_cv_m33/val2_T-pc-f0-x40-oh_s2.pt
08761d65491818dfb39e227728824391bdf1099f11ca61adb2feb10fe20e6b5d      476557  BK_AICUP_2026-1/results/models/ct_cv_m33/val3_C-pc-f0-x40-oh_s0.pt
f514f5e64bf2438b5a7ba0b1301999db30b2701d50ec6fcad7b1e515e4793996      476557  BK_AICUP_2026-1/results/models/ct_cv_m33/val3_C-pc-f0-x40-oh_s1.pt
d26a18c9e08ec5b8b71528268dc00231f59683496a59584ba17b03e310e0aeed      476557  BK_AICUP_2026-1/results/models/ct_cv_m33/val3_C-pc-f0-x40-oh_s2.pt
b2fa93f7c150cfe2799ed5fab5ef63a17383ab1caf91487e0e558701a0c5509e      400397  BK_AICUP_2026-1/results/models/ct_cv_m33/val3_T-pc-f0-x40-oh_s0.pt
9ca9b63eed6decf338bc1e18cdf958baf52504aa1ac546bd3cd889b1cae8f2ae      400397  BK_AICUP_2026-1/results/models/ct_cv_m33/val3_T-pc-f0-x40-oh_s1.pt
67297071414567ccef97fed1db2592bf7638d32fb73a69b4deefd4e0fd228d31      400397  BK_AICUP_2026-1/results/models/ct_cv_m33/val3_T-pc-f0-x40-oh_s2.pt
098c10d8a1f17960292754b5f0560c4cf1c257b7e65e608afba9c5ba1c468811      476557  BK_AICUP_2026-1/results/models/ct_cv_m33/val4_C-pc-f0-x40-oh_s0.pt
22390530fa306515481efeb550bcb8a243ce75466a17fc4294b96b42da3e6783      476557  BK_AICUP_2026-1/results/models/ct_cv_m33/val4_C-pc-f0-x40-oh_s1.pt
921e1f6304536ef9edc378d218993d592a443f97d5ea26a800a6d0d0b725b907      476557  BK_AICUP_2026-1/results/models/ct_cv_m33/val4_C-pc-f0-x40-oh_s2.pt
c94b1aa5d4d62ca678574715fa0e5ce39d037b9743598215667a893ad35f4935      400397  BK_AICUP_2026-1/results/models/ct_cv_m33/val4_T-pc-f0-x40-oh_s0.pt
08203bae9af5381eb03e1e84629797d5a9827d8fe9cf826a1217f061ad355cc6      400397  BK_AICUP_2026-1/results/models/ct_cv_m33/val4_T-pc-f0-x40-oh_s1.pt
5a2aa82a65e74215750b1ba90ec90939a18371b66b6bfb02016ecdeb540581b2      400397  BK_AICUP_2026-1/results/models/ct_cv_m33/val4_T-pc-f0-x40-oh_s2.pt
```

## 基礎集成：L1b 等級頭與微調過的 trunk（4 個檔，219 MB；已在 GitHub（trunk 切成三段））

位置：`AICUP1_MOVE_20261121/` 之下

```
6bb4ebf40747ebdcaeb7e9e2def6b5cbd5e22fd7b13399dde8e55de791bbe2c3     1828289  BK_AICUP_2026-1/results/l1b/heads3_A.pt
5c6acadb53043a383056374f69c3731643c1dc97291afbb021fc8f3606719f64   107920499  BK_AICUP_2026-1/results/l1b/heads3_A.trunk.pt
37dc608239a2457ac51a41c4102a3ef37906a192f213aa2603f95f025a72dd5b     1828289  BK_AICUP_2026-1/results/l1b/heads3_B.pt
a5b047d6cbf89e34edd07597854fe1c9bca64a03894bbd4237f31046134e0c88   107920499  BK_AICUP_2026-1/results/l1b/heads3_B.trunk.pt
```

## 基礎集成：題目層模型（1 個檔，0 MB；已在 GitHub）

位置：`AICUP1_MOVE_20261121/` 之下

```
27e72e338596046bfdd13971ede6f6c7c18c6a1d879c092f2751aa2bd4efda4a      270835  BK_AICUP_2026-1/results/models/rank_model_fused_opp_diff_hd_r.json
```

## 錨點：編碼器（M77）（5 個檔，147 MB；只在本地備份）

位置：`AICUP1_MOVE_20261121/` 之下

```
dd4e5ec91d3b47f4c26738209a6d9043ab394dc3019de3f5b20d21f7456ed1c2    30692799  BK_AICUP_2026-1/results/m77/model_-1.pt
a58360c61520324520cbb5fb288db5c399c2de5ef1a2b069ab3a9c3ecc4cfac8    29033764  BK_AICUP_2026-1/results/m77/model_1.pt
2ee078c3becd303654ad052b4fee44a16b981c83d9ce391763dd7ddad7483876    29039972  BK_AICUP_2026-1/results/m77/model_2.pt
6def0c8b3034846498ac2834e3bdc9110c76cef8620966ca3144957327b28484    29045092  BK_AICUP_2026-1/results/m77/model_3.pt
4302f5bd36c9871f92302b83bb461b10bfe8b583038d0a00dba823709c818132    29046180  BK_AICUP_2026-1/results/m77/model_4.pt
```

## 錨點：序列認人網路 M88（識別器 SP）（5 個檔，624 MB；只在本地備份）

位置：`AICUP1_MOVE_20261121/` 之下

```
ec8d0764d6e24a3665e040fff9fdc8dbb41f06061d5aa4e340d5118f85906a96   124777847  BK_AICUP_2026-1/results/m88/ckpt_-1.pt
e060859e6d56ad351caa84d725937c01961e0d2cf29fa7fb66ec85ea99c997de   124777847  BK_AICUP_2026-1/results/m88/ckpt_1.pt
1fc2d20120f142d90a8209f6febeb7b5a8df66cc6718dac9fdc698781900e8da   124777847  BK_AICUP_2026-1/results/m88/ckpt_2.pt
db4b90d2f7333dd3b163e7e55749bb9ef30a2a611843f7dfe45d2caed493868a   124777847  BK_AICUP_2026-1/results/m88/ckpt_3.pt
9310e60e191c959b1a343121a90f9e1da92220421ccf75ef43ba139954608883   124777847  BK_AICUP_2026-1/results/m88/ckpt_4.pt
```

## 錨點：序列認人網路 M90，三個種子（識別器 S2 用種子 0，SP2e 用三個種子）（15 個檔，1872 MB；只在本地備份）

位置：`AICUP1_MOVE_20261121/` 之下

```
1e1c71db1f78056336beb827d0b38b6798148d5939ec2d40b4c0b86ea96d6890   124777847  BK_AICUP_2026-1/results/m90/ckpt_-1.pt
3001c4765f9a5e9c6ba15387685a21974bd017c4c3af5644ef1b86867b4deed3   124777847  BK_AICUP_2026-1/results/m90/ckpt_-1_s1.pt
6ae5f1af96b2570a7123888b9a11e94ef7fb362a6382ac2ef594e8afec357a63   124777847  BK_AICUP_2026-1/results/m90/ckpt_-1_s2.pt
fb750686eedf60066dab93ef0946df4336a81e4c49ea98da5e450cf2c9470ba5   124777847  BK_AICUP_2026-1/results/m90/ckpt_1.pt
8ece7d1addb2d55aa5ae91376b965190215a7e50ea6459ecb97167a0c76b1e66   124777847  BK_AICUP_2026-1/results/m90/ckpt_1_s1.pt
98f01619f1f8187cdff6c8f157e98e4eb48d5671545e16efbbc40318ababbeba   124777847  BK_AICUP_2026-1/results/m90/ckpt_1_s2.pt
b9d89e9834c4ac93fac0bc4b866ca68328fd2083c650845e82fabac6e749107b   124777847  BK_AICUP_2026-1/results/m90/ckpt_2.pt
3f45dd1cffcd6e05c8bce21022addf73f00d38d39ed64e1e381754c742bd4514   124777847  BK_AICUP_2026-1/results/m90/ckpt_2_s1.pt
e0b2bb593c45bdd84f90d7991d0920c7272b8b76644ea4ec718612e8b1034339   124777847  BK_AICUP_2026-1/results/m90/ckpt_2_s2.pt
57aeb52ff48c7d85a8ddcbe2928e1574967424222a7feb5e862587ddd4db6411   124777847  BK_AICUP_2026-1/results/m90/ckpt_3.pt
b21ee037b2ca5563f0f158aa98468c15785001d0f48bd74151cb4c62362b2640   124777847  BK_AICUP_2026-1/results/m90/ckpt_3_s1.pt
73b0c43503ac684491ff90685ccd2090f68dad6a3832f52933948641f390a434   124777847  BK_AICUP_2026-1/results/m90/ckpt_3_s2.pt
eea7664fdc2554a50a7bbd1fe3e4b2b19e968c56f3d97e6f9f32b5b76bee0992   124777847  BK_AICUP_2026-1/results/m90/ckpt_4.pt
69c264c221597deed17dfd6a6f13f8e6678ab3611bf799ce96059d0cd6c3c48b   124777847  BK_AICUP_2026-1/results/m90/ckpt_4_s1.pt
5e3c14aad32ebb6ea8fc878df1bb89a2b33a0e0f9a9fe33b23441cdc5a0f2513   124777847  BK_AICUP_2026-1/results/m90/ckpt_4_s2.pt
```

## 錨點：M101c（識別器 S3c）（5 個檔，624 MB；只在本地備份）

位置：`AICUP1_MOVE_20261121/` 之下

```
516f4ccc74d9a6e7577e549358c72821e9fa5771ea5c4841a220d4d765a521d3   124777847  BK_AICUP_2026-1/results/m101c/ckpt_-1_s3.pt
f39da0d9f0be55fcc0b8f238c9a475fd4bea045065ed64bd7cb11304d254ce59   124777847  BK_AICUP_2026-1/results/m101c/ckpt_1_s3.pt
405ed06f9e44a78a7dd77e39f29cd132394154542907c45accc88675d1a0edef   124777847  BK_AICUP_2026-1/results/m101c/ckpt_2_s3.pt
9929930cc78c2e71b8c4f04d8e3b93a28b22b221358142c819b4dddcabcbd4bc   124777847  BK_AICUP_2026-1/results/m101c/ckpt_3_s3.pt
fc4dcf596114ef3528e384f07b8b1c41308620a9b4572a0833e1f90ad7892d3c   124777847  BK_AICUP_2026-1/results/m101c/ckpt_4_s3.pt
```

## 錨點：M101（識別器 S3；訓練時加入外部公開帳號）（5 個檔，624 MB；只在本地備份）

位置：`AICUP1_MOVE_20261121/` 之下

```
7df6dff633aa6d1e6d0a70494c8f75b3e3bdb92408eaca2f8e7199f83fe8bc00   124777847  BK_AICUP_2026-1/results/m101/ckpt_-1_s3.pt
4b27489185f30eff6a92e91ddc1dbc9470cdca40e8a02f678fd659e872bf8e38   124777847  BK_AICUP_2026-1/results/m101/ckpt_1_s3.pt
684cb65affc93618237a105796e23f76155d30d6e699cf8f88a808211fc4a0b0   124777847  BK_AICUP_2026-1/results/m101/ckpt_2_s3.pt
9ecc9410d4d01a41862d65d121c9f72a3df0797c55c442c5b087aa36e68d5e27   124777847  BK_AICUP_2026-1/results/m101/ckpt_3_s3.pt
1fedc63f64375e90575d5522a83f3e9604cf525dd64bfefcca6fa23ece5d69eb   124777847  BK_AICUP_2026-1/results/m101/ckpt_4_s3.pt
```

## 錨點：M161p（S6at）（5 個檔，312 MB；只在本地備份）

位置：`AICUP1_MOVE_20261121_work/` 之下

```
ea678472fc77054fbb889ba2df26c65147413aa44cd7e49607bc127f569e160e    62389603  work/m161p_ckpt/ckpt_-1_s20.pt
23504c4ba50d896b972be9fc1d9a98513ffe732e843a16503c71de1b2af14d98    62390251  work/m161p_ckpt/ckpt_1_s20.pt
1f4e584a8a5696c99c940b40760b61f688456cb1d5491af337fb05ab05856ad5    62390251  work/m161p_ckpt/ckpt_2_s20.pt
6fc3f711adef59671f5b333ff2cbaa89d5cd3b61c029337c3fe7a54b878f96cd    62390251  work/m161p_ckpt/ckpt_3_s20.pt
da0c4b67fb708d7158612ffb9f3f168a7d7ac2e7ee2fa01eabeeeaf98f5a0730    62390251  work/m161p_ckpt/ckpt_4_s20.pt
```

## 錨點：M161q（S7t、S7tc）（10 個檔，1124 MB；只在本地備份）

位置：`AICUP1_MOVE_20261121_work/` 之下

```
51a262e28dc5dbad165a1d30a308977f6bf3044765889fd5a326b65fb108ae00   112440104  work/m161q_ckpt/ckpt_-1_s30.pt
c61dc4e81836b49ee0300c7954ed0eec02431a1b1e4fba7e4f80efe9a8581a5c   112440104  work/m161q_ckpt/ckpt_-1_s31.pt
6e3a0d4f5f9454a18c6e2be6404b77314b57cd886ce87883bf17e8e94012888b   112440751  work/m161q_ckpt/ckpt_1_s30.pt
95a96e48412d774ac1b3bebff1c5608e4c54f2d21bb8fad7547dd7ae2826150c   112440751  work/m161q_ckpt/ckpt_1_s31.pt
7550873aa171a81e3fcb0aff4f21222f12303180464fc29dde9e8e063af44e60   112440751  work/m161q_ckpt/ckpt_2_s30.pt
0d81585686da4308a3236d1bbcfd26faf25f4f7b2ee038fb2d281450969183a4   112440751  work/m161q_ckpt/ckpt_2_s31.pt
358bc1a20943edde04b2ef50b1243170ad97e862f4dddf3349d01a6835d9b98a   112440751  work/m161q_ckpt/ckpt_3_s30.pt
b61d44a30e62bd37dd1966dff9dc6811104dd73aad954d2ffe3571a0996d78db   112440751  work/m161q_ckpt/ckpt_3_s31.pt
021d4fbdf3a05b4d6e7dca78d4dd51efc7069ad955747dc7110859611bba0c03   112440751  work/m161q_ckpt/ckpt_4_s30.pt
8a147473a8b952544c636a575b702c684469c3c60c4e520f7dbdd7ab087f6be8   112440751  work/m161q_ckpt/ckpt_4_s31.pt
```


# 研究報告產生器（claude.ai artifact 55JVvt6NaTpeqDkd1kpqNW）

`python3 -I reports/go_report/build.py` 從 `results/*.json` 讀數字（開頭有 assert 守門），輸出 `reports/go_report/site/`（index、ch01–ch09、talk、assets；site 與 shots 不進 git）。
章節內容只在 `build.py` 的 `CHAPTERS`。`assets/` 是研究筆記共用的樣式與腳本，build 時複製進 site。
`results/seq/{hd_check,fingerprint_check_v2,feature_stability}.json` 與 `nano4/issue42-hd-check`、`nano4/issue36-fingerprint-results` 的同名檔位元相同，所以這個分支可單獨建置，那兩條合併時也不會衝突。
只存在其他 PR 分支的四個數（TSFEL、純雜訊、L1、HD 穩定性）以常數寫在 `build.py`，來源註明在旁。
`shot.py` 用 Playwright 量每頁重疊。nano4 上兩支都用 sbatch 跑。發布：Artifact publish，url 同上、root = site、files 對應全部檔案。

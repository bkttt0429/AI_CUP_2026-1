# 研究報告產生器（claude.ai artifact 55JVvt6NaTpeqDkd1kpqNW）

`python3 -I reports/go_report/build.py` 從 `results/*.json` 讀數字（開頭有 assert 守門），輸出 `reports/go_report/site/`（index、ch01–ch09、talk、assets）。
章節內容在 `chapters_v2.py`（同一份文字也拼在 `build.py` 的 `CHAPTERS`；改章節時改 `chapters_v2.py` 再把 `CHAPTERS = [...]` 區塊貼回 `build.py`）。
只存在 PR 分支的四個數（TSFEL、純雜訊、L1、HD 穩定性）以常數寫在 `build.py`，來源註明在旁。
`shot.py` 用 Playwright 量每頁重疊（sbatch 執行）。發布：Artifact publish，url 同上、root = site、files 對應全部檔案。

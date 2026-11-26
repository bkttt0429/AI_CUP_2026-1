# 分支整併紀錄（2026-11-27）

依使用者指示，倉庫整併為單一分支 `main`。`main` 的內容即最終提交所用的程式與紀錄（整併前為 `588cac3`）。

- 其他分支的提交全部保留在 `main` 的歷史裡：尚未包含在 `main` 的九個分支以「只併歷史」的方式併入（`git merge -s ours`），合併前後 `main` 的檔案內容完全相同，因此未採用的實驗程式與較早的版本不會進入最終流程。
- 這些分支上獨有的檔案仍可由下表的提交取回，例如：`git checkout <提交> -- <路徑>`，或 `git branch <名稱> <提交>` 重建分支。

| 分支 | 最後一筆提交 | 日期 | 狀態 |
| --- | --- | --- | --- |
| `claude/festive-ptolemy-org8pz` | `12e96f87b572` | 2026-11-02 | 有 14 筆提交不在 main，以「只併歷史」的方式併入 |
| `final-submission` | `5aa9a9c36f58` | 2026-11-10 | 有 1 筆提交不在 main，以「只併歷史」的方式併入 |
| `nano4/final-chain-20261121` | `755343ca3042` | 2026-11-21 | 原本就包含在 main 裡 |
| `nano4/issue23-error-analysis` | `0f53b8d88ab6` | 2026-11-01 | 原本就包含在 main 裡 |
| `nano4/issue25-fused-opp-diff` | `75a66c60de0b` | 2026-11-01 | 原本就包含在 main 裡 |
| `nano4/issue29-hist-check` | `b4e8df3ad2d9` | 2026-11-01 | 原本就包含在 main 裡 |
| `nano4/issue36-fingerprint-results` | `4cdd0bf3f28e` | 2026-11-02 | 有 1 筆提交不在 main，以「只併歷史」的方式併入 |
| `nano4/issue36-tsfel` | `1447777bffc3` | 2026-11-01 | 有 1 筆提交不在 main，以「只併歷史」的方式併入 |
| `nano4/issue41-qmodel` | `426d71905fbd` | 2026-11-01 | 有 3 筆提交不在 main，以「只併歷史」的方式併入 |
| `nano4/issue42-hd-check` | `f52166e9fab8` | 2026-11-02 | 有 5 筆提交不在 main，以「只併歷史」的方式併入 |
| `nano4/issue42-hd-results` | `2813c404829d` | 2026-11-03 | 有 14 筆提交不在 main，以「只併歷史」的方式併入 |
| `nano4/l1b-opponent` | `d30694dd4035` | 2026-10-27 | 原本就包含在 main 裡 |
| `nano4/l1b-stage2` | `b2351c17d4ca` | 2026-10-26 | 原本就包含在 main 裡 |
| `nano4/report-v2` | `31a2d550e70e` | 2026-11-02 | 有 3 筆提交不在 main，以「只併歷史」的方式併入 |
| `nano4/seq-mamba3` | `049347787319` | 2026-11-01 | 有 2 筆提交不在 main，以「只併歷史」的方式併入 |
| `nano4/seq-transformer-mamba` | `a06b1e93cc0a` | 2026-11-01 | 原本就包含在 main 裡 |

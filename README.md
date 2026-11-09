# AI_CUP_2026-1

最終提交流程（submission_rank_ct28.csv）與重現步驟見 [docs/FINAL_PIPELINE.md](docs/FINAL_PIPELINE.md)。

2026 AI CUP 圍棋競賽「棋力預測」：每題給同一位玩家的 5–20 局棋，預測 10 級段位（D、C、B、A、1D–6D）之一。

- 資料集與規則：[REPORT.md](REPORT.md)
- 方法與實驗結論：[METHOD_NOTES.md](METHOD_NOTES.md)
- 分層架構與時程：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- 國網 nano4 操作步驟：[docs/NANO4.md](docs/NANO4.md)

訓練資料、KataGo 權重與引擎、虛擬環境都不進 git（見 `.gitignore`），依 REPORT.md 與 docs/NANO4.md 重建。

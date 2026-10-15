# 棋力證據回放牆

每張卡片回放一局棋，下面的折線是**目標玩家已下的手**在 18 個 Human SL profile（弱 12k → 強 6d）下的累積後驗平均：線往上代表越來越像強的等級。藍帶是 pilot 中該級玩家最貼合 profile 的中間 80%，橫線是級位／段位分界。用來直接看「證據怎麼一手一手累積」、哪些玩家落在自己等級的帶外。

## 產生頁面（在 lab01 上，`~/AICUP_2026-1`）

```bash
.venv/bin/python -I scripts/build_replay.py --source pilot --fold 0 --per-rank 5
```

- `--per-rank K`：每級 K 位玩家各 1 局，共 10×K 盤（預設 5 → 50 盤）。
- `--player <id 前綴>`：某位玩家評過的全部局（最多 20 盤）。
- `--source pilot`：pilot 的 fold 0／1（每 fold 200 人）。`--source scores --fold F`：全 fold 評分的已完成分段。
- 輸出到 `results/replay/`（不進 git）。超過 100 盤會拒絕，要 `--force` 或改用多視窗。

## 開啟

在 lab01 桌面的終端機：

```bash
scripts/launch_replay.sh results/replay/replay_pilot_fold0_50boards.html
```

從 SSH 開到桌面上（顯示 `:1` 是 lab01 的桌面工作階段）：

```bash
DISPLAY=:1 scripts/launch_replay.sh results/replay/replay_pilot_fold0_50boards.html
```

多視窗模式（每個視窗一盤大棋盤，最多 40 個）：

```bash
DISPLAY=:1 scripts/launch_replay.sh --windows 20 results/replay/replay_pilot_fold0_50boards.html
```

啟動器使用已驗證的 NVIDIA 加速參數 `--enable-gpu --use-angle=vulkan` 與專用 profile（`results/replay/chrome-profile`），不改系統或使用者的 Chrome 設定。沒有 DISPLAY 時會改成無頭模式。

網址參數：`?board=3` 只顯示第 3 盤；`?autoplay=1&speed=10` 開啟即播放，每秒 10 手。

## 容量建議

| 模式 | 數量 | 更新延遲 P95 | 來源 |
| --- | --- | --- | --- |
| 同頁棋盤 | 50 | 28 ms | GPU 壓測（純回放頁） |
| 同頁棋盤 | 100 | 54 ms | GPU 壓測 |
| 同頁棋盤 | 200 | 105 ms | GPU 壓測，不建議 |
| 獨立視窗 | 20 | 24 ms | GPU 壓測 |
| 獨立視窗 | 40 | 52 ms | GPU 壓測 |
| **本頁（含證據折線）** | 50 | 34 ms | `verify_replay.cjs`，評分程式同時佔用兩張 GPU 時 |
| **本頁（含證據折線）** | 100 | 55 ms | 同上 |

建議同頁 50～100 盤，或 20～40 個獨立視窗。500 盤約 268 ms，不適合互動。數字是無頭壓測，不含遠端桌面（AnyDesk）的延遲。壓測紀錄在 `reports/lab01_go_gpu_capacity_20261015/`。

## 驗證

```bash
node scripts/verify_replay.cjs results/replay/replay_pilot_fold0_50boards.html
```

確認渲染器是 NVIDIA、GPU 合成與 2D canvas 加速已開啟、每盤都有證據、沒有主控台錯誤，量所有棋盤同時在畫面上的「全部下一手」延遲，並存一張截圖到 `results/replay/verify_*.png`。

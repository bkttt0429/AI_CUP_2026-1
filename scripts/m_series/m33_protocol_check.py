"""M33 protocol check (compute node; after training, before any comparison is read): every -pc member's sidecar against its ct7 (M22)
twin -- same training questions, held-out players, inner questions, steps per epoch, folds, seed, data and optimiser settings and score
directories; player_contrast 0.1; the same parameter count (the contrastive head is training-only); validation probabilities on disk.
  python -I scripts/m_series/m33_protocol_check.py
"""
import json  # 讀取成員的設定側檔
import sys  # 設定結束碼
from pathlib import Path  # 處理檔案路徑

KEYS = ("train_questions", "held_players", "inner_questions", "steps_per_epoch", "train_folds", "seed", "val_fold", "inner_frac", "extra_games",  # 必須與 ct7 雙胞胎一致的欄位：訓練題、留出玩家、內層題、每 epoch 步數、折、種子與額外局數
        "opp_hsl", "opp_hsl_r", "with_fold0", "epochs", "batch", "lr", "wd", "dropout", "loss", "pooling", "drop", "mix", "swa", "pseudo",  # 資料與最佳化設定
        "hsl_dir", "vec_dir", "l1b_set", "l1b_dirs", "opp_dirs", "opp_hsl_dirs", "extra_dirs", "n_r", "d_token", "n_scores")  # 評分資料夾與輸入尺寸

ROOT = Path.cwd()  # 以目前目錄為 repo 根目錄
bad = 0  # 是否有任何不符
for v in (1, 2, 3, 4):  # 四個驗證折
    for arm in ("C", "T"):  # C、T 兩種結構
        for s in (0, 1, 2):  # 三個種子
            fn, fc = ROOT / f"results/models/ct_cv_m33/val{v}_{arm}-pc-f0-x40-oh_s{s}.json", ROOT / f"results/models/ct_cv/val{v}_{arm}-f0-x40-oh_s{s}.json"  # M33 成員與其 ct7 雙胞胎的設定側檔
            npz = ROOT / f"results/trunk_seq/es/val{v}/{arm}-pc-f0-x40-oh_s{s}.npz"  # M33 成員的驗證機率檔
            if not fn.exists() or not npz.exists():  # 側檔或機率檔缺少
                print(f"MISSING {fn.name if not fn.exists() else npz.name}"); bad = 1; continue  # 記為失敗並跳過
            n, c = json.loads(fn.read_text()), json.loads(fc.read_text())  # 讀入兩邊的設定
            diff = {k: (n.get(k), c.get(k)) for k in KEYS if n.get(k) != c.get(k)}  # 不一致的欄位
            ok = not diff and n.get("player_contrast") == 0.1 and n.get("tag") == f"{arm}-pc-f0-x40-oh" and n["n_params"] == c["n_params"]  # 設定一致、對比損失 0.1、標籤正確且參數量相同
            print(f"pc val{v} {arm} s{s}: {'same' if not diff else diff}; params {c['n_params']} -> {n['n_params']}; best inner {n.get('best_inner'):.4f} vs ct7 {c.get('best_inner'):.4f}; "  # 印出比對結果與內層分數
                  f"stop epoch {n['best_step'] / n['steps_per_epoch']:.2f} vs {c['best_step'] / c['steps_per_epoch']:.2f}" + ("" if ok else "  <-- MISMATCH"))  # 接上停止時的 epoch，不符時加上標記
            bad |= not ok  # 累計失敗
print("PROTOCOL CHECK", "FAIL" if bad else "PASS")  # 總結
sys.exit(1 if bad else 0)  # 失敗時以 1 結束，讓 m33.sh 在比較之前停下

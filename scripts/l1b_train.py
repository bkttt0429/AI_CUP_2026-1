#!/usr/bin/env python
"""Train the L1b rank heads on a featcache (plan 2026-10-25-l1b-rank-heads, Task 4).

The trunk is humanv0 (--ckpt, frozen) or, for tests, a randomly initialised KataGo config (--model-config).
`--check-init` only measures max |Δ log p| between the freshly copied heads and the model's own policy
under the rank-unknown metadata on 256 cached positions (fp32) and exits 0 when it is <= 1e-4.
Otherwise it trains with aicup.l1btrain.train_heads and writes <out> (.pt) and its .json sidecar; an
existing output is refused (exit 2) unless --overwrite, which also discards the checkpoint; an existing
<out>.ckpt without <out> resumes the interrupted run.

  python -I scripts/l1b_train.py --cache results/l1b/cache_A --out results/l1b/heads_A.pt --check-init
  python -I scripts/l1b_train.py --cache results/l1b/cache_A --out results/l1b/heads_A.pt

Stage 2 (issue #6): --trunk-lr > 0 fine-tunes the trunk too (its state_dict is saved as <out stem>.trunk.pt and
named in the sidecar, so l1b_score.py loads it with the heads); --init-heads starts the heads from a saved run
(the stage-1 heads). Several GPUs: run under torchrun; every rank takes its share of each global batch of
--batch x world positions and rank 0 writes the outputs.

  torchrun --standalone --nproc_per_node 8 scripts/l1b_train.py --cache results/l1b/cache_A \
      --out results/l1b/heads2_A.pt --trunk-lr 1e-5 --init-heads results/l1b/heads_A.pt --batch 128

Cross-fit (issue #16): several --cache directories are read as one, and --exclude-folds K leaves out the games of
fold K's players (--folds-csv), so the sidecar's train_folds is the caches' folds minus K and l1b_score.py lets
the heads score fold K only:

  torchrun --standalone --nproc_per_node 8 scripts/l1b_train.py --cache results/l1b/cache_A results/l1b/cache_B \
      results/l1b/cache_0 --exclude-folds 3 --out results/l1b/heads5_f3.pt --batch 128
"""
import argparse  # 解析命令列參數
import os  # 讀取 torchrun 設定的環境變數
import sys  # 設定匯入路徑與結束碼
from pathlib import Path  # 處理路徑

import numpy as np  # 產生局面索引

ROOT = Path(__file__).resolve().parents[1]  # 專案根目錄（scripts/ 的上一層）
if str(ROOT / "src") not in sys.path:  # src/ 還不在匯入路徑時
    sys.path.insert(0, str(ROOT / "src"))  # 加入 src/ 以匯入 aicup 套件
from aicup.featcache import read_caches  # noqa: E402  讀取局面快取
from aicup.l1b import RankHeads, load_heads  # noqa: E402  等級頭的結構與載入
from aicup.l1btrain import HeadsCfg, check_init, train_heads  # noqa: E402  訓練設定、初始化檢查與訓練流程
from aicup.ladder import FOLDS_CSV  # noqa: E402  分折表
from aicup.setmodel_io import file_sha256  # noqa: E402  算檢查點的 sha256 供側檔記錄

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"  # Human SL 網路的檢查點（主幹）
INIT_TOL = 1e-4  # --check-init 允許的最大差異


def load_trunk(ckpt, model_config, device):  # 載入主幹模型
    """(model in eval mode on device, source string, checkpoint sha256 or None)."""
    import torch  # 只在需要時才載入 PyTorch
    if model_config:  # 測試用：隨機初始化的 KataGo 設定
        from katago.train import modelconfigs  # KataGo 的模型設定表
        from katago.train.model_pytorch import Model  # KataGo 的 PyTorch 模型
        torch.manual_seed(0)  # 固定初始化，結果可重現
        model = Model(modelconfigs.config_of_name[model_config], 19)  # 建立 19 路的模型
        model.initialize()  # 隨機初始化權重
        return model.to(device).eval(), f"config:{model_config}", None  # 沒讀檢查點，sha256 為 None
    from aicup.humansl import load_humansl  # 載入 Human SL 權重的工具
    model, _ = load_humansl(str(ckpt), device)  # 載入 humanv0
    return model, str(ckpt), file_sha256(ckpt)  # 一併回傳來源與 sha256 供側檔記錄


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以說明的第一行當描述
    ap.add_argument("--cache", required=True, nargs="+", help="one or more featcache directories, read as one cache")  # 局面快取資料夾（可多個）
    ap.add_argument("--exclude-folds", type=int, nargs="*", default=[],  # 交叉擬合時排除的折
                    help="leave out the games of these folds' players (cross-fit, issue #16); train_folds = the caches' folds minus these")  # 側檔的訓練折 = 快取的折減去這些折
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))  # 分折表
    ap.add_argument("--out", required=True, help="heads .pt (the .json sidecar goes next to it)")  # 等級頭的輸出檔
    ap.add_argument("--ckpt", default=str(CKPT))  # 主幹的檢查點
    ap.add_argument("--model-config", default=None, help="random KataGo config instead of --ckpt (tests)")  # 測試用的隨機模型設定
    ap.add_argument("--epochs", type=int, default=HeadsCfg.epochs)  # 最多訓練幾個 epoch
    ap.add_argument("--batch", type=int, default=HeadsCfg.batch)  # 每個 rank 的批次大小
    ap.add_argument("--lr", type=float, default=HeadsCfg.lr)  # 等級頭的學習率
    ap.add_argument("--eval-every", type=float, default=HeadsCfg.eval_every)  # 每隔多少 epoch 做一次內部評估
    ap.add_argument("--patience", type=int, default=HeadsCfg.patience)  # 連續幾次沒進步就停止
    ap.add_argument("--inner-frac", type=float, default=HeadsCfg.inner_frac)  # 每級留作內部驗證的玩家比例
    ap.add_argument("--inner-max-positions", type=int, default=HeadsCfg.inner_max_positions)  # 內部評估的局面上限
    ap.add_argument("--seed", type=int, default=HeadsCfg.seed)  # 隨機種子
    ap.add_argument("--device", default="cuda")  # 訓練用的裝置
    ap.add_argument("--trunk-lr", type=float, default=HeadsCfg.trunk_lr, help="> 0: fine-tune the trunk at this lr (stage 2)")  # 主幹的學習率（> 0 才微調）
    ap.add_argument("--init-heads", default=None, help="start from this saved run's .pt: its heads and, for a stage-2 run, its fine-tuned trunk")  # 以已存的等級頭為起點（第二階段的檔連同微調主幹）
    ap.add_argument("--disc-weight", type=float, default=HeadsCfg.disc_weight,  # 鑑別式損失的權重
                    help="> 0: add this x the rank cross-entropy of the heads' log p to the loss and to the selection objective")  # 等級交叉熵乘上此權重，加進損失與選模目標
    ap.add_argument("--check-init", action="store_true")  # 只檢查初始化是否等同原 policy
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output and its checkpoint")  # 覆寫既有輸出並丟棄其檢查點
    ap.add_argument("--extend", action="store_true", help="continue a finished run from its kept checkpoint (e.g. more --epochs)")  # 從保留的檢查點延長訓練
    a = ap.parse_args(argv)  # 解析參數
    world = int(os.environ.get("WORLD_SIZE", "1"))  # torchrun 的行程總數
    if world > 1:  # torchrun：每張 GPU 一個行程
        import torch  # 分散式訓練需要 PyTorch
        import torch.distributed as dist  # 分散式通訊
        local = int(os.environ["LOCAL_RANK"])  # 本節點內的 GPU 編號
        torch.cuda.set_device(local)  # 綁定到對應的 GPU
        from datetime import timedelta  # 設定逾時
        # 60 分鐘，而不是 NCCL 預設的 10 分鐘：多個 8 卡工作同時從共享檔案系統讀快取時，
        # 最後一個 rank 抵達第一次集體通訊的時間可能比第一個晚 10 分鐘以上（issue #16，f2 stage 1）
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))  # 建立 NCCL 行程群組
        a.device = f"cuda:{local}"  # 本行程使用的裝置
    me = int(os.environ.get("RANK", "0"))  # 本行程的全域 rank
    try:  # 結束後一定要釋放行程群組
        return _run(a, me)  # 執行檢查或訓練
    finally:  # 收尾
        if world > 1:  # 分散式執行時
            import torch.distributed as dist  # 分散式通訊
            dist.destroy_process_group()  # 釋放行程群組


def _run(a, me):  # 實際的檢查或訓練流程
    out = Path(a.out)  # 輸出的等級頭檔
    cfg = HeadsCfg(epochs=a.epochs, batch=a.batch, lr=a.lr, eval_every=a.eval_every, patience=a.patience, inner_frac=a.inner_frac,  # 由命令列參數組成訓練設定
                   inner_max_positions=a.inner_max_positions, seed=a.seed, device=a.device, trunk_lr=a.trunk_lr,  # 內部評估、種子、裝置與主幹學習率
                   disc_weight=a.disc_weight)  # 鑑別式損失的權重
    model, source, sha = load_trunk(a.ckpt, a.model_config, a.device)  # 載入主幹
    if a.check_init:  # 只做初始化檢查
        _, arrays = read_caches(a.cache)  # 讀入快取的局面
        heads = RankHeads(model.policy_head).to(a.device)  # 由模型的 policy head 複製出各等級頭
        worst = check_init(model, heads, arrays, np.arange(len(arrays["target"])), n=256)  # 前 256 個局面上的最大差異
        print(f"max |Δ log p| between the copied heads and the model's rank-unknown policy on 256 positions: {worst:.3e} "  # 印出最大差異
              f"({'ok' if worst <= INIT_TOL else 'FAIL'}, tolerance {INIT_TOL:g})")  # 以及是否在容許範圍內
        return 0 if worst <= INIT_TOL else 1  # 通過回傳 0，否則 1
    if out.exists() and not (a.overwrite or a.extend):  # 輸出已存在且未指定覆寫或延長
        if me == 0:  # 只由 rank 0 印訊息
            print(f"{out} exists; pass --overwrite to retrain (this also discards {out.name}.ckpt) or --extend to continue it", file=sys.stderr)  # 提示可用的選項
        return 2  # 拒絕覆寫
    ckpt_path = out.with_name(out.name + ".ckpt")  # 訓練中的檢查點
    if a.overwrite and not a.extend and ckpt_path.exists() and me == 0:  # 要重新訓練時
        ckpt_path.unlink()  # 由 rank 0 刪除舊檢查點
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:  # 在 rank 0 刪掉檢查點之前，任何 rank 都不可以去找檢查點
        import torch.distributed as dist  # 分散式通訊
        dist.barrier()  # 等 rank 0 刪完再繼續
    extra = {"model_source": source, "humanv0_sha256": sha}  # 側檔要記錄的主幹來源
    init_state = None  # 預設從 policy head 的複本開始
    if a.init_heads:  # 指定了前一階段的結果
        init, init_meta, _ = load_heads(a.init_heads, model, "cpu")  # 第二階段的檔也會把它的主幹載入 model
        init_state = init.state_dict()  # 前一階段的等級頭權重
        extra.update({"head_init": str(a.init_heads), "head_init_sha256": init_meta.get("state_sha256"),  # 記錄起點檔與其 sha256
                      "head_init_trunk_sha256": init_meta.get("trunk_sha256"), "head_init_inner_nll_best": init_meta.get("inner_nll_best")})  # 以及起點的主幹雜湊與內部 NLL
    try:  # 快取或折的設定有問題時改為回報錯誤
        heads, meta = train_heads(model, a.cache, cfg, out, extra_meta=extra, init_heads=init_state, exclude_folds=a.exclude_folds,  # 訓練並寫出等級頭與側檔
                                  folds_csv=a.folds_csv)  # 排除折時用的分折表
    except ValueError as e:  # 設定不一致
        if me == 0:  # 只由 rank 0 印訊息
            print(e, file=sys.stderr)  # 印出錯誤訊息
        return 2  # 結束碼 2 表示輸入有誤
    if me == 0:  # 只由 rank 0 印摘要
        fmt = lambda v: "-" if v is None else f"{v:.4f}"  # noqa: E731  數值格式化（缺值印 -）
        print(f"wrote {out} (train_folds {meta['train_folds']}): inner NLL {fmt(meta['inner_nll_init'])} -> {fmt(meta['inner_nll_best'])} at epoch {meta['best_epoch']:.2f}"  # 輸出檔、訓練折與內部 NLL 的變化
              + (f" (objective {fmt(meta['inner_objective_init'])} -> {fmt(meta['inner_objective_best'])}, rank CE "  # 有鑑別式損失時另印目標函數
             f"{fmt(meta['inner_disc_init'])} -> {fmt(meta['inner_disc_best'])})" if cfg.disc_weight > 0 else "")  # 以及等級交叉熵的變化
          + f", {'stopped early' if meta['stopped_early'] else 'ran to the end'}, {meta['steps_run']} steps, {meta['wall_seconds']:.0f} s"  # 是否提前停止、步數與耗時
              + (f", trunk fine-tuned ({meta['n_trunk_params']} parameters, lr {cfg.trunk_lr:g}), world {meta['world_size']}"  # 微調主幹時另印參數量與學習率
                 if meta["trunk_trained"] else ""))  # 只在主幹有訓練時才印
    return 0  # 正常結束

if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值當結束碼

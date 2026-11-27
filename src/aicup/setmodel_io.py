"""Saving and loading trained set models (aicup.settrain.Fitted) for the submission pipeline (item 2).

A saved model is two files with one stem: <stem>.pt, the torch state_dict as trained (float32 weights,
buffers included), and <stem>.json, the sidecar with everything needed to rebuild and audit it: `rung`
(a key of aicup.ladder.SET_RUNGS, whose make(d_in) rebuilds the module) and its vector `rep`, `seed`,
`d_in`, `d_glob`, the TrainCfg, the standardisation statistics of the Fitted (tok_mean / tok_std /
glob_mean / glob_std as float64 lists -- exactly the ones fit_set_model used, i.e. of the training
questions without the inner players), the inner early-stopping record (inner_ce, best_eval, best_epoch,
stopped_early, n_inner_players, n_inner_questions) and the per-epoch losses, the parameter count, the
profiles and feature names of the pipeline, the sha256 of the .pt bytes, the provenance of the code
that trained it (aicup.ladder.environment: git_head, git_dirty, code_sha256, code_files, torch, device)
and any `extra` fields the trainer adds (training folds, per_player, the questions' sha, the paired base
model and its sha256).

load_set_model rebuilds the Fitted on a device and returns it with the sidecar and a list of warnings;
its predict_set_model output equals the trained model's bit for bit on the same device. It refuses a
sidecar of another format, an unknown rung, other profiles or features than this pipeline computes, or a
.pt whose bytes do not match the recorded sha256; a code_sha256 that differs from the running code is a
warning (the training modules changed since training), not an error.
"""
import hashlib  # 計算檔案的 sha256
import json  # 讀取附屬檔
from dataclasses import asdict  # 把 TrainCfg 轉成字典
from datetime import datetime  # 記錄寫出時間
from pathlib import Path  # 路徑處理

import numpy as np  # 統計量的型別轉換
import torch  # 存取 state_dict

from aicup.humansl import PROFILES  # 本流程使用的 Human SL 剖面
from aicup.qfeatures import FEATURE_NAMES  # 本流程使用的題目特徵名稱
from aicup.settrain import Fitted, TrainCfg  # 訓練結果與訓練設定

FORMAT = 1  # 附屬檔的格式版本
STATS = ("tok_mean", "tok_std", "glob_mean", "glob_std")  # 要存下的標準化統計量
INNER = ("inner_ce", "best_eval", "best_epoch", "stopped_early", "n_inner_players", "n_inner_questions")  # 要存下的內層提早停止紀錄


def sidecar_path(pt_path):  # 附屬檔的路徑
    """<stem>.json next to <stem>.pt."""
    return Path(pt_path).with_suffix(".json")  # 換成 .json 副檔名


def file_sha256(path):  # 檔案內容的雜湊
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()  # 讀入整個檔案計算 sha256


def save_set_model(fitted, rung, seed, pt_path, extra=None):  # 儲存集合模型與附屬檔
    """Write <pt_path> (state_dict) and its sidecar for the Fitted of `rung` / `seed`; returns the sidecar dict.
    `extra` fields are added to the sidecar (and may not shadow its own keys)."""
    from aicup.aggregators import n_params  # 計算參數量
    from aicup.ladder import SET_RUNGS, environment, to_json  # 在函式內才匯入：rung 定義、程式來源紀錄與嚴格 JSON 輸出
    if rung not in SET_RUNGS:  # 未知的 rung
        raise ValueError(f"unknown rung {rung!r}; known: {list(SET_RUNGS)}")  # 拒絕存檔
    pt_path = Path(pt_path)  # 轉成 Path 物件
    pt_path.parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    torch.save({k: v.detach().cpu() for k, v in fitted.model.state_dict().items()}, pt_path)  # 以 CPU 張量存下權重
    meta = {"format": FORMAT, "rung": rung, "rep": SET_RUNGS[rung][0], "seed": int(seed), "d_in": int(len(fitted.tok_mean)),  # 格式、rung、向量表示、種子、輸入維度
            "d_glob": int(len(fitted.glob_mean)), "train_cfg": asdict(fitted.cfg),  # 題目層維度與訓練設定
            **{k: np.asarray(getattr(fitted, k), dtype=np.float64).tolist() for k in STATS},  # 標準化統計量
            **{k: getattr(fitted, k) for k in INNER}, "losses": list(fitted.losses), "n_params": n_params(fitted.model),  # 內層紀錄、損失與參數量
            "profiles": list(PROFILES), "features": list(FEATURE_NAMES), "state_sha256": file_sha256(pt_path),  # 剖面、特徵與權重檔的雜湊
            **environment(fitted.cfg.device), "written": datetime.now().isoformat(timespec="seconds")}  # 程式來源紀錄與寫出時間
    clash = sorted(set(extra or {}) & set(meta))  # 與既有欄位衝突的鍵
    if clash:  # 有衝突
        raise ValueError(f"extra sidecar fields shadow the model's own: {clash}")  # 拒絕覆蓋
    meta.update(extra or {})  # 加入額外欄位
    sidecar_path(pt_path).write_text(to_json(meta, indent=1) + "\n")  # 寫出附屬檔
    return meta  # 回傳附屬檔內容


def load_set_model(pt_path, device="cpu"):  # 載入集合模型並核對
    """(Fitted on `device`, sidecar dict, warnings) from <pt_path> and its sidecar; ValueError on a file that
    cannot be trusted (see the module docstring)."""
    from aicup.ladder import SET_RUNGS, code_version  # 在函式內才匯入：rung 定義與目前程式的雜湊
    pt_path = Path(pt_path)  # 轉成 Path 物件
    side = sidecar_path(pt_path)  # 附屬檔路徑
    if not pt_path.exists() or not side.exists():  # 缺少任一個檔案
        raise FileNotFoundError(f"{pt_path} and {side} are both needed")  # 兩個檔案都必須存在
    meta = json.loads(side.read_text())  # 讀入附屬檔
    if meta.get("format") != FORMAT:  # 格式版本不符
        raise ValueError(f"{side}: format {meta.get('format')!r}, this code reads {FORMAT}")  # 拒絕載入
    if meta["rung"] not in SET_RUNGS:  # 未知的 rung
        raise ValueError(f"{side}: unknown rung {meta['rung']!r}; known: {list(SET_RUNGS)}")  # 拒絕載入
    if meta["profiles"] != list(PROFILES) or meta["features"] != list(FEATURE_NAMES):  # 剖面或特徵與本流程不同
        raise ValueError(f"{side}: the model was trained with other profiles or features than this pipeline computes")  # 拒絕載入
    sha = file_sha256(pt_path)  # 權重檔實際的雜湊
    if sha != meta["state_sha256"]:  # 與記錄的雜湊不符
        raise ValueError(f"{pt_path}: sha256 {sha[:12]} does not match the sidecar's {meta['state_sha256'][:12]}")  # 權重檔與附屬檔記錄的雜湊不符，拒絕載入
    rep, make, _ = SET_RUNGS[meta["rung"]]  # 這個 rung 的向量表示與建構函式
    if meta.get("rep", rep) != rep:  # 記錄的向量表示與 rung 不一致
        raise ValueError(f"{side}: rep {meta['rep']!r} but the rung {meta['rung']} uses {rep!r}")  # 拒絕載入
    cfg = TrainCfg(**{**meta["train_cfg"], "device": device})  # 還原訓練設定並改用指定裝置
    with torch.random.fork_rng(devices=[]):  # 與 fit_set_model 相同的方式建立模組，權重隨後會被替換；外部的亂數狀態不受影響
        torch.manual_seed(cfg.seed)  # 使用訓練時的種子
        model = make(meta["d_in"])  # 建立模組
    state = torch.load(pt_path, map_location="cpu", weights_only=True)  # 讀取權重（只允許張量）
    model.load_state_dict(state)  # 載入權重
    model.to(device).eval()  # 移到裝置並切到評估模式
    fitted = Fitted(model, *(np.asarray(meta[k], dtype=np.float64) for k in STATS), cfg, losses=list(meta["losses"]),  # 以統計量與設定重建 Fitted
                    inner_ce=list(meta["inner_ce"]), best_eval=meta["best_eval"], best_epoch=meta["best_epoch"],  # 內層評估紀錄
                    stopped_early=meta["stopped_early"], n_inner_players=meta["n_inner_players"],  # 是否提早停止與內層玩家數
                    n_inner_questions=meta["n_inner_questions"])  # 內層題目數
    if len(fitted.tok_mean) != meta["d_in"] or len(fitted.glob_mean) != meta["d_glob"]:  # 統計量的維度與記錄不符
        raise ValueError(f"{side}: statistics of {len(fitted.tok_mean)} / {len(fitted.glob_mean)} columns, d_in / d_glob say "  # 回報不一致的維度
                         f"{meta['d_in']} / {meta['d_glob']}")  # 錯誤訊息的後半
    warnings = []  # 警告清單
    now = code_version()["code_sha256"]  # 目前程式的雜湊
    if meta.get("code_sha256") != now:  # 訓練時的程式與現在不同
        warnings.append(f"{side.name}: trained by other training code (code_sha256 {str(meta.get('code_sha256'))[:12]}, "  # 只記警告，模型照存檔使用
                        f"now {now[:12]}); the model is used as saved")  # 警告訊息的後半
    return fitted, meta, warnings  # 回傳模型、附屬檔與警告


def summary(meta):  # 報告用的附屬檔摘要
    """The sidecar fields worth a report line: rung, seed, d_in, best_epoch, train_folds, git_head, code_sha256, base model."""
    keys = ("rung", "seed", "d_in", "n_params", "best_epoch", "stopped_early", "train_folds", "per_player", "git_head",  # 要列出的欄位
            "code_sha256", "base_model", "base_model_sha256")  # 程式雜湊與配對的基礎模型
    return {k: meta.get(k) for k in keys}  # 缺少的欄位為 None

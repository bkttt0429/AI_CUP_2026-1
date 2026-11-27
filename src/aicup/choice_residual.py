"""M197 feature convention and atomic, provenance-checked compact parts.

The policy mask deliberately matches the probe: occupied board points excluded,
pass included. It is NOT a full legality mask (ko/suicide are not filtered).
"""
import os  # 行程 id、fsync 與原子改名
import shutil  # 查詢剩餘的磁碟空間
from pathlib import Path  # 路徑處理

import numpy as np  # 數值陣列
import torch  # 張量運算


def features(trunk, logits, occupied, played, projection):  # 計算每手 67 維的選擇殘差特徵
    """Return residual projection and (-log p, entropy, log1p rank)."""
    points = trunk.shape[-1]  # 盤上點數
    legal = torch.cat([~occupied, torch.ones(len(trunk), 1, dtype=torch.bool, device=trunk.device)], 1)  # 空點加上永遠可下的 pass
    lp = torch.log_softmax(logits.float().masked_fill(~legal, -torch.inf), dim=-1)  # 遮罩後的 policy 對數機率
    p = lp.exp()  # policy 機率
    mean = torch.einsum("bcp,bp->bc", trunk.float(), p[:, :points])  # 以 policy 加權的全盤主幹特徵平均
    local = torch.zeros_like(mean)  # 落點的主幹特徵（pass 維持 0）
    on = played < points  # 下在盤上的手（不是 pass）
    local[on] = trunk[on.nonzero()[:, 0], :, played[on]].float()  # 取落點那一格的主幹特徵
    chosen = lp.gather(1, played[:, None])[:, 0]  # 實際這一手的 log p
    entropy = -(p * lp.masked_fill(~legal, 0.)).sum(1)  # policy 的熵；遮罩點的 -inf 先換成 0，避免 0 × -inf 產生 NaN
    rank = (lp > chosen[:, None]).sum(1).float()  # 機率比這一手高的點數，即名次
    return torch.cat([(local - mean) @ projection,  # 落點減去加權平均後做隨機投影
                      torch.stack([-chosen.clamp(min=-20.), entropy, torch.log1p(rank)], 1)], 1)  # 再接三個 policy 純量（-log p 上限 20）


def quantize(x, mu, sd, scale=20.):  # 把特徵量化成 int8
    if not np.isfinite(x).all() or not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd <= 0).any():  # 特徵或校正值不合法
        raise ValueError("nonfinite feature or invalid calibration")  # 拒絕量化
    return np.clip(np.rint((x - mu) / sd * scale), -127, 127).astype(np.int8)  # 標準化後乘 scale（預設 20）、四捨五入並截在 int8 範圍內


def dequantize(q, mu, sd, scale=20.):  # 把 int8 還原成浮點特徵
    return q.astype(np.float32) * (sd / scale) + mu  # quantize 的反運算


def check_data(d):  # 檢查一個分段的資料是否自洽
    n, m = len(d["key"]), len(d["T"])  # 局數與總手數
    if len(set(d["key"].tolist())) != n:  # 局鍵重複
        raise ValueError("duplicate game key")  # 拒絕
    if any(len(d[k]) != n for k in ("LEN", "SUBJ", "BOTH")):  # 每局欄位的長度要等於局數
        raise ValueError("game metadata length mismatch")  # 拒絕
    off = d["OFF"]  # 每局在逐手陣列中的起訖位移
    if len(off) != n + 1 or off[0] != 0 or off[-1] != m or (np.diff(off) < 0).any():  # 位移須從 0 開始、單調不減，最後等於總手數
        raise ValueError("invalid offsets")  # 拒絕
    if d["XQ"].shape != (m, 67) or d["XQ"].dtype != np.int8:  # 量化特徵須為 [總手數, 67] 的 int8
        raise ValueError("invalid quantized features")  # 拒絕
    if any(len(d[k]) != m for k in ("MOV", "POS")):  # 逐手欄位的長度要等於總手數
        raise ValueError("ply metadata length mismatch")  # 拒絕
    if ((d["LEN"] < 0) | (d["LEN"] > 256)).any() or not np.isin(d["SUBJ"], [0, 1]).all():  # 局長須在 0-256，本人顏色須為 0 或 1
        raise ValueError("invalid game length or subject")  # 拒絕
    if not np.isin(d["MOV"], [0, 1]).all() or ((d["POS"] < 0) | (d["POS"] > 361)).any():  # 行棋方須為 0 或 1，落點須在 0-361
        raise ValueError("invalid mover or position")  # 拒絕
    for i in range(n):  # 逐局檢查手數對齊
        sl = slice(off[i], off[i + 1]); t = d["T"][sl]  # 這一局的手及其原始手數
        if ((t < 0) | (t >= d["LEN"][i])).any() or (np.diff(t) <= 0).any():  # 原始手數須在局長內且嚴格遞增
            raise ValueError("invalid original ply alignment")  # 拒絕
        if d["BOTH"][i]:  # 存雙方落子的局
            if not np.array_equal(t, np.arange(d["LEN"][i])):  # 每一手都必須存在
                raise ValueError("incomplete both-side game")  # 拒絕
        elif (d["MOV"][sl] != d["SUBJ"][i]).any():  # 只存本人落子的局卻出現對手的手
            raise ValueError("opponent ply in subject-only game")  # 拒絕


def valid_part(path, keys, run_id, input_id):  # 確認分段檔完整且屬於這次抽取
    try:  # 讀檔失敗一律視為無效
        with np.load(path, allow_pickle=False) as d:  # 不允許 pickle 的安全讀取
            if str(d["run_id"]) != run_id or str(d["input_id"]) != input_id or not np.array_equal(d["key"], keys):  # 來源識別或局清單不符
                return False  # 不是這次抽取的分段
            check_data({k: d[k] for k in d.files})  # 內容自洽檢查
        return True  # 通過所有檢查
    except (OSError, ValueError, KeyError, EOFError):  # 檔案損壞、欄位缺少或內容不合法
        return False  # 視為無效


def save_part(path, data, run_id, input_id, reserve_bytes=3 * 2**30):  # 原子寫入一個分段檔（預設保留 3 GiB 空間）
    check_data(data)  # 寫入前先檢查資料
    path = Path(path)  # 統一成 Path
    path.parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    need = sum(v.nbytes for v in data.values()) + 65536  # 估計需要的空間（多留 64 KiB）
    if shutil.disk_usage(path.parent).free < reserve_bytes + need:  # 剩餘空間低於保留量
        raise OSError("quota free space below reserve; extraction stopped without deleting files")  # 停止抽取，不刪除任何檔案
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")  # 每個行程各自的暫存檔名
    try:  # 確保暫存檔最後會被清掉
        with tmp.open("wb") as f:  # 先寫入暫存檔
            np.savez(f, **data, run_id=np.array(run_id), input_id=np.array(input_id))  # 連同來源識別一起存
            f.flush(); os.fsync(f.fileno())  # 確實寫到磁碟
        if not valid_part(tmp, data["key"], run_id, input_id):  # 讀回驗證
            raise ValueError("written part failed validation")  # 驗證失敗就不改名
        os.replace(tmp, path)  # 原子改名成正式檔
    finally:  # 不論成功與否
        tmp.unlink(missing_ok=True)  # 清掉殘留的暫存檔

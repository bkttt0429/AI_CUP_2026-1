"""L1b stage 1: a frozen KataGo trunk under rank-unknown metadata, plus one policy head per rank.

The trunk is humanv0's (or any KataGo `Model`), run once per position with the metadata row of
`neutral_meta()`: the same human / KGS / byo-yomi profile as `humansl.make_profile_meta`, but both
ranks unknown (flags 4 and 5 set, the rank thermometers empty), which humanv0 saw in training.
`RankHeads` holds `n_ranks` deep copies of the model's `PolicyHead`; head k is trained only on the
moves of rank-k players, so its masked log-softmax is log p(a | H, k). At step 0 every head equals
humanv0's own policy under the neutral metadata (the known starting point of any improvement).

`score_positions` has the shape of `humansl.score_positions`: the log p of each target move under
every head, [N, n_ranks]. `save_heads` / `load_heads` store the heads' state_dict as .pt with a JSON
sidecar (format, n_ranks, levels, sha256 of the .pt, provenance, caller metadata) and refuse a file
whose sha256 or n_ranks do not match.
"""
import copy  # 深拷貝 policy 頭
import json  # 讀寫側檔
import os  # 原子改名
from pathlib import Path  # 路徑處理

import numpy as np  # 數值陣列
import torch  # 網路與張量
from katago.train.model_pytorch import ExtraOutputs, PolicyHead  # KataGo 的額外輸出收集器與 policy 頭

from aicup.humansl import make_profile_meta  # Human SL 的剖面 metadata
from aicup.ladder import environment, to_json  # 程式版本與執行環境的紀錄、嚴格 JSON 輸出
from aicup.metric import LEVELS  # 10 個等級
from aicup.setmodel_io import file_sha256, sidecar_path  # 檔案雜湊與側檔路徑

FORMAT = 1  # 側檔的格式版本
N_MOVES = 362  # 361 個點加上 pass，19 路棋盤的 policy 輸出長度


def neutral_meta():  # 雙方等級未知的 metadata
    """humansl.make_profile_meta's profile with both ranks unknown: rank-unknown flags set, thermometers empty."""
    meta = make_profile_meta("1d")  # 先取一般的剖面（等級隨後清除）
    meta.inverseBRank = 0  # 清空黑方等級
    meta.inverseWRank = 0  # 清空白方等級
    meta.bRankIsUnknown = True  # 黑方等級未知
    meta.wRankIsUnknown = True  # 白方等級未知
    return meta  # 回傳中性的 metadata


def neutral_rows():  # 中性 metadata 的兩列
    """Metadata rows [2, 192] float32 for Black to move (row 0) and White to move (row 1)."""
    meta = neutral_meta()  # 雙方等級未知的 metadata
    return np.stack([meta.get_metadata_row(nextPlayer=c, boardArea=361) for c in ("b", "w")]).astype(np.float32)  # 黑方輪下與白方輪下各一列


class RankHeads(torch.nn.Module):  # 等級頭：每個等級一份 policy 頭
    """n_ranks independent deep copies of a KataGo PolicyHead; forward gives policy channel 0 of each."""

    def __init__(self, policy_head: PolicyHead, n_ranks: int = len(LEVELS)):  # 由模型的 policy 頭複製
        super().__init__()  # 初始化 nn.Module
        self.n_ranks = int(n_ranks)  # 等級頭的個數
        self.heads = torch.nn.ModuleList([copy.deepcopy(policy_head) for _ in range(self.n_ranks)])  # 每個等級一份深拷貝
        self.requires_grad_(True)  # 即使從凍結的模型複製也可以訓練

    def forward(self, trunk, mask, mask_sum_hw, mask_sum):  # 所有等級頭的 policy logits
        """[B, n_ranks, 362] policy logits (channel 0 of every head), computed in fp32 as KataGo does."""
        outs = [head(trunk, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=None)[:, 0, :]  # 每個頭取 policy 第 0 通道
                for head in self.heads]  # 逐一等級頭
        return torch.stack(outs, dim=1)  # 疊成 [B, n_ranks, 362]


class _GraphExtraOutputs(ExtraOutputs):  # 保留計算圖的額外輸出收集器
    """ExtraOutputs that keeps the autograd graph: KataGo's report() returns value.detach(), which would cut the
    stage-2 gradient off the trunk. The values are the same tensors (detach shares storage), so no-grad callers
    are unaffected."""

    def report(self, name, value):  # 覆寫 KataGo 的 report
        self.available.append(name)  # 記下可取得的輸出名稱
        if name in self.requested:  # 是呼叫端要求的輸出
            self.returned[name] = value  # 不做 detach，保留梯度


def trunk_features(model, bins, globs, colors, rows=None):  # 取得主幹最後一層的特徵與遮罩
    """(trunkfinal [B, C, 19, 19] float32, mask [B, 1, 19, 19], mask_sum_hw [B, 1, 1, 1], mask_sum) of a batch.

    `rows` are the metadata rows [2, 192] indexed by `colors` (0 Black, 1 White to move); None means
    neutral_rows(). A model without a metadata encoder ignores them. The mask quantities are computed
    as Model.forward computes them (plane 0 = on-board). The caller chooses no_grad / autocast.
    """
    device = bins.device  # 輸入所在的裝置
    bins, globs = bins.float(), globs.float()  # 轉成浮點
    input_meta = None  # 預設不給 metadata
    if model.get_has_metadata_encoder():  # 模型有 metadata 編碼器
        rows_t = torch.as_tensor(neutral_rows() if rows is None else rows, dtype=torch.float32, device=device)  # 黑白兩列的 metadata
        input_meta = rows_t[torch.as_tensor(colors, device=device).long()]  # 依輪到的顏色取列
    extra = _GraphExtraOutputs(["trunkfinal"])  # 要求回傳主幹最後一層
    model(bins, globs, input_meta=input_meta, extra_outputs=extra)  # 前向計算（輸出由 extra 收集）
    mask = bins[:, 0:1, :, :].contiguous()  # 平面 0 即盤上遮罩
    mask_sum_hw = torch.sum(mask, dim=(2, 3), keepdim=True)  # 每個局面的盤上點數
    mask_sum = torch.sum(mask)  # 整個批次的盤上點數
    return extra.returned["trunkfinal"].float(), mask, mask_sum_hw, mask_sum  # 回傳主幹特徵與遮罩相關的量


def masked_logp(logits, legal):  # 遮罩非法點後的 log-softmax
    """log-softmax over the last axis with illegal points (legal False) at -inf; legal is [B, 362] bool."""
    legal_t = torch.as_tensor(legal, device=logits.device).bool()  # 遮罩搬到 logits 所在的裝置
    return torch.log_softmax(logits.masked_fill(~legal_t[:, None, :], float("-inf")), dim=-1)  # 每個頭套用同一個遮罩


def score_positions(model, heads, bins, globs, legal, target, colors, batch=64):  # 以各等級頭為目標落子評分
    """log p of each target move under every head: [N, n_ranks] float32 (humansl.score_positions' shape).

    The trunk runs under no_grad and, on CUDA, bf16 autocast; the heads run in fp32.
    """
    device = next(heads.parameters()).device  # 等級頭所在的裝置
    n = len(target)  # 局面數
    out = np.empty((n, heads.n_ranks), dtype=np.float32)  # 輸出 [N, n_ranks]
    rows = neutral_rows()  # 中性的 metadata
    with torch.no_grad():  # 只做推論
        for i in range(0, n, batch):  # 逐批次
            j = min(n, i + batch)  # 批次結尾
            b = torch.as_tensor(np.asarray(bins[i:j]), device=device)  # 盤面平面
            g = torch.as_tensor(np.asarray(globs[i:j]), device=device)  # 全域特徵
            c = torch.as_tensor(np.asarray(colors[i:j]), device=device).long()  # 輪到的顏色
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):  # 主幹在 CUDA 上用 bf16
                trunk, mask, mask_sum_hw, mask_sum = trunk_features(model, b, g, c, rows)  # 主幹特徵
            logits = heads(trunk.float(), mask.float(), mask_sum_hw.float(), mask_sum.float())  # 等級頭以 fp32 計算
            logp = masked_logp(logits, torch.as_tensor(np.asarray(legal[i:j]), device=device))  # 遮罩後的 log p
            tgt = torch.as_tensor(np.asarray(target[i:j]), device=device).long()[:, None, None].expand(-1, heads.n_ranks, 1)  # 實際落點，擴展到每個頭
            out[i:j] = logp.gather(2, tgt)[:, :, 0].float().cpu().numpy()  # 取實際落點的 log p
    return out  # 回傳 [N, n_ranks]


def atomic_save(obj, path):  # 原子寫入 torch 檔
    """torch.save to <path>.tmp, then rename over path: a crash mid-write never leaves a truncated file."""
    path = Path(path)  # 統一成 Path
    tmp = path.with_name(path.name + ".tmp")  # 暫存檔名
    try:  # 寫入失敗時清掉暫存檔
        torch.save(obj, tmp)  # 先寫暫存檔
    except BaseException:  # 包含中斷在內的任何例外
        tmp.unlink(missing_ok=True)  # 刪除寫到一半的檔案
        raise  # 重新拋出例外
    os.replace(tmp, path)  # 原子改名


def trunk_path(path):  # 微調後主幹的檔案路徑
    """<stem>.trunk.pt next to a heads .pt: the fine-tuned model's state_dict of a stage-2 run."""
    path = Path(path)  # 統一成 Path
    return path.with_name(path.stem + ".trunk.pt")  # 與等級頭同名，副檔名為 .trunk.pt


def save_heads(heads, path, meta, model_state=None):  # 儲存等級頭與側檔
    """state_dict -> path (.pt) and the sidecar path.with_suffix('.json'); returns the sidecar dict.

    model_state (stage 2, the trunk fine-tuned): the whole model's state_dict is written to trunk_path(path)
    and the sidecar records its file name and sha256, so load_heads restores the trunk with the heads."""
    path = Path(path)  # 統一成 Path
    path.parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    atomic_save(heads.state_dict(), path)  # 原子寫入等級頭
    trunk = {}  # 主幹相關的側檔欄位
    if model_state is not None:  # 第二階段有微調主幹
        tp = trunk_path(path)  # 主幹檔的路徑
        atomic_save(model_state, tp)  # 原子寫入主幹
        trunk = {"trunk_file": tp.name, "trunk_sha256": file_sha256(tp)}  # 記下檔名與雜湊
    device = next(heads.parameters()).device  # 等級頭所在的裝置
    side = {"format": FORMAT, "n_ranks": heads.n_ranks, "levels": list(LEVELS[:heads.n_ranks]),  # 格式、頭數與等級
            "n_params": int(sum(p.numel() for p in heads.parameters())), "state_sha256": file_sha256(path),  # 參數量與 .pt 的雜湊
            **trunk, **environment(str(device)), **meta}  # 主幹欄位、程式版本與執行環境、呼叫端的資訊
    sidecar_path(path).write_text(to_json(side, indent=1) + "\n")  # 寫出側檔
    return json.loads(sidecar_path(path).read_text())  # 讀回檔案，回傳的內容與寫入的 JSON 一致


def load_heads(path, model, device="cpu"):  # 讀回並核對等級頭
    """(RankHeads, sidecar, warnings) restored from path onto model.policy_head's architecture.

    ValueError when the sidecar is missing, its format or n_ranks does not fit, or the .pt sha256 differs.
    A code_sha256 other than the current code's is a warning (the heads were trained by other code).
    A stage-2 sidecar names a fine-tuned trunk (trunk_file): it is loaded into `model` in place, after the same
    sha256 check (ValueError when the file is missing, differs or does not fit the model).
    """
    path = Path(path)  # 統一成 Path
    side_path = sidecar_path(path)  # 側檔路徑
    if not side_path.exists():  # 沒有側檔
        raise ValueError(f"{path}: sidecar {side_path.name} is missing")  # 拒絕
    meta = json.loads(side_path.read_text())  # 讀入側檔
    if meta.get("format") != FORMAT:  # 格式版本不符
        raise ValueError(f"{path}: sidecar format {meta.get('format')!r}, this code reads {FORMAT}")  # 拒絕
    sha = file_sha256(path)  # .pt 的實際雜湊
    if meta.get("state_sha256") != sha:  # 與側檔的記錄不符
        raise ValueError(f"{path}: sha256 {sha[:12]} differs from the sidecar's {str(meta.get('state_sha256'))[:12]}")  # 拒絕
    n_ranks = meta.get("n_ranks")  # 等級頭的個數
    if not isinstance(n_ranks, int) or n_ranks < 1 or meta.get("levels") != list(LEVELS[:n_ranks]):  # 頭數或等級與本程式不符
        raise ValueError(f"{path}: n_ranks {n_ranks!r} / levels {meta.get('levels')!r} do not fit this code's LEVELS")  # 拒絕
    if meta.get("trunk_file"):  # 第二階段：有微調後的主幹
        tp = path.with_name(meta["trunk_file"])  # 主幹檔的路徑
        if not tp.exists():  # 主幹檔不存在
            raise ValueError(f"{path}: the fine-tuned trunk {tp.name} is missing")  # 拒絕
        tsha = file_sha256(tp)  # 主幹檔的實際雜湊
        if meta.get("trunk_sha256") != tsha:  # 與側檔的記錄不符
            raise ValueError(f"{tp}: sha256 {tsha[:12]} differs from the sidecar's {str(meta.get('trunk_sha256'))[:12]}")  # 拒絕
        try:  # 載入失敗時轉成 ValueError
            model.load_state_dict(torch.load(tp, map_location="cpu", weights_only=True))  # 把微調後的主幹就地載入模型
        except RuntimeError as e:  # 權重與模型不合
            raise ValueError(f"{tp}: state_dict does not fit the model ({e})") from e  # 拒絕
    heads = RankHeads(model.policy_head, n_ranks)  # 以模型的 policy 頭建立架構
    state = torch.load(path, map_location=device, weights_only=True)  # 讀入等級頭的權重
    try:  # 載入失敗時轉成 ValueError
        heads.load_state_dict(state)  # 套用權重
    except RuntimeError as e:  # 權重與架構不合
        raise ValueError(f"{path}: state_dict does not fit model.policy_head's architecture ({e})") from e  # 拒絕
    heads.to(device).eval()  # 搬到裝置並切成推論模式
    warnings = []  # 警告清單
    current = environment("cpu")["code_sha256"]  # 目前程式的 code_sha256：ladder.CODE_FILES 所列模組的雜湊（不含 l1b.py、l1btrain.py）
    if meta.get("code_sha256") != current:  # 與側檔記錄的不同，視為由其他版本的程式訓練
        warnings.append(f"{path}: trained by other training code (code_sha256 {str(meta.get('code_sha256'))[:12]} vs "  # 只警告，不拒絕
                        f"{current[:12]}); predictions may differ from the training run's")  # 警告訊息的後半
    return heads, meta, warnings  # 回傳等級頭、側檔內容與警告

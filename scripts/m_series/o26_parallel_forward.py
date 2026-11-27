"""Parallelize only embedding computation; keep the full-episode loss and base state intact."""

def make_forward(net):  # 回傳訓練用的前向函式（多卡時只分散嵌入計算）
    import torch  # 需要時才匯入 PyTorch
    if torch.cuda.device_count() < 2:  # 可見的 GPU 不到兩張：不必平行
        return net  # 直接使用原網路，不包裝

    class BF16Replica(torch.nn.Module):  # 在每個複本執行緒裡明確設定 BF16 的包裝
        def __init__(self, base):  # 包住原網路
            super().__init__()  # 初始化父類別
            self.base = base  # 共用原網路與其參數

        def forward(self, *args, **kwargs):  # 依呼叫端的 autocast 狀態前向
            # parallel_apply 會傳遞 autocast 的開啟狀態，但在新建的執行緒中會遺失 dtype。
            # o26 的配方是 BF16；FP32 的呼叫維持不轉型，供數值回歸測試使用。
            with torch.autocast("cuda", dtype=torch.bfloat16,  # 明確指定 BF16
                                enabled=torch.is_autocast_enabled("cuda")):  # 只在呼叫端已開啟 autocast 時生效
                return self.base(*args, **kwargs)  # 呼叫原網路

    print("parallel helper v2: explicit BF16 in replica threads", flush=True)  # 在紀錄中標明載入的是修正版
    return torch.nn.DataParallel(BF16Replica(net), device_ids=list(range(torch.cuda.device_count())))  # 以 DataParallel 把嵌入的前向分到所有可見 GPU，輸出收回主卡

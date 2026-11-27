#!/usr/bin/env python
"""Training of the official rank baseline (Tutorial rank-prediction.ipynb §2-3) on the shards
written by scripts/baseline_extract.py.

Loader, model and optimisation are the notebook's: webdataset shards with a 1000-sample
shuffle buffer, several loader workers only if they read the validation set exactly once,
GoRankResNet(num_classes=10), AdamW lr 1e-3, batch 256, cross-entropy, 5 epochs, FP32 (cuDNN
default TF32 convolutions); the epoch with the best validation accuracy is best_model.pth.
The validation set is the player-grouped slice of folds 1-4 (aicup.baseline.split_players).
Writes <ckpt>/best_model.pth, epoch_{k}.pth, history.csv and run.json.
"""
import argparse
import glob
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.baseline import load_tutorial  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402


def log(msg):
    print(f"{datetime.now():%m-%d %H:%M:%S} {msg}", flush=True)


def decode_sample(sample):
    return (torch.from_numpy(sample["features.npy"]).float(),
            torch.tensor(int(sample["rank_idx.txt"]), dtype=torch.long),
            sample["game_name.txt"])


def make_loader(shards, workers, shuffle, batch_size):
    import webdataset as wds

    ds = wds.WebDataset(shards, shardshuffle=1 if shuffle else 0)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.decode().map(decode_sample).batched(batch_size, partial=True)
    return DataLoader(ds, batch_size=None, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def pick_workers(val_shards, expected, batch_size, candidate=4):
    """Count the validation set once: keep `candidate` workers only if the count matches."""
    total = sum(len(y) for _x, y, _n in make_loader(val_shards, candidate, False, batch_size))
    if total == expected:
        log("num_workers=%d verified (%d samples)" % (candidate, total))
        return candidate
    log("num_workers=%d read %d samples (expected %d), falling back to 0" % (candidate, total, expected))
    return 0


def validate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total, batches = 0.0, 0, 0, 0
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss_sum += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batches += 1
    model.train()
    return (loss_sum / batches if batches else 0.0,
            100 * correct / total if total else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "data" / "baseline_official"))
    ap.add_argument("--ckpt", default=str(ROOT / "models" / "baseline_official"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4, help="loader workers to try (the notebook's candidate)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    feat, ckpt = Path(a.features), Path(a.ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    split = json.loads((feat / "split.json").read_text())
    train_shards = sorted(glob.glob(str(feat / "train" / "*.tar.gz")))
    val_shards = sorted(glob.glob(str(feat / "val" / "*.tar.gz")))
    log("shards: train %d / val %d" % (len(train_shards), len(val_shards)))
    workers = pick_workers(val_shards, split["val"]["written"], a.batch_size, a.workers)
    train_loader = make_loader(train_shards, workers, True, a.batch_size)
    val_loader = make_loader(val_shards, workers, False, a.batch_size)

    model = load_tutorial()[1](num_classes=len(LEVELS)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=a.lr)
    criterion = torch.nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    log("params {:,} | batch {} | lr {} | epochs {} | device {}".format(n_params, a.batch_size, a.lr, a.epochs, device))

    history, best_val_acc, best_epoch = [], 0.0, None
    model.train()
    for epoch in range(a.epochs):
        t0 = time.time()
        total_loss, correct, total, steps = 0.0, 0, 0, 0
        for inputs, labels, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            steps += 1
            if steps % 500 == 0:
                log("  ep%d step %d  loss %.4f  acc %.2f%%  %.1f min"
                    % (epoch + 1, steps, total_loss / steps, 100 * correct / total, (time.time() - t0) / 60))

        train_acc = 100 * correct / total if total else 0.0
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        history.append({"epoch": epoch + 1, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc,
                        "minutes": (time.time() - t0) / 60})
        log("Epoch %d/%d  train acc %.2f%% (%d samples)  |  val loss %.4f acc %.2f%%  |  %.1f min"
            % (epoch + 1, a.epochs, train_acc, total, val_loss, val_acc, (time.time() - t0) / 60))
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch + 1
            torch.save(model.state_dict(), ckpt / "best_model.pth")
            log("  new best val acc %.2f%%, saved best_model.pth" % best_val_acc)
        torch.save(model.state_dict(), ckpt / f"epoch_{epoch + 1}.pth")
        pd.DataFrame(history).to_csv(ckpt / "history.csv", index=False)

    run = {"created": datetime.now().isoformat(timespec="seconds"), "args": vars(a),
           "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
           "torch": torch.__version__, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
           "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_benchmark": torch.backends.cudnn.benchmark,
           "loader_workers": workers, "train_samples_per_epoch": total, "val_samples": split["val"]["written"],
           "params": n_params, "best_epoch": best_epoch, "best_val_acc": best_val_acc,
           "minutes": (time.time() - t_start) / 60}
    (ckpt / "run.json").write_text(json.dumps(run, indent=1))
    log("training done, best val acc %.2f%% (epoch %s), %.1f min" % (best_val_acc, best_epoch, run["minutes"]))


if __name__ == "__main__":
    main()

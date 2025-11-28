# TinyViM/train_tinyvim_s.py
import argparse
import os
from pathlib import Path
import time
import datetime
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from timm.data import resolve_data_config, create_transform
from timm.models import create_model

from torchvision import datasets

# 匯入本專案的 `model`，讓 TinyViM_S / B / L 透過 timm 的 @register_model 註冊
import model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        "TinyViM-S 訓練自訂資料集（不含 Lie-PEFT / liera）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 模型與資料相關設定
    parser.add_argument(
        "--model",
        default="TinyViM_S",
        type=str,
        help="模型名稱（TinyViM_S / TinyViM_B / TinyViM_L ...）",
    )
    parser.add_argument(
        "--num-classes",
        default=1000,
        type=int,
        help="43",
    )
    parser.add_argument(
        "--train-dir",
        required=True,
        type=str,
        help="訓練資料夾（ImageFolder 格式，底下每個子資料夾是一個類別）",
    )
    parser.add_argument(
        "--val-dir",
        required=True,
        type=str,
        help="驗證資料夾（ImageFolder 格式）",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        type=str,
        help="預訓練權重檔路徑（.pth，checkpoint['model'] 格式；留空則從頭訓練）",
    )

    # 訓練超參數
    parser.add_argument("--epochs", default=50, type=int, help="訓練總 epoch 數")
    parser.add_argument("--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("--lr", default=1e-3, type=float, help="學習率")
    parser.add_argument("--weight-decay", default=0.05, type=float, help="weight decay")
    parser.add_argument("--num-workers", default=4, type=int, help="DataLoader workers")

    # 環境與輸出
    parser.add_argument(
        "--device", default="cuda", type=str, help="裝置：cuda 或 cpu"
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints_simple",
        type=str,
        help="輸出目錄（會存放 log / 最佳權重 / 最後權重 / metrics.csv）",
    )

    return parser.parse_args()


def create_dataloaders(train_dir, val_dir, model, batch_size, num_workers, device):
    # 使用 timm 提供的資料前處理，確保跟 TinyViM 的預設設定一致
    config = resolve_data_config({}, model=model)

    transform_train = create_transform(
        **config, is_training=True
    )  # 隨機 crop / flip 等
    transform_val = create_transform(
        **config, is_training=False
    )  # 中心 crop，用於驗證

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    return train_loader, val_loader, len(train_dataset.classes)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)  # [B, num_classes]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        running_correct += preds.eq(targets).sum().item()
        total += images.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    acc1 = 100.0 * running_correct / total if total > 0 else 0.0
    print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, train_acc1={acc1:.2f}%")
    return {"loss": avg_loss, "acc1": acc1}


def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            running_correct += preds.eq(targets).sum().item()
            total += images.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    acc1 = 100.0 * running_correct / total if total > 0 else 0.0
    print(f"Epoch {epoch}: val_loss={avg_loss:.4f}, val_acc1={acc1:.2f}%")
    return {"loss": avg_loss, "acc1": acc1}


def main():
    args = parse_args()

    device = torch.device(args.device)
    cudnn.benchmark = True

    # 建立 TinyViM 模型（不啟用 distillation head）
    print(f"Creating model: {args.model}, num_classes={args.num_classes}")
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        distillation=False,
        pretrained=False,
    )

    # 若有給預訓練權重，先載入（通常是官方 ImageNet-1K 預訓練）
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt)
        msg = model.load_state_dict(state, strict=False)
        print(f"Checkpoint loaded with message: {msg}")

    model.to(device)

    # 準備 DataLoader
    train_dir = Path(args.train-dir) if hasattr(args, "train-dir") else Path(args.train_dir)
    val_dir = Path(args.val_dir)
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Train dir not found: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Val dir not found: {val_dir}")

    train_loader, val_loader, actual_num_classes = create_dataloaders(
        str(train_dir),
        str(val_dir),
        model,
        args.batch_size,
        args.num_workers,
        device,
    )

    print(f"Detected {actual_num_classes} classes from ImageFolder.")
    if actual_num_classes != args.num_classes:
        print(
            f"Warning: args.num_classes={args.num_classes}, "
            f"but dataset has {actual_num_classes} classes. "
            f"請確認是否要將 --num-classes 設成 {actual_num_classes}。"
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # 建立輸出目錄
    out_root = PROJECT_ROOT / args.output_dir
    time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_dir = out_root / f"{args.model}_{time_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    history = []
    best_acc1 = 0.0
    best_ckpt_path = out_dir / "checkpoint_best.pth"
    last_ckpt_path = None

    start_time = time.time()

    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_stats = evaluate(
            model, val_loader, criterion, device, epoch
        )

        scheduler.step()

        # 儲存最後一個 epoch 權重
        last_ckpt_path = out_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            },
            last_ckpt_path,
        )

        # 若有最佳 val_acc1 就更新 best
        if val_stats["acc1"] > best_acc1:
            best_acc1 = val_stats["acc1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(f"New best val_acc1: {best_acc1:.2f}% at epoch {epoch}")

        log_row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc1": train_stats["acc1"],
            "val_loss": val_stats["loss"],
            "val_acc1": val_stats["acc1"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(log_row)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time: {total_time_str}")

    # 存成 CSV 方便後處理 / 繪圖
    metrics_csv = out_dir / "metrics.csv"
    if history:
        fieldnames = list(history[0].keys())
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow(row)
        print(f"Saved metrics CSV to {metrics_csv}")

    print(f"Best val_acc1: {best_acc1:.2f}%")
    if best_ckpt_path.exists():
        print(f"Best checkpoint saved at: {best_ckpt_path}")
    if last_ckpt_path is not None and last_ckpt_path.exists():
        print(f"Last checkpoint saved at: {last_ckpt_path}")


if __name__ == "__main__":
    main()
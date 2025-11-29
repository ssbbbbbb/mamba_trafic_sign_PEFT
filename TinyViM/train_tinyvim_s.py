# TinyViM/train_tinyvim_s.py
import argparse
import os
from pathlib import Path
import time
import datetime
import csv
import sys

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

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    # Lie-Group PEFT（Generalized Tensor-based PEFT via Lie Group）
    from lie_peft import apply_lie_peft, freeze_backbone
    _HAS_LIE_PEFT = True
except ImportError:
    _HAS_LIE_PEFT = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _Tee:
    """同時把輸出寫到原本的 stdout/stderr 與檔案。"""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def parse_args():
    parser = argparse.ArgumentParser(
        "TinyViM-S 訓練自訂資料集（支援 Lie-PEFT / liera）",
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

    # 只訓練分類 head（凍結 backbone）
    parser.add_argument(
        "--only-train-head",
        action="store_true",
        help="只訓練分類 head，其他 backbone 參數全部凍結",
    )

    # Lie-Group PEFT（Generalized Tensor-based PEFT via Lie Group）
    parser.add_argument(
        "--use-lie-peft",
        action="store_true",
        help="啟用 Lie-PEFT 微調（會先凍結 backbone，再注入低秩 Lie-PEFT 參數）",
    )
    parser.add_argument(
        "--lie-rank",
        type=int,
        default=4,
        help="Lie-PEFT rank r",
    )
    parser.add_argument(
        "--lie-alpha",
        type=float,
        default=16.0,
        help="Lie-PEFT scaling factor alpha",
    )
    parser.add_argument(
        "--lie-target",
        type=str,
        default="all",
        choices=["all", "head", "last_stage"],
        help="要在那些層套用 Lie-PEFT 參數",
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

    iterable = loader
    if _HAS_TQDM:
        iterable = tqdm(loader, desc=f"Train epoch {epoch}", leave=False)

    for i, (images, targets) in enumerate(iterable):
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

    iterable = loader
    if _HAS_TQDM:
        iterable = tqdm(loader, desc=f"Val epoch {epoch}", leave=False)

    with torch.no_grad():
        for images, targets in iterable:
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
        checkpoint_model = ckpt.get("model", ckpt)
        state_dict = model.state_dict()
        # 將輸出 head 等 shape 不相符的權重刪掉，避免類別數改成 43 時發生 size mismatch
        for k in list(checkpoint_model.keys()):
            if k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
                print(
                    f"Removing key {k} from pretrained checkpoint due to shape mismatch: "
                    f"{checkpoint_model[k].shape} vs {state_dict[k].shape}"
                )
                del checkpoint_model[k]
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f"Checkpoint loaded with message: {msg}")

    model.to(device)

    # 若啟用 Lie-PEFT，使用 Lie 群低秩參數進行微調
    if args.use_lie_peft:
        if not _HAS_LIE_PEFT:
            raise ImportError(
                "偵測不到 lie_peft 模組，無法使用 --use-lie-peft。"
                "請確認 lie_peft.py 可被 Python 匯入，或在專案根目錄執行。"
            )
        print("Enable Lie-Group based PEFT (Generalized Tensor-based PEFT via Lie Group).")
        # 先凍結 backbone，再注入 Lie-PEFT 參數
        freeze_backbone(model)
        apply_lie_peft(
            model,
            rank=args.lie_rank,
            alpha=args.lie_alpha,
            target=args.lie_target,
        )
        # 此時 optimizer 只會看到 requires_grad=True 的 Lie-PEFT 參數與 head 參數

    # 若只訓練 head，先凍結 backbone，再開啟 head 相關參數
    elif args.only_train_head:
        print("Only training classifier head: freezing all backbone parameters.")
        for n, p in model.named_parameters():
            p.requires_grad = False

        trainable_names = []
        # Unfreeze head
        if hasattr(model, "head"):
            for p in model.head.parameters():
                p.requires_grad = True
            trainable_names.append("head")
        # 某些 timm 模型使用 fc 作為分類 head
        if hasattr(model, "fc"):
            for p in model.fc.parameters():
                p.requires_grad = True
            trainable_names.append("fc")
        # 若有 distillation head 也一併打開
        if hasattr(model, "head_dist") and getattr(model, "head_dist") is not None:
            for p in model.head_dist.parameters():
                p.requires_grad = True
            trainable_names.append("head_dist")

        # 列出實際可訓練參數名稱方便確認
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"Trainable modules: {trainable_names}")
        print(f"Trainable parameters (by name): {trainable_params}")

    # 準備 DataLoader
    # argparse 會把選項名稱中的 '-' 轉成屬性名稱裡的 '_'，
    # 所以這裡正確的寫法就是 args.train_dir（不會有 args.train-dir 這種屬性）。
    train_dir = Path(args.train_dir)
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
    # 只對 requires_grad=True 的參數做優化，確保凍結的層不會更新
    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # 建立輸出目錄
    out_root = PROJECT_ROOT / args.output_dir
    time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_dir = out_root / f"{args.model}_{time_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # 將之後的所有終端輸出同時寫到檔案（類似 tee）
    log_path = out_dir / "train_log.txt"
    log_file = log_path.open("a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    print(f"Output dir: {out_dir}")
    print(f"Logging to: {log_path}")

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

        # 若有 matplotlib，順便畫出訓練 / 驗證曲線圖
        if _HAS_MPL:
            epochs = [h["epoch"] for h in history]
            train_loss = [h["train_loss"] for h in history]
            val_loss = [h["val_loss"] for h in history]
            train_acc1 = [h["train_acc1"] for h in history]
            val_acc1 = [h["val_acc1"] for h in history]

            # Loss 曲線
            plt.figure()
            plt.plot(epochs, train_loss, label="train_loss")
            plt.plot(epochs, val_loss, label="val_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Loss Curve - {args.model}")
            plt.legend()
            loss_png = out_dir / "loss_curve.png"
            plt.savefig(loss_png, bbox_inches="tight")
            plt.close()

            # Acc@1 曲線
            plt.figure()
            plt.plot(epochs, train_acc1, label="train_acc1")
            plt.plot(epochs, val_acc1, label="val_acc1")
            plt.xlabel("Epoch")
            plt.ylabel("Acc@1 (%)")
            plt.title(f"Accuracy Curve - {args.model}")
            plt.legend()
            acc_png = out_dir / "acc1_curve.png"
            plt.savefig(acc_png, bbox_inches="tight")
            plt.close()

            print(f"Saved curves to {out_dir}")
        else:
            print("matplotlib 未安裝，只輸出 metrics.csv，不繪製曲線圖。")

    print(f"Best val_acc1: {best_acc1:.2f}%")
    if best_ckpt_path.exists():
        print(f"Best checkpoint saved at: {best_ckpt_path}")
    if last_ckpt_path is not None and last_ckpt_path.exists():
        print(f"Last checkpoint saved at: {last_ckpt_path}")


if __name__ == "__main__":
    main()
import argparse
from pathlib import Path
from typing import Optional
import csv

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from timm.data import resolve_data_config, create_transform
from timm.models import create_model

from torchvision import datasets

# 匯入本專案的 `model`，讓 TinyViM_S / B / L 透過 timm 的 @register_model 註冊
import model  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        "TinyViM 簡單測試腳本（載入 checkpoint，對單一 ImageFolder Test 資料夾做評估）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="TinyViM_L",
        type=str,
        help="模型名稱（TinyViM_S / TinyViM_B / TinyViM_L ...）",
    )
    parser.add_argument(
        "--num-classes",
        required=True,
        type=int,
        help="資料集的類別數（要跟訓練時一致）",
    )
    parser.add_argument(
        "--test-dir",
        required=True,
        type=str,
        help="測試資料夾（ImageFolder 格式，底下每個子資料夾是一個類別）",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="訓練好的 checkpoint_best.pth 路徑（裡面包含 checkpoint['model']）",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="測試 batch size",
    )
    parser.add_argument(
        "--num-workers",
        default=4,
        type=int,
        help="DataLoader workers 數量",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="裝置：cuda 或 cpu",
    )
    parser.add_argument(
        "--save-preds",
        default="",
        type=str,
        help="若非空字串，會將 (path, label, pred) 存成 CSV 到指定路徑",
    )
    return parser.parse_args()


def create_test_loader(test_dir, model, batch_size, num_workers, device):
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config, is_training=False)

    dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return loader, dataset.classes


@torch.no_grad()
def evaluate(model, loader, device, save_path: Optional[Path] = None):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total = 0
    running_loss = 0.0
    running_correct = 0

    rows = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        running_correct += preds.eq(targets).sum().item()
        total += images.size(0)

        if save_path is not None:
            # 將每張圖片的路徑 / 真實標籤 / 預測標籤記錄下來
            for idx in range(images.size(0)):
                rows.append(
                    {
                        "index": total - images.size(0) + idx,
                        "target": int(targets[idx].item()),
                        "pred": int(preds[idx].item()),
                    }
                )

    avg_loss = running_loss / total if total > 0 else 0.0
    acc1 = 100.0 * running_correct / total if total > 0 else 0.0

    if save_path is not None and rows:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "target", "pred"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Saved predictions CSV to: {save_path}")

    return {"loss": avg_loss, "acc1": acc1, "total": total}


def main():
    args = parse_args()

    device = torch.device(args.device)
    cudnn.benchmark = True

    print(f"Creating model: {args.model}, num_classes={args.num_classes}")
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        distillation=False,
        pretrained=False,
    )

    # 載入訓練好的 checkpoint（通常是 train_nopeft 產生的 checkpoint_best.pth）
    ckpt_path = Path(args.checkpoint)
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    checkpoint_model = ckpt.get("model", ckpt)
    state_dict = model.state_dict()

    # 若 head 的 shape 不符（例如從 ImageNet-1K 轉 4 類），移除不相容權重
    for k in list(checkpoint_model.keys()):
        if k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
            print(
                f"Removing key {k} from checkpoint due to shape mismatch: "
                f"{checkpoint_model[k].shape} vs {state_dict[k].shape}"
            )
            del checkpoint_model[k]
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Checkpoint loaded with message: {msg}")

    model.to(device)

    test_loader, classes = create_test_loader(
        args.test_dir,
        model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    print(f"Detected {len(classes)} classes in Test folder:")
    for i, name in enumerate(classes):
        print(f"  class_id={i}: {name}")

    save_path = Path(args.save_preds) if args.save_preds else None

    stats = evaluate(model, test_loader, device, save_path=save_path)
    print(
        f"Test set: total={stats['total']} images, "
        f"loss={stats['loss']:.4f}, acc1={stats['acc1']:.2f}%"
    )


if __name__ == "__main__":
    main()



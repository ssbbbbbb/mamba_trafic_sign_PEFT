import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from timm.data import resolve_data_config, create_transform
from timm.models import create_model

# 匯入本專案的 `model`，讓 TinyViM_S / B / L 透過 timm 的 @register_model 註冊
import model


def load_image(path: str, transform, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    return tensor.to(device)


def parse_args():
    parser = argparse.ArgumentParser(
        "TinyViM-S inference on自訂資料集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="TinyViM_S",
        type=str,
        help="模型名稱（保持 TinyViM_S 即可）",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="官方預訓練權重檔路徑（.pth，checkpoint['model'] 格式）",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=str,
        help="要做推論的圖片資料夾（會遞迴搜尋所有子目錄）",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="輸出結果的資料夾（會產生 predictions.csv）",
    )
    parser.add_argument(
        "--classes-file",
        default="imagenet_classes.txt",
        type=str,
        help="ImageNet 類別名稱列表檔案（每行一個類別名稱，行號對應 index+1）",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="裝置：cuda 或 cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)

    # 建立 TinyViM-S 模型（不啟用蒸餾 head，方便做單一輸出）
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        num_classes=1000,          # ImageNet-1K 預設 1000 類
        distillation=False,        # 測試時只用主 head 即可
        pretrained=False,
    )

    # 載入官方預訓練權重
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    msg = model.load_state_dict(state, strict=False)
    print(f"Checkpoint loaded with message: {msg}")

    model.to(device)
    model.eval()

    # 建立與訓練時一致的前處理
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 載入類別名稱（若檔案存在）
    class_names = None
    classes_path = Path(args.classes_file)
    if classes_path.is_file():
        with classes_path.open("r", encoding="utf-8") as cf:
            class_names = [line.strip() for line in cf if line.strip()]
        print(f"Loaded {len(class_names)} class names from {classes_path}")
    else:
        print(f"Warning: classes file '{classes_path}' not found, will只輸出類別 index。")

    out_csv = out_dir / "predictions.csv"
    print(f"Writing predictions to {out_csv}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

    with out_csv.open("w", encoding="utf-8") as f:
        # 若有類別名稱，額外輸出一欄 top1_name
        if class_names is not None:
            f.write("filename,top1_idx,top1_name\n")
        else:
            f.write("filename,top1_idx\n")

        img_paths = []
        for root, _, files in os.walk(data_dir):
            for name in files:
                if Path(name).suffix.lower() in exts:
                    img_paths.append(Path(root) / name)

        img_paths = sorted(img_paths)

        for p in img_paths:
            rel_path = p.relative_to(data_dir)
            img = load_image(str(p), transform, device)

            with torch.no_grad():
                logits = model(img)  # [1, 1000]
                probs = torch.softmax(logits, dim=1)
                top1 = int(probs.argmax(dim=1).item())

            if class_names is not None and 0 <= top1 < len(class_names):
                name = class_names[top1]
                f.write(f"{rel_path.as_posix()},{top1},{name}\n")
            else:
                f.write(f"{rel_path.as_posix()},{top1}\n")

    print("Inference finished.")


if __name__ == "__main__":
    main()



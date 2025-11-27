import argparse
import csv
import os
from pathlib import Path

import torch
from PIL import Image
from timm.data import resolve_data_config, create_transform
from timm.models import create_model

import model  # 註冊 TinyViM
from lie_peft import apply_lie_peft

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def load_image(path: str, transform, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    return tensor.to(device)


def parse_args():
    parser = argparse.ArgumentParser(
        "TinyViM_S + Lie-PEFT 自訂 Test 集推論",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model", default="TinyViM_S", type=str)
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="TinyViM_S_best.pth（訓練時存的 best checkpoint，checkpoint['model'] 格式）",
    )
    parser.add_argument(
        "--data-csv",
        required=True,
        type=str,
        help="Test 標註 CSV 檔路徑（第 7 欄為類別，第 8 欄為圖片路徑）",
    )
    parser.add_argument(
        "--img-root",
        required=True,
        type=str,
        help="圖片根目錄（會和 CSV 裡的圖片路徑 join 起來）",
    )
    parser.add_argument(
        "--output-csv",
        default="test_predictions.csv",
        type=str,
        help="輸出預測結果 CSV 檔路徑",
    )
    parser.add_argument(
        "--label-col",
        type=int,
        default=6,  # 第 7 欄（0-based index 6）
        help="CSV 中類別欄位的 index（0-based）",
    )
    parser.add_argument(
        "--path-col",
        type=int,
        default=7,  # 第 8 欄（0-based index 7）
        help="CSV 中圖片路徑欄位的 index（0-based）",
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="CSV 第一列是否為標題列（有的話會自動略過）",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=43,
        help="模型輸出類別數（你的資料集是 43 類）",
    )
    parser.add_argument(
        "--lie-rank",
        type=int,
        default=8,
        help="訓練時使用的 Lie-PEFT rank（要和訓練時相同）",
    )
    parser.add_argument(
        "--lie-alpha",
        type=float,
        default=16.0,
        help="訓練時使用的 Lie-PEFT alpha（要和訓練時相同）",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="cuda 或 cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1. 建立 TinyViM_S 結構（43 類，沒有蒸餾 head）
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        distillation=False,
        pretrained=False,
    )

    # 2. 套用 Lie-PEFT 結構（必須和訓練時設定一樣）
    print("Apply Lie-PEFT wrappers...")
    apply_lie_peft(model, rank=args.lie_rank, alpha=args.lie_alpha, target="all")

    # 3. 載入訓練好的權重
    ckpt_path = Path(args.checkpoint)
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    msg = model.load_state_dict(state, strict=True)
    print(f"Checkpoint loaded: {msg}")

    model.to(device)
    model.eval()

    # 4. 建立與訓練時相容的前處理
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    img_root = Path(args.img_root)
    out_csv_path = Path(args.output_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 5. 讀取 Test CSV：第 7 欄(label_col)、第 8 欄(path_col)
    samples = []
    with open(args.data_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        if args.has_header:
            next(reader, None)
        for row in reader:
            if len(row) <= max(args.label_col, args.path_col):
                continue
            label = int(row[args.label_col])
            img_rel = row[args.path_col].strip()
            samples.append((img_rel, label))

    print(f"Loaded {len(samples)} samples from {args.data_csv}")

    # 6. 逐張圖片做推論，計算 Top-1 accuracy，並把結果寫出
    correct = 0
    total = 0

    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img_path", "label_gt", "label_pred"])

        # 若安裝了 tqdm，就用進度條顯示推論進度；否則維持原本的簡單迴圈
        if _HAS_TQDM:
            iter_samples = tqdm(samples, total=len(samples), desc="Inference")
        else:
            iter_samples = samples

        with torch.no_grad():
            for img_rel, label in iter_samples:
                img_path = Path(img_rel)
                if not img_path.is_absolute():
                    img_path = img_root / img_rel

                if not img_path.is_file():
                    print(f"[WARN] Image not found: {img_path}")
                    continue

                img = load_image(str(img_path), transform, device)
                logits = model(img)  # [1, num_classes]
                pred = int(logits.argmax(dim=1).item())

                total += 1
                if pred == label:
                    correct += 1

                writer.writerow([str(img_rel), int(label), pred])

    if total > 0:
        acc = correct / total * 100.0
        print(f"Test accuracy: {correct}/{total} = {acc:.2f}%")
    else:
        print("No valid samples were evaluated.")

    print(f"Predictions saved to {out_csv_path}")


if __name__ == "__main__":
    main()
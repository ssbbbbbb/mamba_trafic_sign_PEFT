import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from timm.data import resolve_data_config, create_transform
from timm.models import create_model

import model  # 註冊 TinyViM 模型

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
        "TinyViM_S ImageNet 預訓練權重，在自訂 Test 集上做測試（不經過微調，只是當 baseline）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model", default="TinyViM_S", type=str)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../checkpoints/tinyvim_s_1000e.pth",
        help="官方 TinyViM_S ImageNet-1K 預訓練權重（checkpoint['model'] 格式）",
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
        default="checkpoints/test_predictions_imnet_pretrained.csv",
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
        "--classes-file",
        type=str,
        default=None,
        help="類別名稱列表檔案（每行一個類別名稱，行號對應 index+1，例如 COCO 或自訂）",
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

    # 1. 建立 TinyViM_S 結構（ImageNet-1K：num_classes=1000）
    print(f"Creating model: {args.model}")
    model_t = create_model(
        args.model,
        num_classes=1000,   # 保持與預訓練權重一致
        distillation=False,
        pretrained=False,
    )

    # 2. 載入官方 ImageNet 預訓練權重
    ckpt_path = Path(args.checkpoint)
    print(f"Loading ImageNet pretrained checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)

    # 官方權重裡會帶有 distillation head（dist_head.*），
    # 但這裡我們建立的模型是 distillation=False，沒有這些參數。
    # 用 strict=False 來忽略這些「多出來的」鍵即可。
    msg = model_t.load_state_dict(state, strict=False)
    print(f"Checkpoint loaded (strict=False): {msg}")

    model_t.to(device)
    model_t.eval()

    # 3. 建立與預訓練時相容的前處理
    config = resolve_data_config({}, model=model_t)
    transform = create_transform(**config)

    img_root = Path(args.img_root)
    out_csv_path = Path(args.output_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 若提供類別名稱檔案，先讀進來
    class_names = None
    if args.classes_file is not None:
        classes_path = Path(args.classes_file)
        if classes_path.is_file():
            with classes_path.open("r", encoding="utf-8") as cf:
                class_names = [line.strip() for line in cf if line.strip()]
            print(f"Loaded {len(class_names)} class names from {classes_path}")
        else:
            print(f"[WARN] classes file '{classes_path}' not found，將只輸出類別 index。")

    # 4. 讀取 Test CSV：第 7 欄(label_col)、第 8 欄(path_col)
    samples = []
    with open(args.data_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        if args.has_header:
            next(reader, None)
        for row in reader:
            if len(row) <= max(args.label_col, args.path_col):
                continue
            try:
                label = int(row[args.label_col])
            except ValueError:
                continue
            img_rel = row[args.path_col].strip()
            samples.append((img_rel, label))

    print(f"Loaded {len(samples)} samples from {args.data_csv}")
    print("注意：這裡的 GT label 是 0~42，自訂資料集的類別；")
    print("      模型輸出是 ImageNet-1K 的 1000 類，兩者語意並不對齊，")
    print("      所以下面算出的 accuracy 只是『強行比對 index』的 baseline，")
    print("      期望值會接近隨機，純粹拿來感受 fine-tune 前後的落差。")

    # 5. 逐張圖片做推論，計算「直接用 index 相等」的 Top-1 accuracy，並把結果寫出
    correct = 0
    total = 0

    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if class_names is not None:
            writer.writerow(["img_path", "label_gt", "label_pred", "label_pred_name"])
        else:
            writer.writerow(["img_path", "label_gt", "label_pred"])

        # 若安裝了 tqdm，就顯示進度條；否則用原本的迴圈
        if _HAS_TQDM:
            iter_samples = tqdm(samples, total=len(samples), desc="Inference (ImageNet-pretrained)")
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
                logits = model_t(img)  # [1, 1000]
                pred = int(logits.argmax(dim=1).item())

                total += 1
                if pred == label:
                    correct += 1

                if class_names is not None and 0 <= pred < len(class_names):
                    writer.writerow(
                        [str(img_rel), int(label), pred, class_names[pred]]
                    )
                else:
                    writer.writerow([str(img_rel), int(label), pred])

    if total > 0:
        acc = correct / total * 100.0
        print(f"Naive index-based accuracy (期望接近隨機): {correct}/{total} = {acc:.2f}%")
    else:
        print("No valid samples were evaluated.")

    print(f"Predictions saved to {out_csv_path}")


if __name__ == "__main__":
    main()



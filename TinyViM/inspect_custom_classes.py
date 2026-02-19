import argparse
import os

from torchvision import datasets


def parse_args():
    parser = argparse.ArgumentParser(
        "檢查 CUSTOM 資料集（Train 資料夾）的類別順序與編號",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        required=True,
        type=str,
        help="資料根目錄，例如 /mnt/e/FOLDER/學校作業/電腦視覺/data",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    train_root = os.path.join(args.data_path, "Train")
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"Train 資料夾不存在: {train_root}")

    # 不需要實際的 transform，這裡只是想看類別對應
    dataset = datasets.ImageFolder(train_root)

    print(f"Train root: {train_root}")
    print(f"共 {len(dataset.classes)} 個類別")
    print("classes（資料夾名稱，依 ImageFolder 排序）：")
    for i, name in enumerate(dataset.classes):
        print(f"  idx={i:2d}  folder='{name}'")

    print("\nclass_to_idx 映射（資料夾名稱 → 類別 index）：")
    for name, idx in sorted(dataset.class_to_idx.items(), key=lambda x: x[1]):
        print(f"  '{name}' -> {idx}")


if __name__ == "__main__":
    main()



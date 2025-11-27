import argparse
import random
from pathlib import Path


def split_train_val(data_root: Path, val_ratio: float = 0.2, seed: int = 42):
    """
    將 data_root/Train/0..42 下的部分圖片移動到 data_root/Val/0..42 作為驗證集。
    注意：這是「移動」檔案，而不是複製，請先確認有備份。
    """
    train_root = data_root / "Train"
    val_root = data_root / "Val"
    val_root.mkdir(exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    random.seed(seed)

    for cls_dir in sorted(train_root.iterdir()):
        if not cls_dir.is_dir():
            continue

        images = [p for p in cls_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in exts]
        if not images:
            continue

        random.shuffle(images)
        n_val = int(len(images) * val_ratio)
        if n_val == 0:
            continue

        target_dir = val_root / cls_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for img in images[:n_val]:
            img.rename(target_dir / img.name)

        print(f"class {cls_dir.name}: moved {n_val}/{len(images)} images to Val")

    print("Done splitting train/val.")


def main():
    parser = argparse.ArgumentParser(
        description="將 data/Train/0..42 切一部分到 data/Val/0..42 當驗證集的小工具")
    parser.add_argument(
        "--data-root", type=str, required=True,
        help=r"資料根目錄，例如 E:\FOLDER\學校作業\電腦視覺\data")
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="驗證集比例 (預設 0.2 表示 20%)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="隨機種子")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_train_val(data_root, val_ratio=args.val_ratio, seed=args.seed)


if __name__ == "__main__":
    main()



import argparse
import random
from pathlib import Path


def split_train_val(data_root: Path, val_ratio: float = 0.2, seed: int = 42):
    """
    將 data_root/train 底下的圖片，依「類別資料夾」切一部分到 data_root/val 做驗證集。

    自動支援兩種結構：
    1）一層結構（只有類別）：
        data_root/
          train/
            class0/
            class1/
            ...

    2）兩層結構（場景 / 子資料集 / 類別）：
        data_root/
          train/
            sceneA/
              class0/
              class1/
            sceneB/
              class0/
              class1/

    注意：這是「移動」檔案，而不是複製，請先確認有備份。
    """
    train_root = data_root / "train"
    val_root = data_root / "val"
    val_root.mkdir(exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    random.seed(seed)

    any_moved = False

    # 第一層：可能是「場景資料夾」也可能是「直接的類別資料夾」
    for first_level in sorted(train_root.iterdir()):
        if not first_level.is_dir():
            continue

        # 檢查這層底下是否直接就是圖片檔（代表：一層結構 -> 類別資料夾）
        direct_images = [
            p for p in first_level.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]

        if direct_images:
            # 一層結構：first_level 本身是「類別資料夾」
            random.shuffle(direct_images)
            n_val = int(len(direct_images) * val_ratio)
            if n_val == 0:
                continue

            target_dir = val_root / first_level.name
            target_dir.mkdir(parents=True, exist_ok=True)

            for img in direct_images[:n_val]:
                img.rename(target_dir / img.name)
                any_moved = True

            print(
                f"class {first_level.name}: "
                f"moved {n_val}/{len(direct_images)} images to val"
            )
        else:
            # 兩層結構：first_level 是「場景 / 子資料集」，裡面才是類別資料夾
            scene_dir = first_level

            for cls_dir in sorted(scene_dir.iterdir()):
                if not cls_dir.is_dir():
                    continue

                images = [
                    p for p in cls_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in exts
                ]
                if not images:
                    continue

                random.shuffle(images)
                n_val = int(len(images) * val_ratio)
                if n_val == 0:
                    continue

                # 在 val 底下建立對應的 scene / class 結構
                target_dir = val_root / scene_dir.name / cls_dir.name
                target_dir.mkdir(parents=True, exist_ok=True)

                for img in images[:n_val]:
                    img.rename(target_dir / img.name)
                    any_moved = True

                print(
                    f"scene {scene_dir.name}, class {cls_dir.name}: "
                    f"moved {n_val}/{len(images)} images to val"
                )

    if not any_moved:
        print("沒有找到符合條件的圖片（請確認 train 底下的結構與檔案副檔名）。")
    else:
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



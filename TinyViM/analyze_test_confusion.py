import argparse
import csv
import os
from collections import Counter

import numpy as np

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def parse_args():
    parser = argparse.ArgumentParser(
        "從 test_predictions.csv 快速分析 per-class confusion 的小工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="checkpoints/test_predictions.csv",
        help="由 inference 腳本輸出的 test_predictions.csv 路徑",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=43,
        help="類別數（你的資料目前是 43 類）",
    )
    parser.add_argument(
        "--out-img",
        type=str,
        default=None,
        help="輸出混淆矩陣圖的路徑（預設會根據 CSV 路徑存成 *_confusion.png）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    num_classes = args.num_classes
    # confusion[i][j] = GT 為 i、Pred 為 j 的次數
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    total = 0
    correct = 0

    with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or header[:3] != ["img_path", "label_gt", "label_pred"]:
            print("警告：CSV header 與預期不符，仍然嘗試讀取第 2、3 欄作為 GT / Pred")

        for row in reader:
            if len(row) < 3:
                continue
            try:
                gt = int(row[1])
                pred = int(row[2])
            except ValueError:
                continue

            if 0 <= gt < num_classes and 0 <= pred < num_classes:
                confusion[gt][pred] += 1
                total += 1
                if gt == pred:
                    correct += 1

    if total == 0:
        print("沒有有效樣本可供分析，請確認 CSV 內容是否正確。")
        return

    overall_acc = correct / total * 100.0
    print(f"總共統計樣本數: {total}")
    print(f"整體 Top-1 accuracy: {correct}/{total} = {overall_acc:.2f}%")
    print()

    print("=== 每個 GT 類別的統計（只列出有出現過的 GT 類別） ===")
    print("gt_class,total,correct,acc%,top_pred,top_pred_count,top_pred_ratio")

    # 同時計算「最常被預測成哪一類」的 mapping
    gt_to_top_pred = {}

    for gt in range(num_classes):
        row = confusion[gt]
        row_total = sum(row)
        if row_total == 0:
            continue

        row_correct = row[gt]
        row_acc = row_correct / row_total * 100.0

        # 找出該 GT 下最常出現的 Pred
        counter = Counter()
        for pred, cnt in enumerate(row):
            if cnt > 0:
                counter[pred] = cnt
        top_pred, top_cnt = counter.most_common(1)[0]
        top_ratio = top_cnt / row_total * 100.0

        gt_to_top_pred[gt] = (top_pred, top_cnt, row_total)

        print(
            f"{gt},{row_total},{row_correct},{row_acc:.2f},"
            f"{top_pred},{top_cnt},{top_ratio:.2f}"
        )

    print()
    print("=== 依照『幾乎被固定映射成某一類』排序的 GT → Pred 映射概況（top_pred_ratio 高到低） ===")
    print("gt_class,top_pred,top_pred_ratio,total_samples")

    # 將 gt 依照 top_pred_ratio 排序，方便你看哪些幾乎被固定映射
    sortable = []
    for gt, (top_pred, top_cnt, row_total) in gt_to_top_pred.items():
        ratio = top_cnt / row_total * 100.0
        sortable.append((ratio, gt, top_pred, row_total))

    sortable.sort(reverse=True)  # ratio 由高到低

    for ratio, gt, top_pred, row_total in sortable:
        print(f"{gt},{top_pred},{ratio:.2f},{row_total}")

    # 產生並儲存混淆矩陣圖
    if not _HAS_MPL:
        print("未安裝 matplotlib，無法輸出混淆矩陣圖。可以先安裝 matplotlib 後再執行。")
        return

    cm = np.array(confusion, dtype=np.float32)
    # 以每個 GT 類別為 row 做 row-wise normalization（看每類被分到哪裡的比例）
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    if args.out_img is None:
        base, _ = os.path.splitext(args.csv)
        out_img = base + "_confusion.png"
    else:
        out_img = args.out_img

    os.makedirs(os.path.dirname(out_img), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_title("Normalized Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ticks = np.arange(num_classes)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, fontsize=6, rotation=90)
    ax.set_yticklabels(ticks, fontsize=6)

    # 在每一格上標註「原始計數」（不是比例），方便直接看到數量
    for i in range(num_classes):
        for j in range(num_classes):
            val = int(cm[i, j])
            if val == 0:
                continue
            # 根據 normalized 值決定字的顏色，避免看不清楚
            norm_ij = cm_norm[i, j]
            color = "white" if norm_ij > 0.5 else "black"
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                fontsize=4,
                color=color,
            )

    plt.tight_layout()
    fig.savefig(out_img, dpi=300)
    plt.close(fig)

    print(f"混淆矩陣圖已儲存到: {out_img}")


if __name__ == "__main__":
    main()


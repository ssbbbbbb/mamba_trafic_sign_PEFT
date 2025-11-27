import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        "根據目前 confusion 結果，對 TinyViM_S_best.pth 的輸出類別做一次性重排（治標）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../test_output/TinyViM_S_best.pth",
        help="原始訓練好的 checkpoint 路徑（含 'model' state_dict）",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=43,
        help="輸出類別數（你的自訂資料集目前為 43 類）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../test_output/TinyViM_S_best_remapped.pth",
        help="輸出重新整理後的 checkpoint 路徑",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint 檔案: {ckpt_path}")

    print(f"載入 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 取得 state_dict
    state = ckpt.get("model", ckpt)

    num_classes = args.num_classes

    # 這個 mapping 來自你剛剛跑出的 confusion 結果：
    # gt_class,total,correct,acc%,top_pred,top_pred_count,top_pred_ratio
    # 我們用「每個 GT 最常被預測成哪一類」當作目前模型輸出 idx 的語意，
    # 然後把「GT index」當成新的輸出順序。
    #
    # perm[gt] = 舊的輸出 idx（top_pred），也就是：
    # new_weight[gt, :] = old_weight[perm[gt], :]
    perm_list = [
        0,   # 0 -> 0
        1,   # 1 -> 1
        12,  # 2 -> 12
        23,  # 3 -> 23
        34,  # 4 -> 34
        38,  # 5 -> 38
        39,  # 6 -> 39
        40,  # 7 -> 40
        41,  # 8 -> 41
        42,  # 9 -> 42
        2,   # 10 -> 2
        3,   # 11 -> 3
        4,   # 12 -> 4
        5,   # 13 -> 5
        6,   # 14 -> 6
        7,   # 15 -> 7
        8,   # 16 -> 8
        9,   # 17 -> 9
        10,  # 18 -> 10
        13,  # 19 -> 13
        13,  # 20 -> 13（注意：這裡代表訓練時本來就高度混淆，無法完美區分）
        14,  # 21 -> 14
        15,  # 22 -> 15
        16,  # 23 -> 16
        17,  # 24 -> 17
        18,  # 25 -> 18
        19,  # 26 -> 19
        20,  # 27 -> 20
        21,  # 28 -> 21
        22,  # 29 -> 22
        24,  # 30 -> 24
        25,  # 31 -> 25
        26,  # 32 -> 26
        27,  # 33 -> 27
        28,  # 34 -> 28
        29,  # 35 -> 29
        30,  # 36 -> 30
        30,  # 37 -> 30（同樣：與 36 一樣映射，代表本身就混在一起）
        32,  # 38 -> 32
        32,  # 39 -> 32
        35,  # 40 -> 35
        36,  # 41 -> 36
        36,  # 42 -> 36
    ]

    if len(perm_list) != num_classes:
        raise ValueError(
            f"perm_list 長度 ({len(perm_list)}) 和 num_classes ({num_classes}) 不一致"
        )

    perm = torch.tensor(perm_list, dtype=torch.long)

    # 對所有「輸出維度涉及 num_classes」的權重做重排：
    # - 若 shape[0] == num_classes：視為 [out_features, ...]，在 dim 0 做重排
    # - 若 shape[1] == num_classes：視為 [..., out_features]，在 dim 1 做重排
    #
    # 這會同時處理：
    # - 最後一層 Linear 的 weight / bias
    # - Lie-PEFT Linear 的 A (out_f, r), B (r, out_f) 等等
    #
    # 注意：由於訓練本身存在混淆（多個 GT 對到同一個舊 idx），這個操作是「治標」，
    # 無法把本來就分不開的類別 magically 分開，只是讓輸出 index 的語意更貼近 GT。
    modified_keys = []

    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        t = tensor
        changed = False

        if t.ndim >= 1 and t.shape[0] == num_classes:
            t = t[perm]
            changed = True

        if t.ndim >= 2 and t.shape[1] == num_classes:
            t = t[:, perm]
            changed = True

        if changed:
            state[key] = t
            modified_keys.append(key)

    print("已對下列參數做輸出維度重排（治標）：")
    for k in modified_keys:
        print(f"  - {k} {tuple(state[k].shape)}")

    # 寫回 ckpt
    if "model" in ckpt:
        ckpt["model"] = state
    else:
        ckpt = state

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(ckpt, out_path)
    print(f"已將重排後的 checkpoint 儲存到: {out_path}")
    print("之後在推論時，請改用這個新的 .pth 檔作為 --checkpoint。")


if __name__ == "__main__":
    main()



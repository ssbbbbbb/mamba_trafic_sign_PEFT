import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiePEFTConv2d(nn.Module):
    """
    Lie 群版本的 Conv2d PEFT：
    - 冷凍原本的 conv 權重 W0 (out_c, in_c, kH, kW)
    - 學一個低秩切空間 Delta = A @ B ∈ R^{out_c x out_c}
    - 透過矩陣指數 G = exp(Delta) ∈ GL(out_c)
    - 在輸出通道維度做群作用：W = G @ W0_flat
    這樣就等於把每個卷積核視為 Lie 群元素，更新在對應的 Lie 代數上完成。
    """

    def __init__(self, base_conv: nn.Conv2d, rank: int = 4, alpha: float = 16.0):
        super().__init__()
        assert isinstance(base_conv, nn.Conv2d)
        self.base = base_conv
        for p in self.base.parameters():
            p.requires_grad = False

        out_c = self.base.out_channels
        self.rank = rank
        # 與 LoRA 類似的 scaling
        self.scaling = alpha / float(rank)

        # 確保 A / B 一開始就跟 base_conv 在同一個 device，避免 AMP 在 unscale 時混到 CPU 參數
        device = self.base.weight.device
        self.A = nn.Parameter(torch.zeros(out_c, rank, device=device))
        self.B = nn.Parameter(torch.zeros(rank, out_c, device=device))

        # 初始化：A 隨機，B 置零（開始時接近 identity）
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    # 讓外部看起來仍然像 Conv2d，方便 fuse 等工具使用
    @property
    def in_channels(self):
        return self.base.in_channels

    @property
    def out_channels(self):
        return self.base.out_channels

    @property
    def kernel_size(self):
        return self.base.kernel_size

    @property
    def stride(self):
        return self.base.stride

    @property
    def padding(self):
        return self.base.padding

    @property
    def dilation(self):
        return self.base.dilation

    @property
    def groups(self):
        return self.base.groups

    @property
    def bias(self):
        return self.base.bias

    @property
    def weight(self):
        # 給像 Conv2d_BN.fuse 這種函式使用的介面
        return self._effective_weight()

    def _effective_weight(self):
        w0 = self.base.weight  # (out_c, in_c, kH, kW)
        device = w0.device
        out_c = w0.shape[0]
        # 在 no-autocast 區塊中用 float32 計算，再轉回原 dtype
        with torch.cuda.amp.autocast(enabled=False):
            w0_flat = w0.view(out_c, -1).to(device=device, dtype=torch.float32)  # (out_c, D)
            A = self.A.to(device, dtype=torch.float32)
            B = self.B.to(device, dtype=torch.float32)
            delta = (A @ B) / self.scaling  # (out_c, out_c), float32
            G = torch.matrix_exp(delta)  # (out_c, out_c), float32
            w_flat = G @ w0_flat  # (out_c, D), float32
            w_flat = w_flat.to(dtype=w0.dtype)
        return w_flat.view_as(w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._effective_weight()
        return F.conv2d(
            x,
            w,
            self.base.bias,
            stride=self.base.stride,
            padding=self.base.padding,
            dilation=self.base.dilation,
            groups=self.base.groups,
        )


class LiePEFTLinear(nn.Module):
    """
    Lie 群版本的 Linear PEFT：
    - 冷凍原本的線性層 W0 ∈ R^{out x in}
    - 在輸出維度上建構 G = exp(A @ B)，W = G @ W0
    - bias 也用同一個 G 做變換，對應論文中在同一 Lie 群上做一致更新。
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 4, alpha: float = 16.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        out_f = self.base.out_features
        self.rank = rank
        self.scaling = alpha / float(rank)

        # 同樣讓 A / B 落在 base_linear 對應的 device（通常是 CUDA）
        device = self.base.weight.device
        self.A = nn.Parameter(torch.zeros(out_f, rank, device=device))
        self.B = nn.Parameter(torch.zeros(rank, out_f, device=device))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def weight(self):
        return self._effective_weight()

    @property
    def bias(self):
        # 只在需要時計算，避免在沒有 bias 的情況下出錯
        if self.base.bias is None:
            return None
        _, G = self._effective_weight_and_group()
        return G @ self.base.bias

    def _effective_weight_and_group(self):
        w0 = self.base.weight  # (out_f, in_f)
        device = w0.device
        with torch.cuda.amp.autocast(enabled=False):
            w0_f = w0.to(device=device, dtype=torch.float32)
            A = self.A.to(device, dtype=torch.float32)
            B = self.B.to(device, dtype=torch.float32)
            delta = (A @ B) / self.scaling  # (out_f, out_f), float32
            G = torch.matrix_exp(delta)  # float32
            w = (G @ w0_f).to(dtype=w0.dtype)
            G = G.to(dtype=w0.dtype)
        return w, G

    def _effective_weight(self):
        w, _ = self._effective_weight_and_group()
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, G = self._effective_weight_and_group()
        if self.base.bias is not None:
            b = G @ self.base.bias
        else:
            b = None
        return F.linear(x, w, b)


def freeze_backbone(model: nn.Module) -> None:
    """
    冷凍整個 TinyViM backbone 的參數，之後只訓練 Lie-PEFT 相關參數。
    """
    for p in model.parameters():
        p.requires_grad = False


def _wrap_module_with_lie_peft(module: nn.Module, rank: int, alpha: float,
                               target: Literal["all", "head", "last_stage"]) -> nn.Module:
    """
    將 Conv2d / Linear 替換成對應的 Lie-PEFT wrapper。
    注意：已經是 LiePEFTConv2d / LiePEFTLinear 的就不要再包一次。
    """
    if isinstance(module, LiePEFTConv2d) or isinstance(module, LiePEFTLinear):
        return module
    if isinstance(module, nn.Conv2d):
        return LiePEFTConv2d(module, rank=rank, alpha=alpha)
    if isinstance(module, nn.Linear):
        return LiePEFTLinear(module, rank=rank, alpha=alpha)
    return module


def apply_lie_peft(model: nn.Module,
                   rank: int = 4,
                   alpha: float = 16.0,
                   target: Literal["all", "head", "last_stage"] = "all") -> None:
    """
    在 TinyViM 上套用 Generalized Tensor-based PEFT（Lie 群版本）。

    - target="all": 對所有 Conv2d / Linear 做 Lie-PEFT（建議用於小資料集微調）
    - target="head": 只處理分類 head（線性層）
    - target="last_stage": 只處理最後一個 stage + head
    """

    # 先簡化：目前 "all" / "head" / "last_stage" 都共用同一個遞迴，
    # 差別主要在是否進入某些子模組，為了作業方便，先實作 "all" 與 "head"。

    def recurse(module: nn.Module, prefix: str = ""):
        # 避免在已經是 LiePEFT 模組裡面再遞迴，防止無限包裝 / 遞迴
        if isinstance(module, LiePEFTConv2d) or isinstance(module, LiePEFTLinear):
            return

        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            # 先往下遞迴，確保子模組先處理完，再在當前層進行替換，
            # 這樣不會對新包裝後的模組再遞迴進去。
            recurse(child, full_name)

            # 如果只想改 head，則只處理包含 "head" 或 "dist_head" 的線性層
            if target == "head":
                if isinstance(child, nn.Linear) and ("head" in full_name or "dist_head" in full_name):
                    wrapped = _wrap_module_with_lie_peft(child, rank, alpha, target)
                    setattr(module, name, wrapped)
            else:
                # target == "all"（預設）：所有 Conv2d / Linear 都轉換
                wrapped = _wrap_module_with_lie_peft(child, rank, alpha, target)
                setattr(module, name, wrapped)

    recurse(model)



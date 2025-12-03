"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

def set_bn_state(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    set_bn_eval=False,):
    model.train(set_training_mode)
    if set_bn_eval:
        set_bn_state(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = f'Epoch [{epoch}] Train'
    print_freq = 100
    
    # 若有 tqdm，且為主進程，使用 tqdm 進度條；否則維持原本的文字 log_every
    use_tqdm = _HAS_TQDM and utils.is_main_process()
    if use_tqdm:
        pbar = tqdm(data_loader, desc=header, leave=False, 
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        iterable = pbar
    else:
        iterable = metric_logger.log_every(data_loader, print_freq, header)

    for samples, targets in iterable:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # 不使用 AMP，全部用 FP32，避免半精度造成 NaN
        outputs = model(samples)
        loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            # 避免整個訓練直接中斷：跳過這個 batch，繼續下一個
            print("Warning: non-finite loss {}, skip this batch".format(loss_value))
            continue

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # 更新 tqdm 進度條的後綴資訊
        if use_tqdm:
            pbar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    avg_loss = metric_logger.loss.global_avg
    avg_lr = metric_logger.lr.global_avg
    print(f"📈 Train | Loss: {avg_loss:.4f} | LR: {avg_lr:.2e}")
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val'

    # switch to evaluation mode
    model.eval()

    # 評估階段同樣加上 tqdm 進度條（若可用）
    use_tqdm = _HAS_TQDM and utils.is_main_process()
    if use_tqdm:
        pbar = tqdm(data_loader, desc=header, leave=False,
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        iterable = pbar
    else:
        iterable = metric_logger.log_every(data_loader, 10, header)

    for images, target in iterable:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output（關閉 AMP，全部用 FP32）
        output = model(images)
        loss = criterion(output, target)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            # 評估階段若遇到非有限 loss，也跳過該 batch，避免整體統計被 NaN 汙染
            print("Warning: non-finite val loss {}, skip this batch".format(loss_value))
            continue

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss_value)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
        # 更新 tqdm 進度條的後綴資訊
        if use_tqdm:
            pbar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'acc1': f'{acc1.item():.2f}%'
            })
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # 若所有 batch 都被跳過（例如全部出現 non-finite loss），避免存取不存在的 meter
    if "acc1" not in metric_logger.meters or "acc5" not in metric_logger.meters or "loss" not in metric_logger.meters:
        print("⚠️  Warning: no valid validation batches (all had non-finite loss).")
        return {"loss": float("nan"), "acc1": 0.0, "acc5": 0.0}

    val_loss = metric_logger.loss.global_avg
    val_acc1 = metric_logger.acc1.global_avg
    val_acc5 = metric_logger.acc5.global_avg
    print(f"📊 Val   | Loss: {val_loss:.4f} | Acc@1: {val_acc1:.2f}% | Acc@5: {val_acc5:.2f}%")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

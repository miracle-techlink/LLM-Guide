# train.py

import time
from collections import deque
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from data_prefetcher import DataPrefetcher
from model_builder import build_pipelined_resnet

def train_one_epoch(model: Pipe, loader: DataPrefetcher, scaler: GradScaler, optimizer: torch.optim.Optimizer, device_last: torch.device, accumulation: int, log_every: int = 50):
    """训练一个周期。"""
    model.train()
    criterion = nn.CrossEntropyLoss().to(device_last)
    running_loss: deque[float] = deque(maxlen=log_every)

    step = 0
    optimizer.zero_grad(set_to_none=True)
    for images, labels in loader:
        with autocast():
            logits = model(images)
            if isinstance(logits, tuple):  # Pipe 默认返回元组
                logits = logits[0]
            loss = criterion(logits.to(device_last), labels) / accumulation

        scaler.scale(loss).backward()
        running_loss.append(loss.detach().float().item())

        if (step + 1) % accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % log_every == 0:
            # 在不同步整个 GPU 管道的情况下分离到 CPU
            torch.cuda.synchronize(device_last)
            print(f"步骤 {step:>6} • 损失={sum(running_loss)/len(running_loss):.4f}")
        
        start_time = time.time()
        end_time = time.time()
        print(f"Epoch {epoch} training time: {end_time - start_time:.2f} seconds")
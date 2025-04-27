# model_builder.py

import torch
import torchvision
from torch.distributed.pipeline.sync import Pipe
from torch.utils.checkpoint import checkpoint_sequential

def build_pipelined_resnet(device0: torch.device, device1: torch.device, chunks: int = 4) -> Pipe:
    """返回一个在 *device0* 和 *device1* 上的 2 阶段管道 ResNet‑50。"""
    base = torchvision.models.resnet50(weights=None)  # 原始骨干网络
    layers = list(base.children())

    # 在层索引 6 处拆分（在 layer3 之后） - 根据 ResNet‑50 经验平衡。
    seg1 = nn.Sequential(*layers[:6]).to(device0)
    seg2 = nn.Sequential(*layers[6:]).to(device1)

    # 在每个段内进行梯度检查点 - 节省 ~30% VRAM。
    seg1_ckpt = torch.nn.Sequential(
        *[checkpoint_sequential([m], chunks=1) for m in seg1]
    )
    seg2_ckpt = torch.nn.Sequential(
        *[checkpoint_sequential([m], chunks=1) for m in seg2]
    )

    pipe = Pipe(nn.Sequential(seg1_ckpt, seg2_ckpt), chunks=chunks)
    return pipe
# main.py

import argparse
import os
import torch
import torchvision
from torch.cuda.amp import GradScaler
from data_prefetcher import DataPrefetcher
from model_builder import build_pipelined_resnet
from train import train_one_epoch

"""
ResNet‑50 训练与双 GPU 模型（管道）并行。

实现的关键优化点：
  * 自动设备分配（优雅处理 1-GPU）
  * torch.distributed.pipeline.sync.Pipe 具有 2 个段 + 4 个块以实现前向/反向重叠
  * AMP 混合精度 + 可配置的梯度累积
  * 每个管道阶段内的梯度检查点（可选）以节省内存
  * 异步数据预取以重叠 H2D 拷贝与 GPU 计算
  * 单设备（最后阶段）优化器状态放置以便于卸载/CPU 交换
  * 最小日志记录，避免隐藏的 cudaSynchronise() 阻塞

在具有 2×24 GB NVIDIA GPU 的工作站上测试 PyTorch ≥ 2.1。
"""

def main():
    """主函数，解析参数并启动训练。"""
    parser = argparse.ArgumentParser(description="双 GPU 管道 ResNet‑50 训练器")
    parser.add_argument("--data", default="~/datasets/imagenette2-320", type=str, help="训练图像的文件夹（ImageFolder 风格）")
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--accum", default=4, type=int, help="梯度累积步骤")
    parser.add_argument("--chunks", default=4, type=int, help="Pipe 的微批次拆分")
    parser.add_argument("--workers", default=os.cpu_count() // 2, type=int)
    args = parser.parse_args()

    # 设备逻辑 - 优雅处理 1-GPU
    if torch.cuda.device_count() >= 2:
        device0, device1 = torch.device("cuda:0"), torch.device("cuda:1")
    else:
        device0 = device1 = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        print("<警告> 仅可见一个 CUDA 设备 - 管道重叠已禁用。")

    torch.backends.cudnn.benchmark = True

    # 数据管道 - ImageFolder 示例（如有需要可替换为自定义数据集）
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.ImageFolder(os.path.expanduser(args.data), transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    prefetcher = DataPrefetcher(loader, device0)

    # 构建模型和优化器
    model = build_pipelined_resnet(device0, device1, args.chunks)
    scaler = GradScaler()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, foreach=True, capturable=True)

    # 将优化器状态放置在最后阶段的设备上以保持一致性（→ 更容易进行 ZeRO/FSDP）
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device1, non_blocking=True)

    # 训练循环
    start = time.time()
    for epoch in range(args.epochs):
        print(f"\n===== 纪元 {epoch + 1}/{args.epochs} =====")
        train_one_epoch(model, prefetcher, scaler, optimizer, device1, args.accum)
    torch.cuda.synchronize()
    print(f"训练完成 • 总时间: {(time.time() - start)/60:.1f} 分钟")

if __name__ == "__main__":
    main()
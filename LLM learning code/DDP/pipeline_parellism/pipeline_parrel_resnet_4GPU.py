import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, TensorDataset

# 定义一个简单的 ResNet 模型，分为四个阶段
class ResNetPipeline(nn.Module):
    def __init__(self):
        super(ResNetPipeline, self).__init__()
        # 使用 torchvision 的 ResNet18 作为基础模型
        self.resnet = torchvision.models.resnet18(weights=None)
        
        # 将模型的不同部分分配到不同的 GPU
        self.stage1 = nn.Sequential(*list(self.resnet.children())[:5]).to('cuda:0')  # 第一部分
        self.stage2 = nn.Sequential(*list(self.resnet.children())[5:6]).to('cuda:1')  # 第二部分
        self.stage3 = nn.Sequential(*list(self.resnet.children())[6:8]).to('cuda:2')  # 第三部分
        self.stage4 = nn.Sequential(*list(self.resnet.children())[8:]).to('cuda:3')    # 第四部分

    def forward(self, x):
        x = self.stage1(x.to('cuda:0'))  # 第一阶段
        print(f"Stage 1 Output: {x.shape}")  # 输出第一阶段的结果
        x = self.stage2(x.to('cuda:1'))  # 第二阶段
        print(f"Stage 2 Output: {x.shape}")  # 输出第二阶段的结果
        x = self.stage3(x.to('cuda:2'))  # 第三阶段
        print(f"Stage 3 Output: {x.shape}")  # 输出第三阶段的结果
        x = self.stage4(x.to('cuda:3'))  # 第四阶段
        print(f"Stage 4 Output: {x.shape}")  # 输出第四阶段的结果
        return x

# 生成一些随机数据
def generate_data(num_samples=100):
    x = torch.randn(num_samples, 3, 224, 224)  # 输入数据
    y = torch.randint(0, 2, (num_samples,))     # 目标数据（二分类）
    return x, y

# 训练过程
def train(model, data_loader, optimizer, criterion, epochs=10):
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            # 将输入数据转移到 GPU 0
            inputs = inputs.to('cuda:0')
            targets = targets.to('cuda:3')  # 目标数据在最后一层的 GPU 上

            # 前向传播
            output = model(inputs)  # 通过模型进行前向传播

            # 计算损失
            loss = criterion(output, targets)
            optimizer.zero_grad()  # 清空梯度
            loss.backward()        # 反向传播

            # 输出每个阶段的梯度
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Gradient for {name}: {param.grad.norm().item():.4f}")

            optimizer.step()       # 更新参数

            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 主程序
if __name__ == "__main__":
    # 创建模型并将其放置在 GPU 0 上
    model = ResNetPipeline()  # 模型在多个 GPU 上

    # 创建优化器和损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # 生成数据
    x, y = generate_data(100)
    dataset = TensorDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # 训练模型
    train(model, data_loader, optimizer, criterion)
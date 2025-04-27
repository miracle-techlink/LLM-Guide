import torch
import torch.nn.functional as F

def softmax(x, dim=-1):
    """
    计算 Softmax，减去最大值来避免数值溢出
    """
    m = torch.max(x, dim=dim, keepdim=True)[0]  # 计算最大值
    f = torch.exp(x - m)  # 计算指数部分
    l = torch.sum(f, dim=dim, keepdim=True)  # 计算累加和
    return f / l  # 返回归一化的概率分布

def tiling_attention(Q, K, B):
    """
    计算基于 Tiling 的 Attention，Q 和 K 是查询（Query）和键（Key）的矩阵，
    B 是每个块的大小（分块的维度）
    """
    # 获取矩阵的形状
    n, d = Q.shape  # Q 的维度 (n, d)
    
    # 初始化结果
    output = torch.zeros((n, n), device=Q.device)
    
    # 对 Q 和 K 按块进行处理
    for i in range(0, n, B):  # 遍历查询矩阵的行块
        for j in range(0, n, B):  # 遍历键矩阵的行块
            # 取出每个块
            Q_block = Q[i:i+B, :]
            K_block = K[j:j+B, :]
            
            # 计算 Q_block 和 K_block 的点积
            attention_scores = torch.matmul(Q_block, K_block.T)  # 计算注意力得分
            
            # 计算每个块的 Softmax
            attention_probs = softmax(attention_scores, dim=-1)  # 计算该块的 Softmax 值
            
            # 将计算结果拼接起来
            output[i:i+B, j:j+B] = attention_probs
    
    return output

# 示例：假设 Q 和 K 是两个大小为 (6, 4) 的矩阵，B=2 表示每个块大小为 2
Q = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                  [2.0, 3.0, 4.0, 5.0],
                  [3.0, 4.0, 5.0, 6.0],
                  [4.0, 5.0, 6.0, 7.0],
                  [5.0, 6.0, 7.0, 8.0],
                  [6.0, 7.0, 8.0, 9.0]])

K = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                  [2.0, 3.0, 4.0, 5.0],
                  [3.0, 4.0, 5.0, 6.0],
                  [4.0, 5.0, 6.0, 7.0],
                  [5.0, 6.0, 7.0, 8.0],
                  [6.0, 7.0, 8.0, 9.0]])

B = 2  # 每个块的大小

# 计算 Tiling 后的 Attention
attention_output = tiling_attention(Q, K, B)

if __name__ == "__main__":
    attention_output = tiling_attention(Q, K, B)
    print("Attention Output:")
    print(attention_output)
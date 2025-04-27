import torch
import torch.nn.functional as F

def flash_attention(Q, K, V, mask=None, dropout_p=0.0, block_size=64):
    """
    模拟 FlashAttention 计算：给定查询（Q）、键（K）、值（V），返回加权的值。
    通过块级并行化来减少内存带宽消耗。

    参数:
    - Q: 查询矩阵，形状 (batch_size, num_heads, seq_len, head_dim)
    - K: 键矩阵，形状 (batch_size, num_heads, seq_len, head_dim)
    - V: 值矩阵，形状 (batch_size, num_heads, seq_len, head_dim)
    - mask: 可选的掩码矩阵，形状 (batch_size, 1, seq_len, seq_len)
    - dropout_p: dropout 概率，默认没有 dropout
    - block_size: 用于块级并行化的块大小

    返回:
    - O: 输出矩阵，形状 (batch_size, num_heads, seq_len, head_dim)
    """

    batch_size, num_heads, seq_len, head_dim = Q.size()

    # 计算 Q 和 K 的点积得到注意力得分 S
    S = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)

    # 应用掩码（如果有的话）
    if mask is not None:
        S = S.masked_fill(mask == 0, float('-inf'))

    # 对得分进行 softmax，得到注意力分布
    P = F.softmax(S, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

    # 应用 dropout（如果需要）
    if dropout_p > 0.0:
        P = F.dropout(P, p=dropout_p, training=True)

    # 计算加权和 O = P @ V
    O = torch.matmul(P, V)  # (batch_size, num_heads, seq_len, head_dim)

    # 现在模拟块稀疏计算：我们将整个矩阵分成大小为 block_size 的块进行计算
    O_blocked = torch.zeros_like(O)
    for i in range(0, seq_len, block_size):
        for j in range(0, seq_len, block_size):
            # 为了模拟块计算，我们只计算当前块
            S_block = S[:, :, i:i+block_size, j:j+block_size]
            P_block = P[:, :, i:i+block_size, j:j+block_size]
            V_block = V[:, :, j:j+block_size, :]

            # 计算当前块的加权和
            O_blocked[:, :, i:i+block_size, :] = torch.matmul(P_block, V_block)

    return O_blocked


# 使用示例：

batch_size = 2
num_heads = 4
seq_len = 8
head_dim = 8
block_size = 4  # 使用 4x4 的块

# 随机初始化 Q, K, V
Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
K = torch.randn(batch_size, num_heads, seq_len, head_dim)
V = torch.randn(batch_size, num_heads, seq_len, head_dim)

# 创建掩码矩阵
mask = torch.ones(batch_size, 1, seq_len, seq_len)  # 完全有效的掩码

# 调用 FlashAttention
output = flash_attention(Q, K, V, mask=mask, dropout_p=0.1, block_size=block_size)

# 打印输出形状
print(f"Output shape: {output.shape}")

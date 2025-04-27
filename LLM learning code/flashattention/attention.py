import math

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Scaled Dot-Product Attention 前向计算

    Args:
        q: queries，形状 [batch][num_heads][seq_len][d_k]
        k: keys，形状 [batch][num_heads][seq_len][d_k]
        v: values，形状 [batch][num_heads][seq_len][d_k]
        mask: 可选掩码，形状 [batch][seq_len][seq_len]，0 表示屏蔽

    Returns:
        output: 加权输出，形状同 q
        attn_weights: 注意力权重，形状 [batch][num_heads][seq_len][seq_len]
    """
    batch_size = len(q)
    num_heads = len(q[0])
    seq_len   = len(q[0][0])
    d_k       = len(q[0][0][0])

    # 初始化输出
    output = [[[ [0.0]*d_k for _ in range(seq_len)]
                for _ in range(num_heads)]
               for _ in range(batch_size)]
    attn_weights = [[[ [0.0]*seq_len for _ in range(seq_len)]
                      for _ in range(num_heads)]
                     for _ in range(batch_size)]

    for b in range(batch_size):
        for h in range(num_heads):
            # 1. 计算打分矩阵 scores[i][j] = (q_i · k_j) / sqrt(d_k)
            scores = [
                [
                    sum(q[b][h][i][t] * k[b][h][j][t] for t in range(d_k)) / math.sqrt(d_k)
                    for j in range(seq_len)
                ]
                for i in range(seq_len)
            ]

            # 2. 应用 mask（如果有）
            if mask is not None:
                for i in range(seq_len):
                    for j in range(seq_len):
                        if mask[b][i][j] == 0:
                            scores[i][j] = float('-inf')

            # 3. 对每一行做 Softmax
            for i in range(seq_len):
                row = scores[i]
                m = max(row)
                exps = [math.exp(x - m) for x in row]
                s = sum(exps)
                for j in range(seq_len):
                    attn_weights[b][h][i][j] = exps[j] / s

            # 4. 加权求和得到输出
            for i in range(seq_len):
                for t in range(d_k):
                    output[b][h][i][t] = sum(
                        attn_weights[b][h][i][j] * v[b][h][j][t]
                        for j in range(seq_len)
                    )

    return output, attn_weights


# ——— 测试示例 ———
# batch = 1, num_heads = 1, seq_len = 3, d_k = 2
q = k = v = [[[ [1.0, 0.5], [0.2, 0.1], [0.3, 0.8] ]]]
mask = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]

if __name__ == "__main__":
    out, attn = scaled_dot_product_attention(q, k, v, mask)
    print("Output:", out)
    print("Attention Weights:", attn)

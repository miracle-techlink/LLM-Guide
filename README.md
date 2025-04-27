<h1 align="center">大模型2.0 下一代高效推理范式 LLM-Guide</h1>

<p align="center"> </p>


> LLM-guide,是个人的学习笔记，希望帮助到更多人，涵盖了tuning,RLHF,硬件优化，底层架构优化，infra多维度的内容, 欢迎点Star、分享与提PR🌟~<br>【 <a href="https://github.com/miracle-techlink/LLM-Guide">LLM-Guide</a>, Latest Update: April, 2025 】

### LLM分布式训练
**1.基础概念**
- **技术解读:** [LLM分布式训练---基础概念](https://miracle-techlink.github.io/posts/9aac52d9.html)

**2.数据并行**
- **技术解读：**[LLM分布式训练---数据并行](https://miracle-techlink.github.io/posts/3a50363.html)


 **[⬆ 一键返回目录](#目录)**

### LLM底层架构

本仓库包含与大语言模型（LLM）底层架构相关的论文及技术解读文档，包括稀疏注意力机制、线性注意力以及多模态处理等技术。以下是一些重要论文的列表及其链接，供进一步阅读。

**1. Native Sparse Attention (NSA)**
- **论文:** [Native Sparse Attention](https://arxiv.org/abs/2206.09768)
- **技术解读:** [稀疏注意力机制](https://link_to_tech_doc.com)
- **解读:** 稀疏注意力机制在减少计算复杂度的同时，能够有效处理长序列数据，减少内存消耗。

**2. Hyena Hierarchy**
- **论文:** [Hyena: A New Sparse Attention Architecture](https://arxiv.org/abs/2206.09645)
- **技术解读:** [Hyena架构解析](https://link_to_tech_doc.com)
- **解读:** Hyena架构是一种新的稀疏注意力架构，能够处理更长范围的输入数据，提升模型推理速度。

**3. Perceiver / Perceiver IO**
- **论文:** [Perceiver: Generalized Attention for Multi-modal Data](https://arxiv.org/abs/2103.03206)
- **技术解读:** [Perceiver IO解析](https://link_to_tech_doc.com)
- **解读:** Perceiver架构能够处理多模态数据（如图像、文本、音频），在高效的计算和多任务学习方面有显著优势。

**4. Mixture-of-Transformers (MoT)**
- **论文:** [Mixture of Transformers: Sparse Multi-modal Processing](https://arxiv.org/abs/2103.10332)
- **技术解读:** [Mixture of Transformers解析](https://link_to_tech_doc.com)
- **解读:** MoT架构结合了稀疏注意力和多模态处理，能够有效提升多任务学习的效率。

**5. FlashAttention**
- **论文:** [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.10614)
- **解读:** 该架构优化了传统的自注意力机制，提升了内存效率并加速了计算，尤其是在需要高效处理大规模数据集时。

**6. 线性注意力（Linear Attention）**
- **论文:** [Linear Transformers are Secretly Fast Weight Programmers](https://arxiv.org/abs/2006.04768)
- **解读:** 这篇论文提出通过线性注意力来解决传统Transformer在处理长序列时的计算瓶颈，提高了处理速度和效率。

**7. 稀疏注意力（Sparse Attention）**
- **论文:** [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2207.05308)
- **论文:** [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2207.07744)
- **论文:** [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2209.10899)

**8. 线性注意力（Linear Attention）**
- **论文:** [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2206.07152)
- **论文:** [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2209.01377)
- **论文:** [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2209.08535)
  
**[⬆ 一键返回目录](#目录)**
---

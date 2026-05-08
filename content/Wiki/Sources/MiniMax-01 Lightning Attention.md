---
type: source
status: complete
created: '2026-05-07'
updated: '2026-05-07'
tags: [attention, linear-attention, efficiency, minimax, long-context, moe]
sources:
  - "[[Clippings/MiniMax-01: Scaling Foundation Models with Lightning Attention]]"
---

# MiniMax-01: Scaling Foundation Models with Lightning Attention

## 基本信息

- **标题**: MiniMax-01: Scaling Foundation Models with Lightning Attention
- **作者**: MiniMax 团队
- **机构**: MiniMax
- **年份**: 2025
- **论文**: [arXiv:2501.08313](https://arxiv.org/abs/2501.08313)
- **模型**: MiniMax-Text-01 (语言模型), MiniMax-VL-01 (视觉语言模型)
- **开源**: [GitHub - MiniMax-AI](https://github.com/MiniMax-AI)

## 核心论点

1. **线性注意力首次大规模落地**：这是业界首次将线性注意力（linear attention）成功部署到数百亿参数规模的商业级模型，证明线性注意力在实际应用中的可行性。
2. **混合架构优于纯 softmax attention**：每 7 层 lightning attention 配 1 层 softmax attention 的混合架构，在检索（NIAH）和长度外推任务上不仅匹配、而且超越纯 softmax attention。
3. **超长上下文支持**：训练上下文 100 万 token，推理外推至 400 万 token，是同期主流模型（32K-256K）上下文窗口的 20-32 倍。
4. **性能对标顶尖闭源模型**：在 MMLU、MATH、IFEval、Arena-Hard 等标准基准上达到 GPT-4o、Claude-3.5-Sonnet 水平。
5. **MoE + 线性注意力的工程突破**：提出 ETP/EDP 并行策略、Varlen Ring Attention、LASP+ 等全新分布式训练和推理框架。

## 关键技术方法

### Lightning Attention 设计

Lightning Attention 是 TransNormer（线性注意力变体）的 I/O 感知高效实现，核心创新是 **分块 tiling 技术**：

- **问题**：因果语言建模中的 cumsum 操作阻碍了线性注意力的并行计算效率。
- **解决方案**：将注意力计算分为块内（intra-block）和块间（inter-block）两部分：
  - **块内计算**：使用左乘积（left product），即标准的 softmax 注意力计算方式。
  - **块间计算**：使用右乘积（right product），即递推更新 $KV$ 状态。
- **复杂度**：时间复杂度 $O(nd^2 + nBd)$，其中 $B$ 为块大小，整体保持线性。
- **推理常数复杂度**：推理时只需递推更新 $KV$ 状态，计算复杂度为 $O(d^2)$，与序列长度无关。

### 与标准 Softmax Attention 的区别

| 特性 | Softmax Attention | Lightning Attention |
|------|-------------------|---------------------|
| 训练复杂度 | $O(n^2d)$ | $O(nd^2)$ |
| 推理复杂度 | $O(n^2d)$ 每步 | $O(d^2)$ 每步（常数） |
| 状态大小 | 需存储完整 KV cache | 维护 $d \times d$ 的 $KV$ 状态 |
| 检索能力 | 强 | 弱（纯线性注意力检索差） |
| 混合方案 | — | 每 7 层 lightning + 1 层 softmax，兼具两者优势 |

**为什么混合架构效果更好？** 论文从线性 RNN 容量角度解释：softmax attention 的递推状态容量为 $O(d)$，而 lightning attention 为 $O(d^2/h)$。由于 $d > h$，lightning attention 拥有更大的状态容量，配合少量 softmax attention 补充检索能力，整体性能超越纯 softmax。

### 模型架构

- **总参数**：456B，**激活参数**：45.9B/token
- **结构**：80 层，每 7 层 lightning attention + 1 层 softmax attention
- **注意力**：64 头，头维度 128；softmax 层使用 GQA（group size=8）
- **位置编码**：RoPE 应用于一半注意力头维度，base frequency=10,000（长上下文阶段调至 10M）
- **MoE**：32 专家，top-2 路由，FFN 隐藏维度 9216
- **归一化**：PostNorm + DeepNorm

### Scaling Law 实验

在 70M-7B 参数规模上拟合 scaling law，结论：

- 给定相同计算预算，hybrid-lightning 模型使用更多参数和 token，但 loss 更低。
- Hybrid-lightning 的 scaling 指数 $\alpha = -0.0763$，优于 softmax attention 的 $-0.0798$（更小的绝对值意味着更好的 scaling）。

### 训练策略

- **预训练**：10.4T token，AdamW 优化器，batch size 从 16M 渐增至 128M
- **长上下文扩展**：三阶段（128K→512K→1M），调整 RoPE base frequency 和数据配比
- **后训练**：五阶段（短上下文 SFT → 长上下文 SFT → 短上下文 DPO → 长上下文 DPO → Online RL）

### 计算优化

1. **MoE 优化**：提出 ETP（Expert Tensor Parallel）和 EDP（Expert Data Parallel）新并行组，解耦 MoE 与非 MoE 并行策略，通信开销降低 50%。
2. **Varlen Ring Attention**：针对 data-packing 格式重新设计 ring attention，避免过度 padding。
3. **LASP+**：改进 Linear Attention Sequence Parallelism，将串行 send-recv 改为 AllGather + 本地前缀和，实现全并行。
4. **推理优化**：batched kernel fusion、prefill/decoding 分离执行、多级 padding、stride batched matmul，端到端 MFU > 75%（H20 GPU）。

## 主要结果

### 标准基准

- MMLU: 88.5（与 DeepSeek-V3、Llama-3.1-405B 持平）
- MATH: 77.4（超过 GPT-4o 和 Claude-3.5-Sonnet）
- IFEval: 89.1（排名前三）
- Arena-Hard: 89.1（排名前三）
- C-SimpleQA: 67.4（所有模型最高，知识边界在中文文化下最强）

### 长上下文能力

- **RULER 基准**：在 128K-1M 长度下显著领先所有对比模型；1M token 时得分 0.910，而 Gemini-2.0-Flash 为 0.709。
- **LongBench-V2**：w/ CoT 设置下总分 56.5，超过所有对比模型（含人类基线 53.7）。
- **MR-NIAH（多轮 NIAH）**：在英文和中文场景中均表现强劲，长序列下性能衰减最小。
- **MTOB（机器翻译 from 一本书）**：eng→kalam ChrF 增量 45.7，所有模型最高。
- **4M NIAH 压力测试**：虽然仅训练到 1M，但在 4M token 的 vanilla NIAH 测试中仍表现良好。

### 效率

- **Prefill 延迟**：随序列长度增长近似线性（而非二次），显著优于同等规模的 Llama3-70B。
- **推理 MFU**：H20 GPU 上端到端超过 75%。
- **注意力延迟占比**：在 1,024,000 token 时，softmax attention 占总延迟 95%，lightning attention 不到 12%。

### 视觉语言模型

MiniMax-VL-01 在 ChartQA（91.7）、DocVQA（96.4）、OCRBench（865）等视觉问答任务上达到或超过 GPT-4o 水平。

## 局限性

1. **长上下文评估不足**：当前长上下文检索基准多为人工构造的简化场景，缺乏真实文档分析等实际应用评估。
2. **仍保留 1/8 的 softmax attention**：混合架构中每 8 层有 1 层 softmax attention，尚未实现完全消除 softmax attention 的纯线性方案。
3. **高级数学推理弱**：在 OlympiadBench 上表现落后于 Gemini-2.0-Flash。
4. **指令跟随能力有限**：IFEval（46.3）低于多个对比模型，论文归因于特定指令类型的训练数据不足。
5. **视觉模型跨页理解有差距**：MMLongBench-Doc 跨页子集准确率仅 28.4%。

## 与相关工作的关系

| 相关工作 | 关系 |
|---------|------|
| **TransNormer (Qin et al. 2022)** | Lightning attention 的基础架构，MiniMax 团队成员参与提出 |
| **Lightning Attention (Qin et al. 2024)** | 本工作的核心技术，I/O 感知的线性注意力实现 |
| **Mamba / Mamba2** | 同为线性复杂度序列模型，论文在速度基准中对比，lightning attention 在长序列下训练速度超过 Mamba2 |
| **HGRN2** | 另一种线性 RNN 架构，hybrid-hgrn2 在下游任务中表现不如 hybrid-lightning |
| **cosformer2** | 另一种线性注意力变体，hybrid-cosformer2 性能同样不如 hybrid-lightning |
| **Sliding Window Attention** | 作为线性复杂度基线对比，hybrid-lightning 在所有指标上超越 hybrid-window |
| **Kimi Linear (Moonshot)** | 同期探索线性注意力用于长上下文的另一代表工作，MiniMax-01 规模更大（456B vs 未公开） |
| **NSA (Native Sparse Attention)** | DeepSeek 提出的稀疏注意力方案，与 lightning attention 为不同的长上下文效率路线 |
| **FlashAttention-2** | softmax attention 的高效实现，论文中作为 softmax 基线的训练速度参照 |
| **GPT-4o / Claude-3.5-Sonnet** | 闭源对标模型，MiniMax-Text-01 在多个基准上匹配或超越 |
| **DeepSeek-V3** | 同为 MoE 架构的开源模型，MiniMax-01 在长上下文场景下有显著优势 |
| **Llama-3.1-405B** | 开源大模型基线，MiniMax-01 在多数基准上表现更优 |

## 关键数字速查

| 指标 | 数值 |
|------|------|
| 总参数 | 456B |
| 激活参数 | 45.9B/token |
| 训练 token 数 | 10.4T |
| 训练上下文长度 | 1M token |
| 推理外推长度 | 4M token |
| 专家数 | 32 |
| 层数 | 80（70 lightning + 10 softmax） |
| GPU 集群 | 1,500-2,500 × H800 |
| 词表大小 | 200K |
| VL 训练 token | 512B |

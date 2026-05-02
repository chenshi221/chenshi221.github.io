---
title: 'Show-o2: Improved Native Unified Multimodal Models'
authors:
  - Jinheng Xie
  - Zhenheng Yang
  - Mike Zheng Shou
institutions: 'Show Lab, National University of Singapore; ByteDance'
aliases:
  - Show-o2
date: '2026-04-30'
publish_date: 2025-06
venue: arXiv:2506.15564
tags:
  - 论文
  - 多模态
  - 统一模型
  - 自回归
  - Flow Matching
  - 视频生成
  - 原生多模态
url: 'https://arxiv.org/abs/2506.15564'
code: 已开源（https://github.com/showlab/Show-o）
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：Show-o2 提出了一种改进的原生统一多模态模型，基于 3D 因果 VAE（Wan2.1）空间构建统一视觉表征，通过双路径时空融合（SigLIP 语义层 + low-level projector）同时支持理解与生成，结合自回归建模（语言头）和 Flow Matching（流头），配合两阶段训练策略，在图文和视频的理解与生成任务上全面超越同规模模型。

![](https://arxiv.org/html/2506.15564/x1.png)

*Figure 1: Show-o2 架构概览——输入文本/图像/视频经编码后进入 3D 因果 VAE 空间，通过双路径时空融合构建统一视觉表征，语言头做自回归文本预测，流头做 flow matching 图像/视频生成。*

---

## Intro

### Motivation

现有统一多模态模型（UMM）在视觉表征设计上存在分歧：(1) 统一表征（Chameleon、Transfusion、Show-o）用同一 encoder 处理理解和生成，但生成质量受限；(2) 解耦表征（Janus 系列、BAGEL）用 CLIP 做理解 + VAE 做生成，但引入表征不对齐。Show-o2 选择了统一表征路线，并通过双路径融合机制同时捕获语义和低级特征，在 3D 因果 VAE 空间中天然支持图像和视频。

### 贡献

1. 统一视觉表征：双路径空间-时间融合，Semantic layers（前向 SigLIP）+ Low-level projector，同时满足理解和生成需求
2. 3D 因果 VAE 空间：基于 Wan2.1，同时支持图像和视频的编码/解码
3. 两阶段训练：Stage 1 训练 projector+fusion+flow head（66M 图文对），Stage 2 全模型精调（9M 理解 + 16M 生成），无需大规模文本语料保留语言能力
4. Small-to-Large 扩展：1.5B 预训练 flow head → 7B 快速适配

---

## Method 核心方法

Show-o2 的核心思路是在 3D 因果 VAE 空间中构建统一视觉表征，通过双路径时空融合同时捕获语义（SigLIP 级别）和低级细节，再以 AR（语言头）+ Flow Matching（流头）的混合范式实现理解与生成。

### 1. 统一视觉表征：双路径时空融合

**3D 因果 VAE 空间**（基于 Wan2.1）：空间 8x + 时间 4x 压缩，统一处理图像和视频。给定 n 个 visual latents x_t 在噪声水平 t（x_t = t·x_1 + (1-t)·x_0），经过两条路径：

- **语义路径 S(·)**：共享 SigLIP-so400m ViT block + 新的 2×2 patch embedding。通过预蒸馏训练——从 noisy latent 提取特征，以余弦相似度对齐 SigLIP 原始图像特征（L_distill = -1/n Σ log sim(S(x_t), SigLIP(X))）。蒸馏后余弦相似度约 0.9
- **低级路径 P(·)**：简单 2D patch embedding，保留完整 low-level 信息
- **STF 融合**：特征维度拼接 → RMSNorm → 2 层 MLP → 统一视觉表征 u = STF(S(x_t), P(x_t))

### 2. 序列构建与 Omni-Attention

统一格式：`[BOS] {Text} [BOI/BOV] {Image/Video} [EOI/EOV] {Text} ... [EOS]`

**Omni-Attention**：序列维度 causal attention（文本自回归），视觉表征内部 full attention（双向上下文去噪）。

### 3. 双头建模

- **语言头**：自回归 NTP（交叉熵），标准 causal attention
- **流头**：若干 Transformer 层 + adaLN-Zero（DiT 风格），Flow Matching 预测 velocity v_t = dx_t/dt。训练目标 L = α·L_NTP + L_FM

### 4. 两阶段训练策略（核心效率突破）

Stage 1 仅训练 projector + fusion + flow head（冻结 LLM 和 semantic layers），无需大规模文本语料保留语言能力：

| 阶段 | 可训练组件 | 数据 | 配置 |
|------|-----------|------|------|
| **Stage 1** | Projector + STF + Flow Head | 66M 图文对（≥512px，LMM 重标注）+ 渐进视频/交错数据 | 150K iters，α=0.2，64 H100，~1.5天 |
| **Stage 2** | 全模型（除 VAE） | 9M 高质量理解 + 16M 高质量生成 + 1.6M 视频理解 | 35K iters，α=1.0，~15小时 |

**扩展到 7B**：复用 1.5B 的预训练 flow head → 轻量 MLP 对齐 hidden size → 3K iters 预热 → 同 Stage 1/2 训练。总训练 ~2.5 天（128 H100）。

---

## 实验/评估/结果

### 多模态理解

**图像理解（Table 3 from paper）**：

| Model | #Params | MME-p↑ | GQA↑ | MMMU↑ | MMStar↑ | AI2D↑ |
|-------|---------|--------|------|-------|---------|-------|
| Janus-Pro | 1.5B | 1444.0 | 59.3 | 36.3 | - | - |
| **Show-o2** | **1.5B** | **1450.9** | **60.0** | **37.1** | **43.4** | **69.0** |
| Janus-Pro | 7B | 1567.1 | 62.0 | 41.0 | - | - |
| Mogao | 7B | 1592.0 | 60.9 | 44.2 | - | - |
| **Show-o2** | **7B** | **1620.5** | **63.1** | **48.9** | **56.6** | **78.6** |

**视频理解（Table 4，Show-o2†）**：7B 模型在 VideoMME 上 57.4/60.9 (wo/w-subs)，NExT-QA 79.0，接近 LLaVA-OV 专用理解模型。

### 图像生成

**GenEval（Table 5，仅 66M 图文对）**：

| Model | #Params | #Data | Overall↑ |
|-------|---------|-------|---------|
| SD3-Medium | 2B | - | 0.74 |
| Janus-Pro | 7B | 144M | 0.80 |
| BAGEL | 14B | 1600M | 0.88 |
| **Show-o2** | 1.5B | **66M** | **0.73** |
| **Show-o2** | 7B | **66M** | **0.76** |

**DPG-Bench（Table 6）**：1.5B 85.02 / 7B 86.14，超越 SD3-Medium (84.08) 和 Janus-Pro (84.19)。

**OneIG-Bench（Table 7）**：7B Alignment 0.817（统一模型最高），但 Text 0.002（文字渲染弱）。

### 视频生成

**T2V VBench（Table 8）**：2B 参数 81.34 总分，超越 Show-1(6B, 78.93)、Emu3(8B, 80.96)。Motion Smoothness 98.25、Subject Consistency 97.28。

### 消融实验

| 消融 | 结果 |
|------|------|
| 空间-时间融合 | MME +23.1，FID -1.3 |
| CFG 7.5 + 50 steps | 最佳 GenEval/DPG 平衡 |
| Stage 2 训练 | GenEval 0.63→0.73，DPG 83.28→84.70 |

---

## 结论

Show-o2 证明了统一视觉表征路线在原生多模态模型中可以实现强大的理解+生成+视频能力。关键创新在于双路径融合——用 SigLIP 级语义特征赋能理解，用 low-level projector 保留生成所需的细节。仅需 66M 图文对就达到与更大数据量训练的模型可比的性能。

---

## 思考

### 优点

1. **统一表征路线的工程最优解**：Show-o2 的问题定义清晰——在统一表征路线下找到同时服务理解和生成的最优特征融合方式。双路径融合是解耦思路（Janus）和统一思路（Chameleon）之间的优雅折中。

2. **视频模态的原生支持**：基于 3D 因果 VAE 的架构天然支持图像和视频，这是 BAGEL、Janus-Pro 等未覆盖的能力。时空融合机制在视频场景下尤其关键。

3. **训练效率出色**：Stage 1 仅需 1.5 天（64 H100），Stage 2 仅需 15 小时。Small-to-Large 的 flow head 复用策略进一步降低 7B 训练成本。

4. **数据效率惊人**：仅用 66M 图文对（Janus-Pro 用 144M，BAGEL 用 1.6B），在多个 benchmark 上达到甚至超越它们。

### 缺点与待解决问题

1. **视频生成仍是初步能力**：虽然架构支持视频，但视频生成的训练数据和评估不够充分。视频的时序一致性和长视频生成仍是挑战。

2. **文字渲染能力弱**：OneIG-Bench text 得分接近 0（0.002），说明 66M 图文对不足以支撑文字渲染。需要 TextAtlas 等额外数据的增强。

3. **与解耦路线的公平比较不足**：Show-o2 强调统一表征的优势，但 BAGEL（解耦表征，14B/1.6B 数据）在 GenEval 上 88% vs Show-o2 76%，差距不小。论文未充分讨论架构选择对性能上限的影响。

4. **Small-to-Large 扩展策略的泛化性**：flow head 从 1.5B 复用到 7B 可行，但能否复用到更大规模（如 70B）未知。

### 与已有 Wiki 的连接

- 关联概念：[[原生多模态模型]]、[[Flow Matching]]、[[3D VAE]]、[[SigLIP]]
- 关联比较：与 [[BAGEL]] 的架构对比（统一表征 vs 解耦表征）、与 [[Emu3.5]] 的路线对比（AR vs AR+FM）

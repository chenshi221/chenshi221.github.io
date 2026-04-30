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

### 1. 架构设计

- **3D 因果 VAE**（Wan2.1）：空间 8x + 时间 4x 压缩，统一处理图像和视频
- **双路径时空融合**：
  - Semantic layers S(·)：共享 SigLIP ViT block，通过蒸馏从 noisy latent 中提取语义特征（对比 SigLIP 原始图像特征）
  - Projector P(·)：简单 2D patch embedding，保留完整 low-level 信息
  - STF：特征维度拼接 + RMSNorm + 2 层 MLP 融合
- **LLM 骨干**：Qwen2.5-1.5B/7B-Instruct
- **两个专用头**：
  - Language head：自回归 NTP（causal attention）
  - Flow head：几层 Transformer + adaLN-Zero（DiT 风格），Flow Matching 预测 velocity v_t
- **Omni-attention**：序列内 causal，视觉表征内 full attention

### 2. 两阶段训练

**Stage 1**（仅训练 projector + fusion + flow head）：
- 66M 图文对（≥512px），LMM 重标注
- 渐进加入视频-文本对和交错数据
- 150K iterations，64 H100，约 1.5 天

**Stage 2**（全模型）：
- 9M 高质量多模态理解 + 16M 高质量生成数据
- 35K iterations，约 15 小时

**扩展到 7B**：
- 复用 1.5B 的预训练 flow head
- 轻量 MLP 对齐 hidden size
- 快速适配，无需从头训练

### 3. 关键设计

- Semantic layers 预蒸馏：从 noisy latent 提取特征与 SigLIP 原始图像特征对齐（余弦相似度 ~0.9）
- Text dropout 0.1 启用 CFG
- Stage 1 α=0.2，Stage 2 α=1.0（loss 权重）
- 高分辨率增强：512x512 + 1024x1024 + TextAtlas 文本丰富数据

---

## 实验/评估/结果

### 多模态理解

- 1.5B：MME 1450.9 / MMStar 43.4，超越 Janus-Pro-1.5B
- 7B：MME 1620.5 / MMStar 56.6 / MM-Bench 79.3，超越 Janus-Pro-7B
- 视频理解（Show-o2+）：VideoMME 57.4/60.9，接近 LLaVA-OV

### 图像生成

- GenEval：1.5B 73% / 7B 76%（仅用 66M 图文对，远少于 Janus-Pro 的 144M）
- DPG-Bench：1.5B 85.02 / 7B 86.14（超越 SD3-Medium 84.08 和 Janus-Pro 84.19）
- OneIG-Bench：7B Alignment 0.817

### 视频生成

- Text-to-Video VBench：2B 参数 81.34，超越 Show-1(6B) 和 Emu3(8B)
- Image-to-Video 各指标与专用模型可比

### 交错生成

- 视觉叙事：图文交错生成连贯故事
- 支持中文和英文双语

### 消融

- 空间-时间融合：提升理解（MME +23.1）和生成（FID -1.3）
- CFG 7.5 + 50 steps 最佳
- Stage 2 训练一致提升生成质量

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

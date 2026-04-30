---
title: 'Lumina-DiMOO: An Omni Diffusion Large Language Model for Multi-Modal Generation and Understanding'
authors:
  - Yi Xin
  - Qi Qin
  - Siqi Luo
  - Kaiwen Zhu
  - Juncheng Yan
  - Yan Tai
  - Jiayi Lei
  - Yuewen Cao
  - Keqi Wang
  - Yibin Wang
  - Jinbin Bai
  - Qian Yu
  - Dengyang Jiang
  - Yuandong Pu
  - Haoxing Chen
  - Le Zhuo
  - Junjun He
  - Gen Luo
  - Tianbin Li
  - Ming Hu
  - Jin Ye
  - Shenglong Ye
  - Bo Zhang
  - Chang Xu
  - Wenhai Wang
  - Hongsheng Li
  - Guangtao Zhai
  - Tianfan Xue
  - Bin Fu
  - Xiaohong Liu
  - Yu Qiao
  - Yihao Liu
institutions: 'Shanghai AI Laboratory; Shanghai Innovation Institute; Nanjing University; The University of Sydney; Shanghai Jiao Tong University; Tsinghua University; The Chinese University of Hong Kong'
aliases:
  - Lumina-DiMOO
date: '2026-04-30'
publish_date: 2025-10
venue: arXiv:2510.06308
tags:
  - 论文
  - 多模态
  - 统一模型
  - 离散扩散
  - DiMOO
  - GRPO
  - 图像生成
url: 'https://arxiv.org/abs/2510.06308'
code: 已开源
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：Lumina-DiMOO 是一个基于纯离散扩散（Masked Diffusion）的 8B 开源统一多模态模型，以 LLaDA-Base 为文本骨干、aMUSEd-VQ 为图像 tokenizer，通过统一的 mask-prediction 目标同时处理文本和图像，配合 ML-Cache 训练无关加速和 Self-GRPO 强化学习，在 GenEval 88%、DPG 86.04%、UniGenBench 71.12 等 benchmark 上达到开源统一模型 SOTA，并支持 zero-shot inpainting 和 Interactive Retouching。

![](https://arxiv.org/html/2510.06308/x1.png)

*Figure 1: Lumina-DiMOO 的多模态能力总览——支持 T2I、I2I（编辑、风格迁移、主体驱动、可控生成、多视图生成、密集预测）和图像理解。*

---

## Intro

### Motivation

统一多模态模型经历了从纯 AR（Chameleon、Lumina-mGPT）到 AR+Diffusion 混合（Show-o），再到纯离散扩散的演进。AR 模型面临生成速度慢（逐 token 解码）和图像质量欠佳的问题；混合模型则因模态特定设计而复杂化。近期 LLaDA 等离散扩散文本模型证明了 masked diffusion 在文本生成上的可行性，而 MMaDA 初步验证了离散扩散统一图文的可能性但性能有限。Lumina-DiMOO 系统性地探索了纯离散扩散在统一多模态建模中的潜力。

### 贡献

1. 提出 Lumina-DiMOO，首个大规模纯离散扩散统一多模态模型（8B），支持 T2I、I2I、图像理解
2. ML-Cache：训练无关的推理加速方法，利用高 max logit token 的表征稳定性，额外加速 2 倍
3. Self-GRPO：轨迹一致的强化学习框架，同时优化 T2I 生成和 MMU 理解
4. <end-of-line> token 实现任意分辨率图像生成
5. 四阶段训练管线：Pre-Training → Mid-Training → SFT → Self-GRPO
6. 多个 benchmark SOTA，UniGenBench 开源第一

---

## Method 核心方法

### 1. 统一离散扩散建模

![](https://arxiv.org/html/2510.06308/x3.png)

*Figure 3: Lumina-DiMOO 离散扩散建模概览——(a) 训练阶段同时 mask 文本和图像 token 并用统一目标预测；(b) 推理阶段从全 mask 状态开始，逐步并行预测并细化 token，实现高质量生成。*

- 文本和图像 token 共享同一 mask-prediction 训练目标
- 前向过程：随机 mask 部分 text/image token，替换为 [Mask]
- 反向过程：模型并行预测所有被 mask 位置的原始 token
- 训练最小化 masked cross-entropy：

$$\mathcal{L}(\theta)=\mathbb{E}_{\mathbf{x}, m, \mathcal{M}}\left[-\sum_{i\in\mathcal{M}}\log p_{\theta}(x_i \mid \tilde{\mathbf{x}}_{\overline{\mathcal{M}}}, c)\right]$$

### 2. 架构设计

- **文本骨干**：LLaDA-Base（8B，126,345 text tokens），无结构修改
- **图像 Tokenizer**：aMUSEd-VQ（16x 下采样，8,192 visual tokens），比 SBER-MoVQGAN（8x）序列更短，兼顾效率与质量
- **特殊 token**：<IMAGE>/</IMAGE>、<canny>/</canny>（可控生成）、<end-of-line>（任意分辨率）、<uncondition>（CFG）等
- **任意分辨率**：每行图像 token 后插入 <end-of-line>，使 1D RoPE 能正确解析 2D 结构

### 3. 推理策略

**图像生成（并行采样）**：
- 4 阶段 MaskGIT 风格解码：Predict → Sample → Mask Schedule → Remask
- 仅从 8,192 图像 codebook 中采样（非全词表）
- cosine schedule 控制每步 remask 数量
- CFG 增强生成质量

**图像理解（block-wise 并行采样）**：
- 半自回归策略：block 内并行预测，block 间顺序推进
- 引入 early stopping：检测到 </answer> 即停止，减少冗余计算

### 4. ML-Cache 训练无关加速

- 核心发现：高 max logit 的 token 在相邻步骤间表征高度稳定（top 94% 的 token 余弦相似度 >0.99）
- 策略：选取 top cache_ratio% 的 token 复用前一步 K/V 和 logits，其余 token 正常计算
- warmup_ratio 步内不缓存（避免上下文不足时的误差），refresh_interval 步后全量计算（避免误差累积）
- 额外 2 倍加速

### 5. 四阶段训练

**Stage 1（Pre-Training）**：80M 高质量图文对，图文对齐，渐进分辨率 256→512，64 A800

**Stage 2（Mid-Training）**：引入 I2I 任务（编辑、主体驱动、可控生成、风格迁移、多视图）+ 增强理解（表格、图表、UI、数学、几何），I2I 分辨率 512，T2I 分辨率 1024

**Stage 3（SFT）**：30M 高质量数据（15M 理解 + 15M 生成），指令遵循

**Stage 4（Self-GRPO）**：轨迹一致的 RL，同时优化 T2I 生成和 MMU 理解，采用 selective timestep 梯度计算降低显存

![](https://arxiv.org/html/2510.06308/x5.png)

*Figure 5: Self-GRPO 框架概览——统一 T2I 生成和多模态理解（MMU）在轨迹一致的强化学习框架下，通过 Group Relative Policy Optimization 同时优化生成质量和理解能力。*

---

## 实验/评估/结果

### T2I 生成

- GenEval 88%（超越 Janus-Pro 80%、BAGEL 82%、GPT-4o 84%），Self-GRPO 后 91%
- DPG-Bench 86.04（超越所有统一模型），Entity 和 Relation 维度尤其突出
- UniGenBench 71.12（开源第一，Layout 82.84、Attribute 81.62）
- OneIG-EN 0.455（Alignment 0.816 第一，Text 0.551 大幅领先 BAGEL 0.244）
- 支持任意分辨率生成，MMaDA 仅支持固定 512x512

### 图像理解

- 通过扩大理解数据弥补 aMUSEd-VQ 缺乏语义信息的不足

### I2I 能力

- 支持编辑、inpainting、外推、风格迁移、主体驱动、可控生成、多视图生成、密集预测
- Zero-shot inpainting → Interactive Retouching（用户自由标注区域精修）

### 与 AR 模型速度对比

- 比 Lumina-mGPT 2.0（AR）快 32 倍
- ML-Cache 额外 2 倍加速

---

## 结论

Lumina-DiMOO 证明了纯离散扩散架构可以在统一多模态模型中实现与 AR/Diffusion 混合方案竞争甚至超越的性能。关键优势包括：(1) 统一的 mask-prediction 目标简化训练；(2) 并行采样大幅提升推理速度；(3) Self-GRPO 统一生成和理解优化；(4) 架构简洁但功能全面。

---

## 思考

### 优点

1. **架构选择的清晰性**：从纯 AR → AR+离散扩散 → 纯离散扩散的演进路径逻辑清晰。论文很好地论证了为何离散扩散是统一多模态的自然选择——它天然支持双向注意力和并行解码。

2. **ML-Cache 的巧妙性**：这是一项很好的工程创新。利用高 max logit token 表征稳定的特性做训练无关加速，2 倍加速无需任何额外训练。

3. **Self-GRPO 的完整性**：同时优化 T2I 和 MMU，且设计了轨迹一致的选择性梯度计算来降低显存。这种"闭环"RL 设计比纯生成 RL 更全面。

4. **任意分辨率方案简单有效**：<end-of-line> token 是一个优雅的最小修改方案，避免了设计新的位置编码。

5. **四阶段训练的系统性**：Pre-Training → Mid-Training → SFT → Self-GRPO，每一阶段目标清晰，逐步增强能力。

### 缺点与待解决问题

1. **文本渲染仍是短板**：尽管 OneIG-EN Text 得分 0.551 在统一模型中领先，但与专用模型仍有差距。UniGenBench Text 仅 25.57。

2. **低层视觉任务表现不佳**：论文自述在超分、去雾、去噪等 low-level 任务上表现差，离散 tokenizer 对这些任务不友好。

3. **与连续扩散的本质差距**：离散扩散虽然推理更快，但生成质量的上限可能不如连续扩散（如 FLUX）。纯离散路线能否在更大规模上追平连续扩散仍是开放问题。

4. **训练效率问题**：bidirectional attention 导致每步计算成本高于 AR，mask ratio 训练导致监督稀疏、梯度方差大。

5. **架构继承自 AR 模型**：使用为 AR 设计的 LLaDA 架构，可能不是离散扩散的最优架构。

### 与已有 Wiki 的连接

- 关联概念：[[离散扩散]]、[[Masked Diffusion]]、[[GRPO]]、[[统一模型]]
- 关联实体：[[LLaDA]]、[[aMUSEd-VQ]]、[[MMaDA]]
- 关联比较：与 [[Show-o2]]（AR+FM 混合路线）、[[Emu3.5]]（纯 AR 路线）的架构对比，与 [[BAGEL]]（AR+连续扩散）的生成速度对比

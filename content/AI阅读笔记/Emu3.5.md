---
title: 'Emu3.5: Native Multimodal Models are World Learners'
authors:
  - Emu3.5 Team
institutions: BAAI (北京智源人工智能研究院)
aliases:
  - Emu3.5
date: '2026-04-30'
publish_date: 2025-10
venue: arXiv:2510.26583
tags:
  - 论文
  - 多模态
  - 原生多模态
  - 世界模型
  - 自回归
  - 扩散模型
  - GRPO
  - 图像生成
url: 'https://arxiv.org/abs/2510.26583'
code: 已开源（https://github.com/baaivision/Emu3.5）
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：Emu3.5 是一个 34.1B 参数的原生多模态世界模型，通过在 13T token 的视频交错数据上进行统一的 next-token prediction 预训练，配合大规模 RL 后训练（GRPO）和 DiDA（Discrete Diffusion Adaptation）将逐 token 解码加速约 20 倍，在图像生成、编辑、视觉叙事和世界探索等任务上达到与 Gemini 2.5 Flash Image 可比水平，是首个在质量和速度上能与闭源扩散模型竞争的自回归模型。

![](https://arxiv.org/html/2510.26583/x1.png)

*Figure 1: Emu3.5 与 SOTA 模型的图像生成和编辑 benchmark 对比，以及与 Gemini 2.5 Flash Image 的交错生成偏好评估。*

---

## Intro

### Motivation

语言模型在文本推理和生成上取得了巨大成功，但文本只提供了有限的世界视图。人类通过视觉（尤其是长时间视频）感知和学习世界，这些视频承载着丰富的上下文、因果和时序一致性信息。前期 Emu 系列证明了统一多模态 next-token prediction 的可行性，但关键问题未解决：
1. 如何规模化预训练长视频交错数据？
2. 如何实现通用多模态交互？
3. 如何高效预测数万个视觉 token？

### 贡献

1. **34.1B 原生多模态模型**：统一 next-token prediction，预训练 13T token（主要来自视频交错数据）
2. **大规模 RL 后训练**：GRPO 增强多模态推理和生成
3. **DiDA**（Discrete Diffusion Adaptation）：将逐 token 自回归转换为双向并行预测，加速 20 倍
4. **长时域多模态能力**：视觉叙事、视觉引导、世界探索、具身操控
5. **首个开源自回归模型**能在质量和速度上媲美闭源扩散模型
6. **完全开源**

---

## Method 核心方法

Emu3.5 是一个 34.1B 参数的原生多模态世界模型，通过标准 decoder-only Transformer + IBQ 离散视觉 tokenizer + 统一的 next-token prediction 目标，在 13T token 的视频交错数据上预训练，配合 GRPO 大规模 RL 后训练和 DiDA 推理加速。

### 1. 统一架构

| 参数 | 值 |
|------|-----|
| 总参数 | 34.1B（31.2B Transformer + 2.9B Embedding） |
| 层数 | 64，hidden 5120，intermediate 25600 |
| 注意力 | GQA（64Q/8KV）+ QK-Norm + RoPE |
| 激活 | SwiGLU + RMSNorm pre-norm |
| 词表 | 282,926（151,854 text + 131,072 vision tokens） |
| 上下文 | 32,768 tokens，dropout 0.1 |

标准 decoder-only Transformer，文本和视觉 token 统一序列处理。视觉 token loss 权重 0.5（防止视觉 token 主导训练）。

### 2. Tokenizer：IBQ + SigLIP 蒸馏 + 扩散解码器

采用 **IBQ**（Improved Binary Quantization）框架，降采样因子 f=16。视觉 codebook 扩展至 131,072（参数 455M，宽度缩放），通过 **SigLIP 特征蒸馏**（REPA 式）增强离散 token 语义信息。

**扩散图像解码器（可选）**：同量化 token → 2x 分辨率输出 + 细节增强。LoRA 蒸馏加速 10x（50→4 steps）。

**视频解码器**：DiT 架构，以生成的关键帧 token 为条件，额外 4 通道 mask 支持任意数量中间帧。

### 3. 预训练：13T tokens，视频交错为中心

**数据组成**：视频交错数据（63M 视频，平均 6.5 分钟，790 年连续视频，占采样比 55%）为核心；500M 图文对 + 30M 视频-文本对 + 27.35M Any-to-Image + 3T 纯文本为辅助。

**视频预处理 pipeline**：PySceneDetect 分镜 → Whisper ASR → spaCy 文本分割 → 时序对齐交错序列。两阶段过滤（基础：时长/分辨率/人脸检测/语言平衡；高级：DeQA 帧质量 + DINO/FG-CLIP 去重 + LLM 文本评分）。

**两阶段预训练**：

| 阶段 | Tokens | 分辨率 | 数据采样比 | LR |
|------|--------|--------|-----------|-----|
| S1 | 10T | max 1024 tokens (512²) | 视频交错 55% | 5e-4 |
| S2 | 3T | 1024-4096 tokens (512²-1024²) | 视频交错 55% + 增强标注 | 1e-5 |

S2 引入语义分割/摘要、视觉 caption、多模态摘要等丰富标注。所有阶段 AdamW（β₁=0.9, β₂=0.95），cosine schedule，TP=8, CP=2。

### 4. 后训练：SFT + GRPO RL + DiDA

**SFT（150B tokens，两阶段）**：建立统一多模态生成接口，覆盖 6 大任务——通用（T2I/语言/VL 29.7B）、Any-to-Image（56.2B）、视觉叙事（10.1B）、视觉引导（22.5B）、世界探索（17.5B）、具身操控（14.1B）。Stage 1 标准分辨率 → Stage 2 高分辨率（T2I 1024px）。

**GRPO RL**：首个大规模联合多任务多模态 RL。统一奖励系统包含通用奖励（CLIP 对齐 + VLM 准确度 + 美学评分）+ 任务特定奖励（OCR 保真度、人脸识别、VLM 一致性）。GRPO 算法，batch 640，LR 1e-6，rollout 8。平均 reward 从 4.5 升至 7.1+。

**DiDA（离散扩散适配）**：将逐 token 自回归转换为双向并行预测。每张图复制含噪副本，噪声 token 因果关注前置干净 token 并双向关注同图内噪声 token。加速约 20 倍，仅需几十亿 tokens 适配数据，不牺牲质量。

---

## 实验/评估/结果

### 图像生成与编辑

与 Gemini 2.5 Flash Image (Nano Banana) 在图像生成和编辑 benchmark 上性能可比，文本渲染超越 Gemini。

**GenEval 对比（Figure 1a）**：Emu3.5 在 GenEval 上的综合表现与闭源扩散模型竞争，作为首个自回归模型在质量和速度上媲美扩散模型。

**交错生成偏好评估（Figure 1b）**：在视觉叙事和视觉引导任务上，Automated Win Rate 超越 Gemini 2.5 Flash Image。

### 预训练 Scaling 行为

**九个验证集上的 loss 曲线**：3 个域内（T2I/I2T/视频交错）+ 3 个 OOD（ISG-Bench/OpenING/MMIE）+ 3 个 SFT 下游（视觉叙事/视觉引导/世界探索），全部持续下降——证明视频交错为中心的大规模预训练具有稳健的泛化能力。

### 训练 Recipe 汇总

| Hyperparameter | Stage 1 | Stage 2 |
|---------------|---------|---------|
| Learning Rate | 5e-4 | 1e-5 |
| Batch Size | 448 | 448 |
| Sequence Length | 32768 | 32768 |
| Seen Tokens | 10.3T | 3.5T |
| Visual:Text Loss Weight | 0.5:1.0 | 0.5:1.0 |
| Text / Image-Text / Video-Text / X2I / Video Interleaved | 0.2/0.2/0.05/0/0.55 | 0.18/0.16/0.08/0.03/0.55 |

### SFT 任务数据统计

| Task | #Tokens (B) | Output Type |
|------|------------|-------------|
| General (T2I+Language+VL) | 29.7 | Text/Image |
| Any-to-Image (X2I) | 56.2 | Image |
| Visual Narrative | 10.1 | Interleaved |
| Visual Guidance | 22.5 | Interleaved |
| World Exploration | 17.5 | Interleaved |
| Embodied Manipulation | 14.1 | Interleaved |

### RL Training

多任务 GRPO RL 训练过程中，平均 reward 从 ~4.5 稳定上升至 >7.1，证明统一奖励聚合机制有效整合了异构任务反馈。

---

## 结论

Emu3.5 代表了大规模原生视觉语言生成的重要一步。它证明了：(1) 视频交错数据规模化预训练可以涌现长时域多模态推理能力；(2) 自回归模型通过 DiDA 可以达到扩散模型的推理速度；(3) 统一的多模态接口使不同任务之间可以相互增强。

---

## 思考

### 优点

1. **数据哲学的前瞻性**：Emu3.5 的核心理念——视频是最大最自然的模拟器，视频的时序连续性和跨模态对齐为世界建模提供了天然监督。这比单纯堆图文对数据更有深度。

2. **DiDA 的工程突破**：自回归模型的逐 token 解码一直是速度瓶颈。DiDA 将 AR 转换为离散扩散，实现了 20 倍加速，且不牺牲质量。这是让自回归视觉生成走向实用的关键工程突破。

3. **任务定义的想象力**：视觉叙事、视觉引导、世界探索、具身操控——这些任务定义超出了传统 benchmark 的范畴，体现了"世界模型"的野心。

4. **开源承诺**：34.1B 参数模型完全开源，在 GPT-4o/Gemini 的技术细节完全保密的情况下，Emu3.5 为社区提供了研究原生多模态模型的宝贵资源。

### 缺点与待解决问题

1. **训练成本极高**：13T tokens + GRPO RL 的训练成本对于大多数研究机构不可承受。论文未报告总 GPU 小时。

2. **DiDA 的局限性**：DiDA 仅加速每张图生成，文本部分仍需自回归逐 token 解码。在交错生成场景中，文本解码可能仍是瓶颈。

3. **世界模型的评估标准缺失**：世界探索和具身操控的评估主要依赖定性展示，缺乏系统性的定量指标。如何衡量世界模型的好坏仍是开放问题。

4. **与扩散模型的根本差异**：AR 模型和扩散模型在生成任务上各有所长，Emu3.5 在某些编辑任务上仍可能弱于专用扩散模型。论文对此讨论不够深入。

### 与已有 Wiki 的连接

- 关联概念：[[原生多模态模型]]、[[下一个token预测]]、[[离散扩散]]、[[GRPO]]、[[世界模型]]
- 关联实体：[[Emu3]]、[[Gemini 2.5 Flash]]
- 关联比较：与 [[BAGEL]] 的统一多模态路线对比（AR vs AR+Flow Matching），与 [[Show-o2]] 的 AR+FM 混合路线对比

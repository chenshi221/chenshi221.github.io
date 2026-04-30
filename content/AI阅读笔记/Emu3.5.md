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

### 1. 统一架构

![](https://arxiv.org/html/2510.26583/x3.png)

*Figure 3: Emu3.5 架构概览——标准 decoder-only Transformer 统一处理文本和视觉 token，通过 next-token prediction 进行端到端大规模训练。推理时通过 DiDA 将逐 token 自回归转换为双向并行生成，加速约 20 倍。*

- 标准 decoder-only Transformer（64 层，5120 hidden，34.1B 参数）
- GQA（64Q/8KV）+ QK-Norm + SwiGLU + RoPE
- Vocab 282,926（151,854 text + 131,072 vision tokens）
- 上下文长度 32,768

### 2. Tokenizer

- IBQ 框架，降采样因子 f=16
- 视觉 codebook 扩展至 131,072（相比 Emu3 大幅增加）
- SigLIP 特征蒸馏（REPA）增强离散 token 语义信息
- 扩散解码器（可选）：2x 分辨率 + 细节增强，LoRA 蒸馏加速 10x（50 steps → 4 steps）
- 视频解码器：DiT 架构，以生成的关键帧 token 为条件，支持任意数量中间帧

### 3. 预训练（13T tokens，两阶段）

**数据组成**：
- 视频交错数据：63M 视频（平均 6.5 分钟，约 790 年连续视频），通过 PySceneDetect 分镜 + Whisper ASR → 时序对齐的交错图文序列
- 图文对：500M image-text pairs + 30M video-text pairs
- Any-to-Image：27.35M 编辑样本
- 纯文本：3T tokens

**两阶段预训练**：
- S1（10T tokens）：基础对齐，max 1024 visual tokens（512x512）
- S2（3T tokens）：高质量增强，更高分辨率（up to 1024x1024），更多交错标注

### 4. 后训练

- **SFT**（150B tokens）：建立统一多模态生成接口，涵盖通用任务、X2I、视觉叙事、视觉引导、世界探索、具身操控
- **RL**（GRPO）：大规模强化学习增强多模态推理和生成

### 5. DiDA（Discrete Diffusion Adaptation）

- 将自回归逐 token 解码转换为离散扩散的双向并行预测
- 每张图推理加速约 20 倍
- 仅需几十亿 tokens 的适配数据
- 不牺牲质量

![](https://arxiv.org/html/2510.26583/x9.png)

*Figure 9: DiDA（Discrete Diffusion Adaptation）——(a) 预训练/SFT/RL 阶段使用标准 next-token prediction；(b) 适配阶段每张图复制一份含噪副本，噪声 token 因果关注前置干净 token 并双向关注同图内噪声 token，实现并行生成。*

### 6. 任务能力

- **Any-to-Image（X2I）**：开放世界编辑、精确控制、文本渲染 SOTA
- **视觉叙事**：交错生成连贯故事线（图文交织），维持角色/风格一致性
- **视觉引导**：逐步操作过程的交错生成（烹饪、手工等）
- **世界探索**：用户交互式 + 自由探索式的虚拟世界漫游
- **具身操控**：长时域任务分解为子任务序列，预测关键帧

---

## 实验/评估/结果

### 图像生成与编辑

- GenEval / DPG-Bench 等上与 Gemini 2.5 Flash Image 可比
- 文本渲染超越 Gemini 2.5 Flash Image
- X2I 编辑任务可与 FLUX.1 Kontext 竞争

### 交错生成

- 视觉叙事和视觉引导上，Automated Win Rate 超越 Gemini 2.5 Flash Image
- 角色一致性、时序连贯性突出

### 世界建模

- 世界探索和具身操控展现通用泛化能力
- 跨域泛化：从真实场景泛化到想象场景

### 关键发现

- 随着预训练计算量增加，OOD 多模态任务的 validation loss 持续下降——更强的泛化信号
- RL 后训练建立了共享的多模态接口，不同任务间可相互迁移和受益
- NTP 模型可高效转化为双向预测器（DiDA）

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

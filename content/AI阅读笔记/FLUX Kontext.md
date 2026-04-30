---
title: 'FLUX.1 Kontext: Flow Matching Rectified Transformer for Unified Image Generation and Editing'
authors:
  - 论文作者（Black Forest Labs）
institutions: Black Forest Labs
aliases:
  - FLUX Kontext
  - FLUX.1 Kontext
date: '2026-04-30'
publish_date: 2025-04
venue: 'arXiv:2504.19713'
tags:
  - 论文
  - 图像生成
  - 图像编辑
  - 流匹配
  - Rectified Flow
  - 统一模型
  - DiT
url: 'https://arxiv.org/abs/2504.19713'
code: 已开源
rating: ⭐⭐⭐⭐
status: Read
---
**一句话总结**：FLUX.1 Kontext 将图像生成和编辑统一在单个模型中，通过序列拼接（输入图+用户指令+目标图）的简洁方案实现零样本图像编辑，并利用 3D RoPE、Rectified Flow 和 LADD 蒸馏实现了高效的推理。

---

## Intro

### Motivation

传统的图像编辑方法通常由两个独立模型组成：一个模型理解编辑指令，另一个模型执行生成。这种 pipeline 存在信息瓶颈和累积误差的问题。此外，现有的图像编辑模型通常只在固定的编辑类型上训练（如 inpainting、outpainting），缺乏泛化到新编辑任务的能力。

FLUX.1 Kontext 的出发点是：**能否训练一个原生支持图像编辑的统一模型**，它既能生成图像，也能在同一个前向传播中完成图像编辑？

### 核心主张

1. **序列拼接（Sequence Concatenation）**是统一生成和编辑的极简方案：将输入图像 token、指令 token、输出图像 token 拼接成一个序列，让模型自回归或流式地预测
2. **3D RoPE** 处理拼接序列中不同来源 token 的位置关系
3. **Rectified Flow** 提供高效的概率流路径
4. **LADD 蒸馏**将大模型的高质量生成能力迁移到小模型

### 贡献

1. 提出 FLUX.1 Kontext，单一模型统一处理 T2I 生成、图像编辑、inpainting 等多种任务
2. 序列拼接范式无需任务特定的架构修改
3. 3D RoPE 有效处理跨图像 token 的位置编码
4. 通过 LADD 蒸馏实现高效推理
5. 提出 KontextBench 评估统一模型的编辑能力

---

## Method 核心方法

![](https://ar5iv.labs.arxiv.org/html/2506.15742/assets/img/kontext_v2.jpg)

*Figure 1: FLUX.1 Kontext 的架构总览——左侧为输入图像和文本指令，通过 VAE 编码为 latent token，与文本 token 拼接后送入 DiT 进行 Rectified Flow 去噪，最终解码为编辑后的目标图像。序列拼接方案无需任务特定的架构修改即可统一生成和编辑。*

### 1. 序列拼接范式（Sequence Concatenation）

核心思想非常简洁：

- **T2I 生成模式**：[BOS][Text Tokens][SEP][Noise Tokens]
- **图像编辑模式**：[BOS][Input Image Tokens][SEP][Text Tokens][SEP][Output Noise Tokens]

模型看到的是输入图 + 指令 + 输出图的完整上下文，通过 Rectified Flow 将输出部分的噪声 token 逐步去噪为目标图像。

这种设计的优势：
- 不需要额外的编码器处理输入图像
- 输入和输出共享同一个 VAE 和 Transformer
- 任务拓展只需改输入格式，不需要改架构

### 2. 3D RoPE（3D 旋转位置编码）

拼接序列中同时存在输入图像、文本和输出图像的 token，它们各自有二维空间结构。3D RoPE 为每种模态的 token 分配独立的位置维度：

- X 维度和 Y 维度：图像 token 的空间位置
- Z 维度：区分不同模态/图像（输入图=0，输出图=1）
- 文本 token 的 X、Y 维度设为 0

这使得模型能区分"这是输入图左上角的 token"v.s."这是输出图左上角的 token"。

![](https://ar5iv.labs.arxiv.org/html/2506.15742/assets/img/fusedditblock.jpg)

*Figure 2: FLUX 的 Fused DiT Block 及 3D RoPE 示意图——混合双流/单流 Transformer 块，结合 3D RoPE 为每个 latent token 赋予 (t, h, w) 时空坐标，实现跨图像 token 的位置区分。*

### 3. Rectified Flow

FLUX Kontext 使用 Rectified Flow（一种 OT 路径的变体）作为生成框架：

- 前向过程：x_t = (1-t)·x_0 + t·ε
- 向量场方向：从噪声直线指向干净数据
- 训练目标：学习 v_θ(x_t, t)，预测速度方向 (x_1 - x_0)
- 直线轨迹使得少步采样更高效

### 4. LADD 蒸馏（Latent Adversarial Diffusion Distillation）

将大 FLUX 模型的生成能力蒸馏到更小的模型：

- 使用大型 FLUX 模型作为 teacher
- 学生模型在前向 ODE 轨迹上学习 teacher 的向量场
- 结合对抗训练（discriminator）提升蒸馏后的感知质量
- 实现少步（1-4 步）高质量生成

---

## 实验/评估/结果

### 文本到图像生成（T2I）

- FLUX.1 Kontext 在 GenEval 等 benchmark 上达到 SOTA 水平
- 与原版 FLUX.1 生成质量持平
- 支持多种宽高比的生成

### 图像编辑

**KontextBench**（自建 benchmark，评估多种编辑类型）：
- FLUX Kontext 在指令式编辑、inpainting、outpainting、风格迁移、物体替换等任务上表现出色
- 序列拼接方案在零样本编辑任务上展现了强大的泛化能力
- 无需任务特定的微调即可处理多样化的编辑需求

**关键定性结果**：
- 指令遵循度高，能准确理解编辑意图
- 非编辑区域保持良好（背景一致性）
- 可同时处理多种编辑操作

![](https://ar5iv.labs.arxiv.org/html/2506.15742/assets/x3.png)

*Figure 3: FLUX.1 Kontext 的文本编辑效果示例——模型能够精准修改图像中的文字内容（如"Flowers"改为"Flux"），同时保持字体风格和背景的一致性。*

### 消融实验

- 序列拼接 v.s. 分开处理：拼接方案显著优于先 encode 再 decode 的两阶段方案
- 3D RoPE v.s. 标准位置编码：3D RoPE 对区分输入/输出图像 token 至关重要
- 蒸馏模型的效率：LADD 蒸馏后模型在 4 步采样下质量接近 teacher 的 50 步结果

---

## 结论

FLUX.1 Kontext 展示了序列拼接作为统一图像生成和编辑的简洁而强大的范式。通过 3D RoPE 的位置编码设计和 Rectified Flow 的高效采样，单一模型即可覆盖多种视觉任务。结合 LADD 蒸馏，模型在保持高质量的同时实现了高效推理。

---

## 思考

### 优点

1. **极简设计哲学**：序列拼接方案不需要任何任务特定的架构修改——改变输入格式就能切换任务。这种"让模型自己从数据中学习任务结构"的思路符合 scaling law 的精神。

2. **3D RoPE 的巧妙设计**：用第三个维度区分不同图像的 token，简单但解决了核心难题（输入图 token v.s. 输出图 token 的位置歧义）。

3. **与 FLUX 生态的整合**：继承了 FLUX 系列的 Rectified Flow、DiT 架构等积累，Kontxt 能力是增量的而不是推倒重来。

4. **蒸馏的实际价值**：LADD 蒸馏让大模型的编辑能力可以被小模型继承，对实际部署意义重大。

### 缺点与待解决问题

1. **序列长度问题**：拼接方案导致序列长度显著增加（输入图+文本+输出图），对于高分辨率或多图编辑场景，显存开销是瓶颈。

2. **编辑粒度**：虽然能处理多种编辑类型，但精细的局部编辑（如只修改一个人的眼睛颜色）的能力可能不如专用工具（如 BrushNet）精确。

3. **多轮编辑**：论文没有深入探讨多轮编辑（先编辑 A，在 A 的结果上再编辑 B），这在实际使用中很常见。

4. **KontextBench 的标准化**：自建 benchmark 虽然好，但与社区标准（如 MagicBrush, EditBench）的对比不够充分。

### 与已有 Wiki 的连接

- 关联概念：[[Rectified Flow]]、[[DiT]]、[[知识蒸馏]]、[[图像编辑]]、[[旋转位置编码]]
- 关联实体：[[FLUX]]、[[Stable Diffusion 3]]
- 关联论文：[[AI阅读笔记/Flow Matching]]
- 关联比较：[[Wiki/Comparisons/扩散模型架构比较 UNet vs DiT]]

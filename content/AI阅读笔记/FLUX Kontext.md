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

FLUX.1 Kontext 的方法论是一个四层架构：底座 FLUX DiT → 序列拼接统一范式 → Rectified Flow 训练 → LADD 蒸馏加速。以下逐层拆解。

### 1. 底座架构：FLUX DiT (Diffusion Transformer)

FLUX.1 Kontext 建立在 FLUX.1 的 DiT 架构之上，后者是一个大规模 Rectified Flow Transformer，在图像 VAE 的潜空间中训练。

#### 1.1 混合双流/单流 Transformer 块

FLUX DiT 由两种 Transformer 块交替组成：

![](https://ar5iv.labs.arxiv.org/html/2506.15742/assets/img/fusedditblock.jpg)

*Figure 2: FLUX 的 Fused DiT Block。混合双流（double stream）和单流（single stream）块，结合 3D RoPE 为每个 latent token 赋予 (t, h, w) 时空坐标。融合 FFN 块将注意力输入/输出投影与 MLP 合并以提升 GPU 利用率。*

- **双流块（Double Stream Blocks）**：文本 token 和图像 token 使用独立的权重参数。注意力操作在拼接后的 token 序列上进行（实现跨模态信息混合），但 FFN 分别处理各自模态。这种分离设计允许文本和视觉特征保持各自的表示空间特性。
- **单流块（Single Stream Blocks）**：在双流块之后，将文本和图像 token 拼接为统一序列，经过 38 层单流 Transformer 块。此时文本和图像共享全部参数，实现深层统一建模。最终丢弃文本 token，仅保留图像 token 用于解码。

**融合 FFN 块（Fused FFN Block）**：受 ViT-22B 启发，FLUX 将注意力层的输入/输出投影与 MLP 层合并：
- 将调制参数数量减少一半
- 合并后的矩阵乘法更大，GPU 利用率更高
- 训练和推理效率显著提升

#### 1.2 高保真 VAE

FLUX 从头训练了一个卷积 VAE，关键设计选择：

| VAE | 潜通道数 | PDist ↓ | SSIM ↑ | PSNR ↑ |
|-----|---------|---------|--------|--------|
| **FLUX-VAE** | **16** | **0.332** | **0.896** | **31.1** |
| SD3-VAE | 16 | 0.452 | 0.858 | 29.6 |
| SD3-TAE | 16 | 0.746 | 0.774 | 27.9 |
| SDXL-VAE | 4 | 0.890 | 0.748 | 25.9 |
| SD-VAE | 4 | 0.949 | 0.720 | 25.0 |

- 16 个潜通道（是 SDXL 的 4 倍），大幅提升重建保真度
- 使用感知距离（VGG PDist）和对抗损失联合训练
- 冻结的 VAE：训练和推理时保持固定，所有图像先编码为潜 token 再送入 DiT

#### 1.3 3D 旋转位置编码（3D RoPE）

FLUX 使用分解式三维 RoPE，为每个 latent token 分配时空坐标：

$$\text{pos}(t, h, w) = [\text{RoPE}_t(t), \text{RoPE}_h(h), \text{RoPE}_w(w)]$$

- **t 维度**：时间步（单图时 t≡0；多帧/视频时递增）
- **h, w 维度**：图像空间位置（对应 latent grid 的行列索引）
- 三个维度的 RoPE 在注意力计算中分别作用于 Q 和 K 的不同子空间

这种设计让模型在注意力计算中能够天然地区分 token 的空间邻近性和时间连续性。

### 2. 序列拼接范式——统一生成与编辑的极简方案

这是 FLUX Kontext 最核心的设计哲学：**用单一的 token 序列格式覆盖所有任务**，通过改变序列构成来切换任务模式，而非修改模型架构。

#### 2.1 Token 序列构造

所有图像首先通过冻结的 FLUX-VAE 编码为 latent patch token。序列拼接规则：

**T2I 生成模式**（y = ∅）：
```
[BOS] [Text Tokens: T₁ T₂ ... Tₙ] [SEP] [Noise Tokens: N₁ N₂ ... Nₘ]
```
纯文本到图像生成，与标准 FLUX.1 完全一致。

**图像编辑模式**（y ≠ ∅）：
```
[BOS] [Context Image Tokens: C₁ C₂ ... Cₖ] [SEP] [Text Tokens: T₁ ... Tₙ] [SEP] [Noise Tokens: N₁ ... Nₘ]
```
上下文图像 token + 编辑指令 + 噪声 token 的拼接，模型在完整上下文中去噪生成目标图像。

#### 2.2 3D RoPE 在拼接序列中的应用

拼接序列的关键挑战：**如何让模型区分"输入图的左上角 token"和"输出图的左上角 token"**（它们的 h, w 坐标可能相同）。

3D RoPE 的解决方案：为上下文图像 token 和输出图像 token 分配不同的 t 坐标：

$$\mathbf{u}_{\text{output}} = (0, h, w), \quad \mathbf{u}_{\text{context}} = (1, h, w)$$

t 坐标作为"虚拟时间维度"将上下文和目标的潜在空间干净地分离，同时保持各自的内部空间结构。扩展到多图场景（y₁, y₂, ..., y_N）只需递增 t 坐标：u_{y_i} = (i, h, w)。

#### 2.3 设计方案对比

| 方案 | 效果 | 问题 |
|------|------|------|
| **通道拼接**（输入+输出 latent 沿通道维拼接） | 初始实验效果较差 | 强制输入和输出共享空间坐标，语义混淆 |
| **序列拼接**（token 序列首尾相接） ✓ | 显著优于通道拼接 | 序列长度增加，但对 Transformer 天然支持 |
| **独立编码器**（额外 encoder 处理输入） | 增加参数和复杂度 | 违背"极简统一"的设计哲学 |

### 3. Rectified Flow 训练

#### 3.1 前向过程与训练目标

FLUX Kontext 使用 Rectified Flow（OT 路径的变体）在潜空间中训练：

**前向加噪**：$z_t = (1-t) \cdot x_0 + t \cdot \varepsilon$，其中 $\varepsilon \sim \mathcal{N}(0,1)$

**速度预测目标**（velocity prediction）：

$$\mathcal{L}_{\theta} = \mathbb{E}_{t \sim p(t), x, y, c, \varepsilon}\left[\|v_{\theta}(z_t, t, y, c) - (\varepsilon - x_0)\|_2^2\right]$$

与 DDPM 预测噪声 ε 不同，这里预测的是速度方向 $\varepsilon - x_0$（从数据指向噪声的向量）。这使得采样路径成为**直线**：$x_t = (1-t)x_0 + t\varepsilon$。

**纯文本模式**：当 y=∅ 时，省略所有上下文 token，保留标准 T2I 生成能力。

#### 3.2 Logit-Normal 时间步采样

时间步 t 从 Logit-Normal 分布采样（而非均匀分布）：

$$p(t; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \cdot \frac{1}{t(1-t)} \cdot \exp\left(-\frac{(\text{logit}(t) - \mu)^2}{2\sigma^2}\right)$$

其中 $\text{logit}(t) = \log\frac{t}{1-t}$，默认 σ=1.0。

**μ 的物理意义——与 Resolution Shift 的关系**：论文证明，此前工作中用于高分辨率训练的"时间步 shift"参数 α 可以通过 Logit-Normal 的 μ 参数统一表达：μ = log α。例如 α=3.0（1024² 分辨率常用值）对应 μ≈1.0986。

**分辨率自适应**：训练时根据图像分辨率调整 μ：
- 低分辨率（256²）：μ 较小（噪声时间占比少）
- 高分辨率（1024²）：μ 较大（噪声时间占比多，网络更多时间学习全局结构）

推理时可通过公式 $t' = \frac{\alpha t}{1 + (\alpha-1)t}$ 重新分配时间步。

#### 3.3 多任务联合训练

从纯 T2I 的 FLUX.1 checkpoint 出发，在 I2I（图像编辑）和 T2I 数据上联合微调：
- 收集和筛选数百万关系对 $(x | y, c)$（目标图 | 上下文图, 文本指令）
- I2I 和 T2I 在同一个 batch 中混合训练
- 单上下文图像为主要条件模式（多图能力天然支持但未重点优化）

### 4. LADD 蒸馏——交互式速度的引擎

LADD（Latent Adversarial Diffusion Distillation）是实现 3-5 秒/图推理速度的核心技术。

#### 4.1 两阶段流程

**Stage 1：Teacher 训练**。使用完整的 Rectified Flow 目标（Eq.3）训练大型 FLUX Kontext 模型，50-250 步 ODE 采样 + CFG（classifier-free guidance）。

**Stage 2：学生蒸馏**。将 teacher 的生成能力迁移到更小的学生模型：

- **蒸馏损失**：学生在 teacher 的 ODE 轨迹上学习向量场。给定从 teacher 采样路径上的点 $z_t$，学生预测的向量场应与 teacher 在该点的向量场一致
- **对抗损失**：引入判别器（discriminator），对学生生成的低步数（1-4 步）样本进行真伪判断，提升感知质量
- **学生模型变体**：
  - FLUX.1 Kontext \[pro\]：完整模型 + LADD，平衡质量与速度
  - FLUX.1 Kontext \[dev\]：12B DiT，仅 I2I 训练（无 T2I），通过 guidance-distillation 获得
  - FLUX.1 Kontext \[max\]：更多计算预算，追求极致生成质量

#### 4.2 蒸馏效果

| 模型 | 采样步数 | 推理时间（1024²） | 质量 |
|------|---------|------------------|------|
| Teacher（未蒸馏） | 50-250 | >30s | Reference |
| \[pro\]（LADD） | 4-8 | 3-5s | 接近 teacher |
| \[dev\]（12B） | 1-4 | 1-3s | 略低于 \[pro\] |

LADD 使 FLUX Kontext 在保持编辑质量的同时，推理速度比竞品（如 GPT-Image-1）快一个数量级。

### 5. 训练基础设施与工程细节

#### 5.1 分布式训练

| 组件 | 配置 | 说明 |
|------|------|------|
| 并行策略 | FSDP2 (Full Sharded Data Parallel v2) | 分片参数、梯度和优化器状态 |
| 混合精度 | all-gather: bfloat16, reduce-scatter: float32 | 通信用低精度节省带宽，梯度和用高精度保证数值稳定 |
| 激活检查点 | 选择性（selective activation checkpointing） | 只重计算 attention 和 FFN 的激活，减少 VRAM 峰值 |
| 注意力加速 | Flash Attention 3 | 利用异步和低精度加速注意力计算 |
| 编译优化 | 逐块区域编译（regional compilation） | 对单个 Transformer 块做 torch.compile |

#### 5.2 安全训练

- 基于分类器的数据过滤：去除 NCII（非自愿亲密图像）和 CSAM（儿童性虐待材料）相关内容
- 对抗训练（adversarial training）：在蒸馏阶段引入对抗样本增强鲁棒性

#### 5.3 推理时技术

- **Classifier-Free Guidance (CFG)**：文本丢弃率 0.1，引导强度根据任务调整
- **ODE 求解器**：Euler 或 Heun 方法，步数可调（质量-速度权衡）
- **VAE 解码**：16 通道 latent → 1024×1024×3 像素，单次前向传播

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

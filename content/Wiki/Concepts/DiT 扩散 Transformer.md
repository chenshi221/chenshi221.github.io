---
type: concept
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - Diffusion Transformer
  - 扩散Transformer
tags:
  - DiT
  - 扩散模型
  - Transformer
  - 架构
  - 图像生成
sources:
  - '[[Wiki/Sources/Seedream 4.0 多模态图像生成]]'
  - '[[Wiki/Sources/FLUX.1 Kontext 上下文编辑]]'
  - '[[Wiki/Sources/OminiControl]]'
confidence: high
---
# DiT 扩散 Transformer

## 定义

**DiT（Diffusion Transformer）** 是 Peebles & Xie 于 2023 年（ICCV 2023）提出的扩散模型架构，核心思想是用 **Transformer 替代 UNet** 作为扩散模型的去噪骨干网络。它标志着扩散模型架构从 CNN 时代进入 Transformer 时代。

## 核心设计

### 架构组成

1. **Patchify（图像分块）**：将输入图像切分为 patch（类似 ViT），每个 patch 经线性投影得到 token 序列
2. **Transformer Block**：标准 DiT block 包含多头自注意力和 pointwise feedforward，在 layer norm 和 attention/FFN 之间注入条件信息
3. **条件注入**：通过 **adaLN-Zero**（自适应层归一化零初始化）将时间步 t 和类别标签 c 作为条件注入每个 Transformer block

### adaLN-Zero 机制

- 回归生成 scale 和 shift 参数：`(β, γ) = MLP(c, t)`
- 对每个 block 进行条件归一化：`x = γ * LayerNorm(x) + β`
- 零初始化：将最后的线性层权重初始化为 0，确保训练初期模型作为恒等映射

### 四种条件策略

| 策略 | 方式 | 特点 |
|------|------|------|
| In-context conditioning | 将 t, c 作为额外 token 拼入序列 | 类似 ViT 的 class token |
| Cross-attention block | 在自注意力后插入交叉注意力 | 类似 LDM 的 UNet 设计 |
| Adaptive layer norm (adaLN) | 用 t, c 回归 γ, β | 标准 Transformer 中最常用 |
| **adaLN-Zero** | adaLN + 零初始化残差 | **最终采用方案**，训练稳定 |

## 为什么 DiT 重要

### UNet → DiT 的驱动力

1. **可扩展性更好**：Transformer 的 scaling law 已被充分验证，DiT 可以直接受益于模型规模的扩大
2. **统一的架构范式**：文本（LLM）、视觉（ViT）、生成（DiT）全部统一到 Transformer，简化工程栈
3. **多模态融合更自然**：Transformer 原生支持序列拼接和交叉注意力，适合条件生成（如 text-to-image）
4. **去除了 CNN 的归纳偏置**：UNet 的下采样/上采样、跳跃连接等 CNN 特定设计被证明不是必需的

### 关键影响

- **Sora（OpenAI）**：视频生成的 DiT backbone
- **Stable Diffusion 3 / FLUX**：图像生成的 DiT backbone
- **Seedream 4.0**：字节多模态统一的 DiT backbone
- **OminiControl**：DiT 上的极简控制，仅 0.1% 额外参数即实现空间条件控制

## DiT vs UNet 对比

| 维度 | UNet | DiT |
|------|------|-----|
| 基础单元 | 卷积 | 自注意力 |
| 归纳偏置 | 强（局部性、平移等变性） | 弱（需更多数据） |
| 可扩展性 | 受限 | 已验证 scaling law |
| 条件注入 | Cross-attention | adaLN / Cross-attention |
| 计算模式 | 空间局部 | 全局注意力 |
| GFLOPs 效率 | 更高 | 较大模型时更优（参数效率更好） |

## 训练策略

### 缩放行为（DiT paper 发现）

- GFLOPs 是比参数量更好的计算量指标
- 更大的 DiT 模型持续提升 FID，没有明显饱和迹象
- 最优配置：**Transformer 深度 > 宽度**（更深的 DiT 效果更好）

### Classifier-free Guidance

- 训练时随机 dropout 条件（10-20%）
- 推理时：`ε̂ = ε_uncond + w * (ε_cond - ε_uncond)`
- w 越大，图像质量越高但多样性降低

## 与已有 Wiki 的连接

- 来源：DDPM（扩散模型原理）→ ViT（Transformer 进入视觉）→ DiT（扩散+Transformer）
- 关联概念：[[扩散模型原理]]、[[Flow Matching]]、[[Vision Transformer (ViT)]]
- 关联实体：[[Seedream 系列模型]]（DiT backbone）、[[OminiControl]]（DiT 控制）
- 所在主题：[[Wiki/Topics/扩散模型与 Flow Matching 基础]]、[[Wiki/Topics/扩散模型图像编辑与生成]]

## 深度分析

### UNet 为什么被替换不是技术必然，而是 Scaling Law 的引力

UNet 的卷积归纳偏置在中小规模训练中是优势——它天然知道像素之间有空间关系。但随着数据量和计算预算的指数增长，**归纳偏置从"资产"变成了"负债"**：CNN 的局部感受野限制了模型从海量数据中学习长程依赖。

DiT 的成功本质上不是架构创新的胜利，而是 **Scaling Law 跨架构迁移的胜利**：Transformer 的缩放行为已被 LLM 充分验证，扩散社区利用了这个已知的"免费午餐"。

### adaLN-Zero 的真正价值被低估

多数讨论集中在"DiT 用 Transformer 替换 UNet"，但忽略了 adaLN-Zero 的巧妙之处。它解决了 Transformer 用于扩散时的一个关键问题：**如何在不去除条件信息的情况下去噪**。传统的 cross-attention 条件注入在深层会"遗忘"条件，而 adaLN 在每个 block 都重新注入条件，配合零初始化实现稳定的训练启动。这是一种"便宜但有效"的设计智慧。

### 一个悬而未决的问题

DiT 的设计空间远未被充分探索。当前的 DiT 基本上是将 ViT 的 block 原样搬过来加 adaLN，但扩散任务和分类任务的信息处理需求完全不同——去噪是逐像素精细化，分类是全局抽象。是否存在**扩散专用的 Transformer block 设计**，仍有大量探索空间。

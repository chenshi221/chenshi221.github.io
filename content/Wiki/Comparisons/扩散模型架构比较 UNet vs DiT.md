---
type: comparison
status: active
created: '2026-04-30'
updated: '2026-04-30'
tags:
  - 扩散模型
  - UNet
  - DiT
  - 架构比较
  - 图像生成
sources:
  - '[[Wiki/Sources/DDPM 扩散模型奠基论文]]'
  - '[[Wiki/Sources/FLUX.1 Kontext 上下文编辑]]'
  - '[[Wiki/Sources/Seedream 4.0 多模态图像生成]]'
  - '[[Wiki/Sources/OminiControl]]'
confidence: medium
---
# 扩散模型架构比较：UNet vs DiT

## 比较概述

扩散模型的核心是**去噪骨干网络**。2020 年 DDPM 奠定了扩散模型的数学框架，但骨干架构的演变（UNet → DiT）带来了质的变化。这一比较梳理两条路线的设计哲学、性能特征和适用场景。

## 架构对比

### UNet（卷积扩散时代）

代表：DDPM (2020), Stable Diffusion 1/2/3, ControlNet, InstructPix2Pix

**结构特征：**
- 对称的编码器-解码器结构
- 逐层下采样（编码）→ 瓶颈 → 逐层上采样（解码）
- 跳跃连接（skip connections）连接同尺度编码器和解码器层
- 基础单元：ResBlock + Self-Attention + Cross-Attention

**关键设计：**
- 时间步 t 通过 embedding 注入 ResBlock（如 GroupNorm 的 scale/shift）
- 文本条件通过 Cross-Attention 注入
- 空间条件（如 ControlNet）通过额外的编码器分支注入

### DiT（Transformer 扩散时代）

代表：FLUX, SD3 (MMDiT), Seedream 4.0, Sora, OminiControl

**结构特征：**
- Patchify → Transformer blocks → unpatchify
- 无下采样/上采样（纯 Transformer）
- 无跳跃连接（全局自注意力替代了局部到全局的信息流）
- 基础单元：Self-Attention + FFN + adaLN-Zero

**关键设计：**
- 时间步 t 和条件 c 通过 adaLN-Zero 注入每个 block
- 全局自注意力处理所有 patch token 之间的关系
- 序列化处理：图像被展平为 1D token 序列

### 混合方案：MMDiT（Multi-Modal DiT）

SD3 和 FLUX 采用的折中方案：
- 对图像 token 和文本 token 使用**独立的注意力权重**，但共享 FFN
- 图像 token 之间做 self-attention，文本 token 之间也做 self-attention
- 两者通过交叉注意力或序列拼接交互
- 保留了 DiT 的 Transformer 结构，但为多模态做了适配

## 维度对比

| 维度 | UNet | DiT | 分析 |
|------|------|------|------|
| **归纳偏置** | 强（局部性、平移等变、多尺度） | 弱（仅位置编码） | UNet 小数据更好，DiT 大数据更好 |
| **可扩展性** | 有限（卷积的 scaling 饱和较早） | 强（已验证 scaling law） | DiT 是"规模化时代的正确选择" |
| **分辨率灵活性** | 中（受下采样因子约束） | 高（patch size 可调） | DiT 更容易适配不同分辨率 |
| **训练效率（小模型）** | 高（卷积快、通信少） | 低（注意力 O(n²) 在大分辨率时贵） | UNet 在 < 2B 参数时仍有优势 |
| **训练效率（大模型）** | 低（卷积参数效率低） | 高（注意力参数效率高） | DiT 在 > 2B 参数时明显领先 |
| **长程依赖** | 弱（仅通过深层传递） | 强（每层全局感受野） | DiT 对全局结构理解更好 |
| **空间控制** | 方便（ControlNet, 特征注入） | 需要适配（OminiControl 等） | UNet 在精细控制上更成熟 |
| **多模态融合** | Cross-Attention（不自然） | 序列拼接（自然） | DiT 融合多模态更统一 |
| **推理速度** | 快（卷积优化成熟） | 中（FlashAttention 有改善） | 两者差距在缩小 |
| **生态成熟度** | 极高（LoRA, ControlNet 等） | 中（快速发展中） | UNet 生态仍更丰富 |

## 关键拐点

### UNet 为什么统治了 2020-2023

1. **卷积的工程积累**：ResNet/UNet 有 10+ 年的优化历史
2. **SD 1.5/2.1 的巨大成功**：社区在 UNet 上建立了庞大的工具链和知识库
3. **ControlNet 的诞生**：让 UNet 具备了强大的空间控制能力
4. **LoRA 的轻量级适配**：UNet + LoRA 成为最灵活的定制方案

### DiT 为什么在 2024 年崛起

1. **Sora 证明了 DiT 的可扩展性上限**（视频生成的 DiT 远超 UNet）
2. **FLUX 证明了 DiT 的图像质量可以超越 UNet**
3. **OminiControl 证明了 DiT 可以做到极简控制**（仅 0.1% 额外参数）
4. **Seedream 4.0 证明了 DiT 可以做多模态统一**
5. **社区快速跟进**：DiT 的 LoRA 生态在 2024 年迅速建立

### UNet and DiT 会长期共存吗？

**短期（2025-2026）：DiT 主导，UNet 过渡**

- 新模型几乎都是 DiT（FLUX, SD3.5, Seedream 3/4）
- 但 SD 1.5/XL 的庞大 UNet 生态仍在使用（惯性）

**中期（2027-2028）：DiT 成为唯一标准**

- 当 UNet 生态的开发者逐渐迁移到 DiT
- DiT 的控制工具（ControlNet-DiT, OminiControl）成熟

**长期：UNet 的某些设计可能回归**

- 多尺度处理在视觉任务中有天然优势（不同信息在不同尺度上）
- 未来可能出现 DiT + 多尺度特征的混合架构
- 这不是 UNet 的回归，而是其核心思想（多尺度）在 Transformer 框架下的重新表达

## 深层分析

### 归纳偏置的"资产到负债"转换

UNet 的多尺度卷积结构在 2020 年是巨大优势——它让模型在有限的训练数据下快速收敛。但到 2024 年，当训练数据从 LAION-400M 扩展到数十亿级别时，UNet 的归纳偏置从"帮助"变成了"限制"：

- 卷积的局部感受野限制了模型学习全局语义关系
- 固定的下采样/上采样结构限制了分辨率的灵活性
- 多尺度特征虽然理论上有优势，但在大规模训练中重要性下降

这符合深度学习的一个普遍规律：**数据量越大，手工设计的结构（归纳偏置）越不值钱，通用的计算结构（自注意力）越值钱**。

### 一个被忽略的问题：DiT 真的是"更好"还是只是"更大"？

目前 DiT 的成功有可能主要是**规模化**的结果（DiT 模型普遍比 UNet 大），而非架构本身的优越性。一个公平的比较应该是"同等 FLOPs 下 DiT vs UNet"，但这样的实验几乎没有人做，因为 DiT（15B+ FLOPs）和 UNet（通常 < 5B FLOPs）所在的 scaling 区间不同。

如果有一天我们发现，把同样的 FLOPs 砸进一个优化过的 UNet，性能和 DiT 差不多，那今天关于"DiT 优于 UNet"的叙事就需要重新审视。

## 与已有 Wiki 的连接

- 关联概念：[[扩散模型原理]]、[[DiT 扩散 Transformer]]、[[Flow Matching]]
- 关联实体：[[FLUX]]、[[Seedream 系列模型]]、[[ControlNet]]
- 所在主题：[[Wiki/Topics/扩散模型与 Flow Matching 基础]]

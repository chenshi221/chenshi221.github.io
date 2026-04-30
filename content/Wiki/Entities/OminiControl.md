---
type: entity
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - OminiControl
tags:
  - image-generation
  - DiT
  - control
  - diffusion-transformer
  - minimal
sources:
  - '[[Wiki/Sources/OminiControl]]'
---

# OminiControl

## 基本信息

| 属性 | 值 |
|------|-----|
| 全称 | OminiControl: Minimal and Universal Control for Diffusion Transformer |
| 作者 | Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, Xinchao Wang |
| 机构 | 新加坡国立大学 (NUS) |
| 年份 | 2024 (arXiv 2411.15098) |
| 类型 | DiT 架构的极简条件控制框架 |

## 核心创新

重新思考 Diffusion Transformer (DiT) 时代的条件控制：**利用 DiT 已有的能力做控制，而非像 ControlNet 那样外挂一个庞大的控制分支**。

```
ControlNet 思路：锁定 UNet → 复制编码器 → 训练控制分支 → 加入主支
OminiControl 思路：条件图像 → 复用 DiT 自己的 VAE + Transformer → 仅 0.1% 额外参数
```

## 三大技术贡献

### 1. 极简架构（仅 0.1% 额外参数）
- 条件图像通过 DiT 自己的 VAE encoder 编码
- 条件 token 与噪声图像 token 在同一 Transformer 中处理
- 不需要复制整个编码器分支（ControlNet 的做法）

### 2. 统一序列处理
```
序列 = [条件图像 token] + [噪声图像 token] + [文本 token]
       ←─── 同一个 Transformer 统一处理 ───→
```
这种设计使得条件信息可以自然地与生成内容交互。

### 3. 动态位置编码
- 适配两类控制任务：
  - **空间对齐**（如 Canny edge、深度图）：条件和生成图像空间位置对应
  - **非对齐**（如主体驱动）：条件只是参考图，不需要空间对齐

## 与 ControlNet 的深度对比

| 维度 | ControlNet | OminiControl |
|------|-----------|--------------|
| **基座架构** | UNet (SD) | DiT (FLUX 等) |
| **额外参数** | ~361M (复制编码器) | ~0.1% (复用已有) |
| **控制信号注入** | Zero convolution → 逐层注入 | 统一序列 → attention 自然交互 |
| **设计哲学** | 锁定 + 复制 + 外挂 | **融入已有架构，做最小的加法** |
| **控制类型** | 空间条件为主（edge, depth, pose） | 空间 + 主体驱动（subject-driven） |
| **生态成熟度** | 🔥🔥🔥🔥🔥 社区海量预训练模型 | 🔥 起步阶段 |
| **适用范围** | SD 生态（SD1.5, SDXL） | DiT 生态（SD3, FLUX） |

## Subjects200K 数据集

- 用 DiT 自身合成的身份一致性图像对
- 每个主体（如特定人物、物品）有多张不同姿态/背景的图像
- 训练模型在生成时保持主体身份一致

## 在控制架构演进中的位置

```mermaid
UNet 时代                          DiT 时代
ControlNet (2023) ──────────────→ OminiControl (2024)
  └─ Zero convolution               └─ Unified sequence
  └─ Lock + Copy                     └─ Reuse VAE + Transformer
  └─ 社区生态王者                     └─ 极简哲学

关键转折：从"外挂控制模块"到"让基座模型自己学会控制"
```

## 我的判断

OminiControl 代表了控制架构的**正确演进方向**——不是"给模型加控制能力"，而是"让模型本身就能被控制"。0.1% 的参数意味着控制能力几乎免费。

但 ControlNet 的成功不仅在于架构——更在于**社区贡献的海量预训练条件模型**（数千个 ControlNet 模型覆盖各种条件类型）。OminiControl 能否达到同样的生态丰富度，取决于 DiT 社区的发展速度。

实际上，两者可能长期共存：ControlNet 统治 SD 生态，OminiControl 统治 DiT 生态。

## 在 Wiki 中的关联

- 来源：[[Wiki/Sources/OminiControl]]
- 主题：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 相关：[[Wiki/Entities/ControlNet]]（UNet 时代的控制范式）
- 比较：[[Wiki/Comparisons/统一多模态模型架构比较]]（DiT 时代的控制讨论）

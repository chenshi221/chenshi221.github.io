---
type: source
status: processed
source_file: "[[Clippings/Adding Conditional Control to Text-to-Image Diffusion Models.md]]"
title: "Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)"
site: "arXiv (2302.05543)"
author: "Lvmin Zhang, Anyi Rao, Maneesh Agrawala"
published: "2023-02"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [image-generation, control, diffusion, ControlNet, foundational]
aliases: [ControlNet]
---

# ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models

Stanford，2023。**可控生成的里程碑工作**。

## 核心结论

- 提出 **ControlNet** 架构，为大规模预训练扩散模型添加空间条件控制（边缘、深度、分割、姿态等）。
- 锁定预训练模型权重，复用其深度编码层作为骨干，训练可训练副本 + **零卷积（zero convolution）** 逐步从零增长参数。
- 支持小数据集（<50K）和大数据集（>1M）训练，鲁棒性强。
- 支持单条件、多条件、有/无文本 prompt。

## 关键设计

### Zero Convolution
- 初始化为零的 1×1 卷积层
- 训练初期不引入噪声，参数从零逐步增长
- 保护预训练权重的完整性

### 锁定 + 复制架构
- 锁定原始 SD 编码器 → 保留大规模预训练知识
- 复制可训练副本 → 学习条件控制信号
- 零卷积连接副本到原始解码器

### 支持的条件类型
Canny 边缘、HED 边界、深度图、法线图、人体姿态、语义分割、涂鸦等

## 历史地位

- 被引超千次，成为扩散模型可控生成的标准范式
- 催生了 T2I-Adapter、IP-Adapter、UniControl 等大量后续工作
- ControlNet 的 "锁定基座 + 轻量适配" 思路被 EmoEdit 的 Emotion adapter 等沿用

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]
- EmoEdit 引用了 ControlNet 作为对比 baseline，Emotion adapter 的设计哲学类似（锁定基座，训练 adapter）
- Step1X-Edit 等 MLLM 编辑方法也使用了类似的空间控制思路

## 局限性

- 依赖预定义条件类型（边缘、深度等），无法处理抽象条件（情感、风格描述）
- 每种条件需单独训练一个 ControlNet

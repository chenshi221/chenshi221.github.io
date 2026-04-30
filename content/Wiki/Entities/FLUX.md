---
type: entity
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - FLUX.1
  - Flux
tags:
  - FLUX
  - 扩散模型
  - Flow Matching
  - 图像生成
  - DiT
  - Black Forest Labs
sources:
  - '[[Wiki/Sources/FLUX.1 Kontext 上下文编辑]]'
  - '[[Wiki/Sources/Seedream 2.0 中英双语图像生成]]'
  - '[[Wiki/Sources/Seedream 3.0 技术报告]]'
confidence: high
---
# FLUX

## 简介

**FLUX**（也称 FLUX.1）是 Black Forest Labs（BFL，Stable Diffusion 原团队）于 2024 年发布的文本到图像生成模型系列。它基于 **Flow Matching** 和 **DiT（Diffusion Transformer）** 架构，在图像质量和 prompt 跟随方面显著超越了当时的 SD3 和 Midjourney，成为 2024 年下半年最受关注的开源图像生成模型之一。

BFL 团队包括 Robin Rombach、Andreas Blattmann 等 Stable Diffusion 的核心作者，FLUX 可以看作是他们离开 Stability AI 后的"重做版"——吸收了 Diffusion 社区的全部经验教训，从零构建。

## 技术架构

### Flow Matching Backbone

- 采用 rectified flow 的 ODE 路径（从噪声到图像的直线插值）
- 相比 DDPM 的随机微分方程，Flow 路径步数更少、训练更稳定

### DiT Backbone

- 使用标准的 DiT 架构作为去噪网络
- 条件注入方式：T5 文本编码器 + CLIP 文本编码器的双文本条件
- 相比 SD3 的 MMDiT（多模态 DiT），FLUX 的设计更简洁

### 双文本编码器

- **T5-XXL**：提供深层语义理解
- **CLIP-L**：提供视觉-语言对齐
- 双编码器配合：CLIP 提供粗略的语义对齐，T5 提供精细的文本理解

## 模型变体

| 变体 | 定位 | 特点 |
|------|------|------|
| **FLUX.1 [pro]** | 商用 API | 最高质量，通过 API 访问 |
| **FLUX.1 [dev]** | 开源研究 | 非商用许可，蒸馏版 |
| **FLUX.1 [schnell]** | 快速推理 | Apache 2.0 开源，4 步采样，速度极快 |

### FLUX.1 Kontext

FLUX 的一个扩展方向：通过**序列拼接**将参考图像和编辑指令同时输入 DiT，实现**上下文感知的 in-context 图像编辑**。不需要额外控制网络（如 ControlNet），编辑质量高且架构统一。

关键特性：
- 参考图像 + 文本指令 → 编辑图像
- 支持多图参考、风格迁移、ID 保持等多种任务
- 来源：[[Wiki/Sources/FLUX.1 Kontext 上下文编辑]]

## FLUX 的定位与影响

### 相比 SD3

- SD3 的发布遭遇了严重的人体结构生成问题（争议很大）
- FLUX 解决了这些问题（prompt 跟随、人体结构、文字渲染）
- 但 FLUX [dev] 的非商用许可限制了其开源生态的扩展

### 相比 Seedream 系列

- Seedream 2.0/3.0 在中英双语、美学质量上有独特优势
- FLUX 在开源社区的渗透更深（大量社区 LoRA 和 ControlNet 变体）
- Seedream 4.0 的多模态统一能力是 FLUX 尚不具备的

### 相比 Midjourney

- Midjourney 是闭源美学之王，FLUX 是开源技术标杆
- FLUX 的技术透明性使其成为研究社区的首选 backbone

## 开源生态

FLUX 催生了庞大的社区生态：
- 大量社区训练的 LoRA（角色、风格、概念）
- 多种 ControlNet 变体
- 视频生成扩展（AnimateDiff FLUX 等）
- ComfyUI 优先支持

## 与已有 Wiki 的连接

- 关联概念：[[Flow Matching]]、[[DiT 扩散 Transformer]]、[[扩散模型原理]]
- 关联实体：[[Seedream 系列模型]]
- 所在主题：[[Wiki/Topics/扩散模型与 Flow Matching 基础]]

## 批判性评估

### FLUX 的真正创新是"重新做的勇气"

FLUX 架构上并没有根本性突破——Flow Matching 来自 Meta，DiT 来自 Peebles & Xie。但 BFL 团队的价值在于：(1) 有了 SD 系列的全部经验教训，(2) 不受 Stability AI 内部政治和商业压力的影响，(3) 从零重新设计训练配方。

FLUX 告诉我们一个道理：**在 AI 领域，有时候"知道不要做什么"比"发明新东西"更重要**。

### 开源策略的两难

FLUX [dev] 的非商用许可（非 Apache 2.0）限制了其成为"行业标准 backbone"。相比之下，Seedream 4.0、SD3 Medium 等 Apache 2.0 模型在商业部署上更有吸引力。FLUX 在"开源影响力"和"商业可持续性"之间的张力，是所有开源 AI 公司面临的经典困境。

### 从 SD 到 FLUX 的产业启示

SD 团队离开 Stability AI 后创建的 FLUX 反而超越了 SD3，这是一个耐人寻味的"分叉叙事"：有时创新不是来自持续积累，而是来自**放下包袱重新开始**。FLUX 在 2024 年下半年的成功，某种程度上也是对 Stability AI 管理混乱的间接批评。

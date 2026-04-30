---
type: source
status: processed
source_file: "[[Clippings/OminiControl Minimal and Universal Control for Diffusion Transformer.md]]"
title: "OminiControl: Minimal and Universal Control for Diffusion Transformer"
site: "arXiv (2411.15098v6)"
author: "Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, Xinchao Wang"
published: "2024-11"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [image-generation, DiT, control, diffusion-transformer, minimal]
aliases: [OminiControl]
---

# OminiControl: Minimal and Universal Control for Diffusion Transformer

新加坡国立大学 (NUS)，2024。

## 核心结论

- 重新思考图像条件如何集成到 Diffusion Transformer (DiT) 架构。
- 三大创新：(1) **极简架构**：仅 0.1% 额外参数，利用 DiT 自身 VAE encoder + transformer blocks；(2) **统一序列处理**：条件 token 与图像 token 统一处理；(3) **动态位置编码**：适配空间对齐和非对齐控制任务。
- 引入 **Subjects200K** 数据集：用 DiT 自身合成的身份一致性图像对。

## 与 ControlNet 的对比

| 维度 | ControlNet | OminiControl |
|------|-----------|--------------|
| 基座 | UNet (SD) | DiT |
| 额外参数 | 较多（复制编码器） | 0.1% |
| 控制类型 | 空间条件为主 | 空间 + 主体驱动 |
| 设计哲学 | 锁定 + 复制 | 融入已有架构 |

## 历史意义

- 代表了从 UNet-based ControlNet 到 DiT-based 控制的架构演进
- 与 SD3/FLUX 等 DiT 架构的普及同步
- "Minimal but universal" 的设计哲学影响后续工作

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]
- DiT 控制的新范式，与 [[Wiki/Sources/ControlNet]] 互补

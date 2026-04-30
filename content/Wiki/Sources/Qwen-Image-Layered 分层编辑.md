---
type: source
status: processed
source_file: >-
  [[Clippings/Qwen-Image-Layered Towards Inherent Editability via Layer
  Decomposition]]
title: 'Qwen-Image-Layered: Towards Inherent Editability via Layer Decomposition'
site: arxiv
author: Shengming Yin et al. (Alibaba)
published: '2025'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - Qwen-Image
  - 分层编辑
  - RGBA
  - 图层分解
  - 图像编辑
  - 一致性
aliases:
  - Qwen-Image-Layered
---
# Qwen-Image-Layered 分层编辑

## 核心结论

Qwen-Image-Layered（阿里通义）提出了一种全新的图像编辑范式：将单张 RGB 图像端到端分解为多个语义解耦的 RGBA 层，每层可独立操作而不影响其他内容，从根本上解决了传统编辑中的语义漂移和几何错位问题。这是一个从"编辑像素"到"编辑图层"的根本性范式转变。

## 关键事实

- 作者：Shengming Yin 等（HKUST(GZ) & 阿里巴巴），2025
- 核心三板斧：
  1. RGBA-VAE：统一 RGB 和 RGBA 图像的潜在空间，消除输入（RGB）和输出（RGBA 层）的分布差距
  2. VLD-MMDiT：可变层数分解架构，支持每次分解不同数量的图层
  3. 多阶段训练：Text-to-RGB → Text-to-RGBA → Text-to-Multi-RGBA → Image-to-Multi-RGBA，渐进适配
- Layer3D RoPE：引入层维度位置编码，区分不同层
- 数据：从真实 PSD（Photoshop）文件提取和标注多层图像，解决高质量多层数据稀缺问题
- 性能：在 Crello 数据集上显著超越 LayerD、Accordion 等 baseline

## 方法或论证路径

- 问题是"图像表示本身"而非"模型设计"——传统栅格图像将所有内容纠缠在单一画布上
- 解决方案：将图像表示为一堆语义解耦的 RGBA 层
- 与 mask-based 编辑的区别：mask 方法无法处理遮挡和软边界，图层方法物理隔离编辑区域
- 保持了 Qwen-Image（MMDiT + Flow Matching）的生成质量基础
- 不仅支持编辑，还支持多层图像合成（Text-to-Multi-RGBA）

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]——代表了图像编辑的新范式
- 对比：与 InstructPix2Pix、AnyEdit、GoT 等编辑方法的根本思路不同（图层 vs 像素/潜在空间修改）
- 补充：为统一编辑模型提供了新的技术路线——"编辑友好的表示"优于"更强的编辑模型"

## 可能的矛盾或待核实点

- 自动分解的质量上限：对于复杂场景（遮挡、半透明、光影交织），自动分解是否始终可靠？
- 与 InstructPix2Pix 类方法的用户体验对比：用户是否愿意接受"先分解再编辑"的额外步骤？

## 后续问题

- RGBA 层分解能否推广到视频编辑（时序一致的多层分解）？
- 该范式与 Agent 编辑（Mind-Brush、VisionCreator）能否结合？

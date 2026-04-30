---
type: comparison
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - Embedding 模型比较
tags:
  - embedding
  - comparison
  - multimodal
sources:
  - Wiki/Sources/Magic-MM-Embedding
  - Wiki/Sources/ObjEmbed
  - Wiki/Sources/RzenEmbed
  - Wiki/Sources/SAIL-Embedding
confidence: high
---
# 多模态 Embedding 模型比较

## 概览

本文比较 4 个近期 MLLM 基多模态 Embedding 模型：Magic-MM-Embedding、ObjEmbed、RzenEmbed、SAIL-Embedding。它们代表了该方向的效率优化、细粒度表示、模态扩展和工业落地的四个维度。

## 核心维度对比

| 维度 | Magic-MM-Embedding | ObjEmbed | RzenEmbed | SAIL-Embedding |
|------|-------------------|----------|-----------|---------------|
| 基座模型 | InternVL3 | Qwen3-VL | Qwen2-VL | SAIL 系列 |
| 支持模态 | 文本 + 图像 | 文本 + 图像（物体级） | 文本 + 图像 + 视频 + 文档 | 视觉 + 文本 + 音频 |
| 核心创新 | 视觉 token 压缩 75% | 物体级双 embedding（语义+IoU） | Hardness-weighted loss + 假负消除 | 全模态 + 推荐系统蒸馏 |
| 粒度 | 图像级 | 物体级 | 图像/视频/文档级 | 元素/视频级 |
| 评测基准 | MMEB 35 | 18 benchmark（检测+检索） | MMEB / MMEB-V2 | 学术+抖音线上 |
| 训练规模 | 16M 样本 | 1.3M 样本 | 两阶段 | >10B 样本 |
| 应用场景 | 通用检索 | 目标检测+局部检索 | 跨模态检索 | 推荐系统 |
| Token 效率 | 高（1/4 token） | 中（<2000/图） | 中 | 中 |
| 产出形式 | 开源 | 开源 | 开源 | 技术报告 |

## 技术路线对比

### 效率优化

- **Magic-MM-Embedding** 是唯一专门解决 token 效率的工作。通过参数无关的双线性插值压缩 75% 视觉 token，配合三阶段渐进训练恢复性能。
- 其他三个模型均未专门考虑 token 压缩问题。

### 表示粒度

- **ObjEmbed** 独树一帜，支持物体级（region-level）表示，生成语义 embedding + IoU quality embedding。
- 其余三个模型均为图像/视频/文档级的全局表示。

### 模态覆盖

- **SAIL-Embedding** 模态覆盖最广（视觉+文本+音频），面向抖音短视频场景。
- **RzenEmbed** 覆盖文本+图像+视频+视觉文档，适合通用检索。
- **Magic-MM-Embedding** 和 **ObjEmbed** 主要为图像+文本。

### 训练策略

- **SAIL-Embedding** 数据规模最大（>100 亿样本），包含推荐系统特有的 ID-to-Item 和 Sequence-to-Item 蒸馏。
- **RzenEmbed** 的 hardness-weighted loss 和 false negative 消除值得其他模型借鉴。
- **Magic-MM-Embedding** 的 MLLM-as-Judge 数据筛选策略可迁移到其他模型。

## 互补性

四个模型并非直接竞争，而是从不同角度推进多模态 Embedding：
- 需要高效推理：Magic-MM-Embedding
- 需要物体级检索：ObjEmbed
- 需要视频/文档支持：RzenEmbed
- 需要全模态工业部署：SAIL-Embedding

## 相关页面

- [[Wiki/Topics/多模态 Embedding 与检索]]
- [[Wiki/Concepts/多模态 Embedding 模型]]
- [[Wiki/Sources/Magic-MM-Embedding]]
- [[Wiki/Sources/ObjEmbed]]
- [[Wiki/Sources/RzenEmbed]]
- [[Wiki/Sources/SAIL-Embedding]]

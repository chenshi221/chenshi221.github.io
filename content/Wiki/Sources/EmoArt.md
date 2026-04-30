---
type: source
status: processed
source_file: "[[Clippings/EmoArt A Multidimensional Dataset for Emotion-Aware Artistic Generation.md]]"
title: "EmoArt: A Multidimensional Dataset for Emotion-Aware Artistic Generation"
site: "arXiv (2506.03652v1)"
author: "Cheng Zhang, Hongxia Xie, Bin Wen, Songhan Zuo, Ruoxuan Zhang, Wen-Huang Cheng"
published: "2025-06"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [emotion, dataset, art, diffusion, benchmark, valence-arousal]
aliases: [EmoArt, EmoArt-130k]
---

# EmoArt: A Multidimensional Dataset for Emotion-Aware Artistic Generation

吉林大学 + 国立台湾大学，2025。

## 核心结论

- 提出 **EmoArt**，目前最全面的情感标注艺术数据集之一：**132,664 张艺术作品**，覆盖 **56 种绘画风格**（印象派、表现主义、抽象艺术等），横跨东西方传统。
- 每张图像含五层结构化标注：客观场景描述、五个关键视觉属性（笔触、构图、色彩、光线、线条）、二值 arousal-valence 标签、12 类情感类别、潜在艺术治疗效果。
- 用 GPT-4o 辅助标注 + 人类验证（5,600 张采样，情感维度一致性 91.47%）。
- 基于 EmoArt 微调 FLUX.1-dev（LoRA）后在主观属性对齐上全面优于所有基线。

## 数据集特点

### 标注框架

| 维度 | 内容 |
|------|------|
| 场景描述 | 客观图像内容描述 |
| 视觉属性 | 笔触、色彩、构图、光线、线条 |
| Arousal-Valence | 二值高低标签（Russell 环状模型） |
| 情感类别 | 12 类（Calm, Excited, Contentment, Alarmed, Sad 等） |
| 治疗效果 | 潜在艺术治疗应用 |

### 数据分布

- **情感偏向正向**：87.93% 正向 valence，76.41% 低 arousal
- **主导情感**：Calm (55.95%)、Excited (15.50%)、Contentment (15.35%)
- **风格差异显著**：现实主义/浪漫主义 → 平静正向；表现主义/超现实主义 → 高 arousal + 负面情感
- **文化差异**：中国传统绘画 99.76% 低 arousal、99.95% 正向（Calm 89.42%）

### 语言多样性

在所有对比数据集（Flickr30K、ArtEmis、COCO Captions）中，EmoArt 的 TTR、MTLD、Entropy 均最高，描述更丰富。

## 生成模型 Benchmark

- 评估 7 个模型：FLUX.1-dev、FLUX.1-schnell、SDXL、SD3.5、PixArt-sigma、Playground、Openjourney
- FLUX.1-dev-finetuned（LoRA 微调）在笔触 (0.6388)、色彩 (0.6974)、构图 (0.6698) 等主观指标上全面最优
- 传统像素指标（FID/PSNR/LPIPS/SSIM）无法完全捕捉情感和感知质量 → 属性对齐评估作为新视角

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/情感计算与图像生成]]、[[Wiki/Concepts/Valence-Arousal 情感模型]]
- 与 [[Wiki/Sources/EmoEdit]] 互补：EmoEdit 做编辑，EmoArt 做数据集和生成 benchmark
- 情感标注维度比 EmoSet 更丰富（增加视觉属性 + 治疗效果）

## 后续问题

- EmoArt 的情感分布（高度偏向 calm/positive）是否会影响训练模型的多样性？
- 56 种风格的情感特征差异能否用于风格迁移中的情感控制？

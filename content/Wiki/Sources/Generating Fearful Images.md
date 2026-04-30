---
type: source
status: processed
source_file: "[[Clippings/Generating Fearful Images Investigating Potential Emotional Biases in Image-Generation Models.md]]"
title: "Generating Fearful Images: Investigating Potential Emotional Biases in Image-Generation Models"
site: "arXiv (2411.05985v2)"
author: "Maneet Mehta, Cody Buntain"
published: "2024-11"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [emotion, bias, AI-safety, image-generation, evaluation]
aliases: [Generating Fearful Images, AI emotional bias]
---

# Generating Fearful Images: Investigating Potential Emotional Biases in Image-Generation Models

独立研究员 + 马里兰大学，2024。

## 核心结论

- AI 生成图像存在显著的情感偏差：**无论 prompt 是什么，生成图像常偏向负面情感（特别是 fear）**。
- 比较了三种图像情感识别方法：传统监督学习（ViT 微调最优）、零样本 VLM、跨模态 auto-captioning。
- Google ViT 在图像情感识别上显著优于零样本和 caption 方法。
- 跨模态对比：文本 prompt 的情感 vs 生成图像的情感 → **生成图像系统性地更负面**。

## 关键发现

1. **情感识别方法比较**：微调 ViT > 零样本 VLM > auto-captioning
2. **情感偏差方向**：AI 生成图像偏向 fear（恐惧），独立于 prompt 情感
3. **社会影响**：这种偏差可能在数字空间中放大负面情感内容

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/情感计算与图像生成]]、[[Wiki/Concepts/AI 生成图像的情感偏差]]
- 这是一个**元层次**的研究：不提出新的生成方法，而是审视现有生成模型的情感安全问题
- 与 [[Wiki/Sources/EmoEdit]]、[[Wiki/Sources/EmotiCrafter]] 等方法形成对照：这些方法试图精确控制情感，但底层模型本身可能存在未被察觉的情感偏差

## 限制与后续问题

- 仅评估了 8 类离散情感，未涉及连续情感空间
- 情感偏差的来源是什么？训练数据、模型架构、还是安全过滤？
- 如何在保持安全对齐的同时减少不必要的情感偏差？

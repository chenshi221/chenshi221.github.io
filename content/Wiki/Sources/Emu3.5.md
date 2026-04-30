---
type: source
status: processed
source_file: '[[Clippings/Emu3.5 Native Multimodal Models are World Learners.md]]'
title: 'Emu3.5: Native Multimodal Models are World Learners'
site: arXiv
author: BAAI et al.
published: '2025'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - native-multimodal
  - world-model
  - next-token-prediction
  - DiDA
aliases:
  - Emu3.5
---
# Emu3.5: Native Multimodal Models are World Learners

## 核心结论

- Emu3.5 是一个**原生多模态世界模型**，通过统一的 next-token prediction 在视觉-语言交错数据上做端到端预训练（超 10 万亿 token），实现图文交错的输入和输出。
- 核心创新：**Discrete Diffusion Adaptation (DiDA)**——将逐 token 自回归解码转换为双向并行预测，单图推理加速约 20 倍且不损失性能。
- 展现出多种原生多模态能力：长时间跨度的视觉-语言生成、Any-to-Image (X2I) 生成、复杂含文本图像生成，以及通用世界建模能力。

## 关键事实

- 作者来自 BAAI（智源研究院）、清华等。2025 年（arXiv:2510.26583）。
- 训练数据：来自互联网视频的序列帧 + 字幕，超 10 万亿 token 的图文交错语料。
- 大型 RL 后训练增强多模态推理和生成能力。
- 性能对标：图像生成和编辑性能与 Gemini 2.5 Flash Image（Nano Banana）可比，交错生成任务上表现更优。
- 开源发布：https://github.com/baaivision/Emu3.5。

## 方法亮点

- **原生多模态**：不是 VLM + Diffusion 的拼装方案，而是从头用 next-token prediction 统一预测视觉和语言。
- **DiDA**：在推理时将离散 token 的自回归生成转换为扩散式并行预测，是原生模型加速的关键突破。
- **世界模型**：从视频序列中学习时空一致的世界动态，支持开放环境下的具身操作（embodied manipulation）。

## 与现有 Wiki 的关系

- 是 [[Wiki/Concepts/原生多模态模型]] 的核心代表，与组装式方案（VLM+Diff）形成关键区分。
- 属于 [[Wiki/Topics/扩散模型图像编辑与生成]] 的统一多模态模型演进路线，与 [[Wiki/Entities/BAGEL]]、[[Wiki/Entities/Show-o2]] 等并列。
- DiDA 技术让原生多模态模型在推理效率上首次可与扩散模型竞争。

## 后续问题

- DiDA 的并行预测是否在所有场景下都等价于自回归？在高度依赖上下文的生成长文本时表现如何？
- 原生模型 vs 组装式方案的性能边界在哪里？什么任务上各自更适合？

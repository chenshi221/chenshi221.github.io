---
type: source
status: processed
source_file: >-
  [[Clippings/CLAP  Learning Audio Concepts From Natural Language
  Supervision.md]]
title: 'CLAP: Learning Audio Concepts From Natural Language Supervision'
site: arXiv
author: Benjamin Elizalde et al. (Microsoft)
published: 2022-06
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - contrastive-learning
  - audio
  - multimodal
  - zero-shot
aliases:
  - CLAP
---
# CLAP: Learning Audio Concepts From Natural Language Supervision

## 核心结论

- CLAP 将 CLIP 的**对比语言-图像预训练范式迁移到音频领域**，提出了 Contrastive Language-Audio Pretraining。
- 通过双编码器（音频编码器 + 文本编码器）+ 对比学习，将音频和文本描述映射到联合多模态空间，实现零样本音频分类。
- 仅使用 128K 音频-文本对训练（远少于 CLIP 的 400M 图文对），仍在 16 个下游任务上取得 SOTA 零样本性能。

## 关键事实

- 作者：Benjamin Elizalde, Soham Deshmukh 等（Microsoft）。
- 发表于 2022（arXiv:2206.04769，后发表于 ICASSP 2023）。
- 架构：Audio Encoder（CNN14/HTSAT） + Text Encoder（BERT/RoBERTa）→ 对比损失（CLIP 式 infoNCE），使配对音频-文本在嵌入空间中靠近。
- 涵盖 8 个领域：Sound Event Classification、Music、Speech 等。
- 在监督微调设置下也在 5 个任务上达到 SOTA。

## 方法亮点

- 将 CLIP 范式成功从视觉迁移到音频，验证了对比语言-感知预训练的跨模态通用性。
- 支持零样本推理：无需训练分类器，直接用文本查询任意声音类别。
- 训练数据效率极高，128K 对即取得强泛化能力。

## 与现有 Wiki 的关系

- 是 [[Wiki/Concepts/多模态对比学习]] 在音频领域的代表工作。
- 与视觉对比学习（CLIP、SigLIP）共享核心思想，见 [[Wiki/Sources/SigLIP 2]]。
- 扩展了"自然语言作为监督信号"的应用范畴。

## 后续问题

- 音频-文本对比学习的 scaling law 如何？更多数据（如 LAION-Audio-630K）能带来多大提升？
- 是否能像 CLIP 支持图像生成（如 stable diffusion）一样，支持文本到音频生成？

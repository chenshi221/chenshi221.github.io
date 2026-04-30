---
type: source
status: processed
source_file: "[[Clippings/Tuna-2 Pixel Embeddings Beat Vision Encoders for Multimodal Understanding and Generation.md]]"
title: "Tuna-2: Pixel Embeddings Beat Vision Encoders for Multimodal Understanding and Generation"
site: "arXiv (Meta AI, HKU, Waterloo)"
author: "Zhiheng Liu, Weiming Ren, Xiaoke Huang, Shoufa Chen, Yuren Cong et al."
published: "2026-04-27"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [unified-multimodal, encoder-free, pixel-space, tuna-2]
aliases: [Tuna-2]
---

# Tuna-2: Pixel Embeddings Beat Vision Encoders

## 核心结论

- Tuna-2 是一个**无编码器（encoder-free）** 的统一多模态模型，直接基于像素嵌入（pixel embeddings）进行理解和生成，完全摒弃了 VAE 和表示编码器。
- 证明**预训练的视觉编码器对多模态建模并非必要**，端到端像素空间学习是一条可扩展的路径。
- 无编码器设计在精细视觉感知任务上优于编码器变体 Tuna-R。

## 关键事实

- **架构**：单 Transformer decoder（基于 Qwen2.5-7B-Instruct），直接用 patch embedding 层处理图像，移除 VAE 和 SigLIP 等模块。
- **像素空间生成**：采用 JiT 的 $x$-prediction 和 $v$-loss 范式进行像素空间 flow matching。
- **掩码特征学习**：随机 mask 部分图像 patch，迫使模型在部分视觉观察下进行多模态推理，学习更鲁棒的视觉表示。
- **两阶段训练**：Stage 1 全模型预训练（550M 图文对，300K steps）；Stage 2 SFT（13M FineVision + 2M OmniEdit）。
- **受控对比实验**：Tuna-2（无编码器）vs Tuna-R（带 SigLIP2 编码器）→ 生成能力相当，理解能力 Tuna-2 更优，尤其是在细粒度感知任务上。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 关联：[[Wiki/Sources/BAGEL]]（BAGEL 使用双编码器 MoT；Tuna-2 走完全相反的无编码器路线）
- 关联：[[Wiki/Sources/UniWorld-V1]]（UniWorld 用 SigLIP 替代 VAE；Tuna-2 连表示编码器都去掉）

## 可能的矛盾或待核实点

- 无编码器设计在更大规模（>7B）模型上是否仍然有效？
- 像素空间建模的计算成本是否在更高分辨率下变得不可接受？

## 后续问题

- 无编码器范式和语义编码器范式，哪一个最终会主导统一多模态模型？
- Tuna-2 的 mask 策略是否与 MAE 有深层联系？

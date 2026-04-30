---
type: source
status: processed
source_file: "[[Clippings/Affective Image Editing Shaping Emotional Factors via Text Descriptions.md]]"
title: "AIEdiT: Affective Image Editing Shaping Emotional Factors via Text Descriptions"
site: "arXiv (2505.18699v1)"
author: "Peixuan Zhang, Shuchen Weng, Chengxuan Zhu, Binghao Tang, Zijian Jia, Si Li, Boxin Shi"
published: "2025-05"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [emotion, image-editing, affective-computing, MLLM]
aliases: [AIEdiT, Affective Image Editing]
---

# AIEdiT: Affective Image Editing Shaping Emotional Factors via Text Descriptions

北京大学 + 等，2025。

## 核心结论

- 提出 **AIEdiT**（Affective Image Editing using Text descriptions），通过文本描述自适应地塑造全图多个情感因子来唤起特定情感。
- 三大组件：(1) **连续情感谱 (Continuous Emotional Spectrum)** 表示通用情感先验；(2) **情感映射器 (Emotional Mapper)** 将视觉抽象的情感请求翻译为视觉具体的语义表示；(3) **MLLM 监督** 确保编辑结果唤起目标情感。
- 推理时策略性地扭曲视觉元素，再塑造相应情感因子。
- 引入大规模情感对齐数据集（文本-图像对）。

## 关键方法

1. **连续情感谱**：将情感表示为连续空间（vs 离散类别），提取细腻的情感需求
2. **情感映射器**：桥接"抽象情感需求 → 具体视觉元素"的语义鸿沟
3. **MLLM 监督训练**：用多模态大模型作为情感忠实度的评判标准
4. **编辑策略**：先扭曲 → 再塑形，分两阶段操控视觉元素和情感因子

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/情感计算与图像生成]]
- 连续情感谱设计与 [[Wiki/Sources/EmotiCrafter]] 的 V-A 空间思路相近但路径不同
- MLLM 监督训练与 [[Wiki/Sources/EmoEdit]] 的人工审核 + 自动指标形成互补
- 延续了作者之前 AIF (Affective Image Filter, ICCV 2023) 的工作线

## 后续问题

- 连续情感谱与 V-A 模型的关系是什么？是否兼容？
- MLLM 作为情感监督的可靠性如何量化？

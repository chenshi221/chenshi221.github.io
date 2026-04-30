---
type: source
status: processed
source_file: "[[Clippings/Seedream 2.0 A Native Chinese-English Bilingual Image Generation Foundation Model]]"
title: "Seedream 2.0: A Native Chinese-English Bilingual Image Generation Foundation Model"
site: "arXiv (2025)"
author: "ByteDance Seed Vision Team"
published: "2025"
processed: "2026-04-30"
updated: "2026-04-30"
tags: ["Seedream", "image-generation", "bilingual", "DiT", "flow-matching", "text-rendering"]
aliases: ["Seedream 2.0"]
confidence: high
---

# Seedream 2.0

## 核心结论

Seedream 2.0 是字节跳动的中英双语图像生成基础模型，针对现有模型（Flux、SD3.5、Midjourney）在模型偏见、文本渲染和对中国文化理解方面的不足进行优化。采用自研双语 LLM 作为文本编码器，能原生学习中文知识文化。Glyph-Aligned ByT5 实现字符级灵活文本渲染，Scaled RoPE 泛化到未训练分辨率。多阶段后训练（SFT+RLHF）进一步提升整体能力。

## 关键方法

1. **架构选择**：MMDiT（Multimodal DiT）处理图像和文本 token，双流/单流混合 block，自研双语 LLM 作为文本编码器。

2. **数据与标注**：构建强大数据系统实现知识整合，标注系统平衡准确性与丰富度。严格过滤策略移除带水印、文字叠加、马赛克的样本。

3. **Glyph-Aligned ByT5**：用于字符级文本渲染的编解码模型，支持中英文字符灵活生成。

4. **Scaled RoPE**：扩展旋转位置编码以适应未训练分辨率和宽高比。

5. **后训练流程**：Continuing Training (CT) -> Supervised Fine-Tuning (SFT) -> RLHF -> Prompt Engineering (PE)，另有 Refiner 超分阶段。

6. **模型加速**：量化（W4A8）将推理延迟减半。

## 与现有 Wiki 的关系

- 系列入口：[[Wiki/Entities/Seedream 系列模型]]
- 基于 Flow Matching：[[Wiki/Concepts/Flow Matching]]
- 扩散模型基础：[[Wiki/Concepts/扩散模型原理]]
- 串联脉络：[[Wiki/Topics/扩散模型与 Flow Matching 基础]]

## 后续问题

- Seedream 2.0 仍需 Refiner 处理高分辨率，3.0 解决了原生高分辨率输出问题。
- 复杂提示对齐、细粒度排版和美学质量是后续版本改进方向。

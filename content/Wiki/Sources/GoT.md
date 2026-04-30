---
type: source
status: processed
source_file: "[[Clippings/GoT Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing.md]]"
title: "GoT: Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing"
site: "arXiv (2503.10639v1)"
author: "Rongyao Fang, Chengqi Duan, et al."
published: "2025-03"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [image-generation, image-editing, reasoning, chain-of-thought, MLLM, diffusion]
aliases: [GoT, Generation Chain-of-Thought]
---

# GoT: Unleashing Reasoning Capability of MLLM for Visual Generation and Editing

CUHK MMLab + HKU + SenseTime + Shanghai AI Lab + THU + BUAA，2025。

## 核心结论

- 提出 **Generation Chain-of-Thought (GoT)**：在生成图像之前先进行显式语言推理，分析语义关系和空间布局。
- 将传统 text-to-image 转变为 **reasoning-guided framework**。
- 构建 **9M+ 样本**的 GoT 数据集，含详细推理链（语义-空间关系）。
- 统一框架：Qwen2.5-VL 生成推理链 → Semantic-Spatial Guidance Module 增强的端到端扩散模型。

## 关键创新

### Reasoning Chain 结构
- 输入：文本描述 → 推理分析语义关系 → 规划空间布局 → 生成图像
- 与传统方法的区别：不是直接从文本到像素，而是"先想再画"

### Semantic-Spatial Guidance Module
- 将推理链中的语义和空间信息注入扩散模型
- 语义引导：哪些物体在哪，什么属性
- 空间引导：物体的位置关系和布局

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]
- Reasoning 先行的思路与 [[Wiki/Sources/Mind-Brush]]（think-research-create）和 [[Wiki/Sources/VisionCreator]]（UTPC）一致
- 代表了从"端到端生成"到"推理-生成"的范式转变

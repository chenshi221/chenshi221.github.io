---
type: source
status: processed
source_file: "[[Clippings/WEAVE Unleashing and Benchmarking the In-context Interleaved Comprehension and Generation.md]]"
title: "WEAVE: Unleashing and Benchmarking the In-context Interleaved Comprehension and Generation"
site: "arXiv (2511.11434v1)"
author: "Wei Chow, Jiachun Pan, et al. (NUS + NTU + Maryland + Zhejiang)"
published: "2025-11"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [multimodal, comprehension, generation, benchmark, multi-turn, interleaved]
aliases: [WEAVE, WEAVE-100k, WEAVEBench]
---

# WEAVE: In-context Interleaved Comprehension and Generation

NUS + NTU + 马里兰 + 浙大，2025。

## 核心结论

- 现有 UMM benchmark 聚焦单轮交互，无法捕捉真实世界中**多轮、上下文依赖**的图像创建和编辑。
- 提出 **WEAVE**，首个上下文交织跨模态理解和生成套件。
- **WEAVE-100k**：10 万交织样本，37 万+ 对话轮次，50 万+ 图像。
- **WEAVEBench**：100 个任务、480 张图像的人工标注 benchmark，混合 VLM judger 评估。

## 核心特点

### 多轮上下文编辑
- 不是"给指令→得图"，而是多轮对话中逐步精修
- 模型需要记住之前的编辑意图和图像状态
- 更接近真实工作流（设计师不断迭代）

### 交织理解与生成
- 同一对话中可能同时包含"描述这张图"和"修改这张图"
- 要求模型在理解和生成模式之间灵活切换

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 为统一多模态模型的上下文编辑能力提供了关键 benchmark
- 与 [[Wiki/Comparisons/统一多模态模型架构比较]] 中覆盖的模型直接相关

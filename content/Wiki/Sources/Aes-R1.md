---
type: source
status: processed
source_file: "[[Clippings/Unlocking the Essence of Beauty Advanced Aesthetic Reasoning with Relative-Absolute Policy Optimization.md]]"
title: "Aes-R1: Unlocking the Essence of Beauty — Advanced Aesthetic Reasoning with Relative-Absolute Policy Optimization"
site: "arXiv (2509.21871v1)"
author: "Boyang Liu, Yifan Hu, Senjie Jin, et al. (Fudan + Tsinghua + ByteDance)"
published: "2025-09"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [aesthetic, IAA, reasoning, RL, MLLM]
aliases: [Aes-R1, Unlocking the Essence of Beauty]
---

# Aes-R1: 审美推理 — Unlocking the Essence of Beauty

复旦 + 清华 + 字节跳动，2025。

## 核心结论

- MLLM 在图像审美评估（IAA）中展现潜力，但直接应用 RL 无法激发推理模式。
- 提出 **Aes-R1** 框架：(1) **AesCoT** 数据管线构建高质量审美推理数据做 cold-start；(2) **RAPO (Relative-Absolute Policy Optimization)** RL 算法联合优化绝对分数回归和相对排序。
- 在多个 IAA benchmark 上 SOTA。

## 关键方法

### AesCoT (Aesthetic Chain-of-Thought)
- 训练模型在评分前生成结构化解释（构图、色彩、光影、情感冲击等）
- Cold-start：先用审美推理数据微调，再 RL

### RAPO (Relative-Absolute Policy Optimization)
- **Absolute**：回归绝对审美分数
- **Relative**：优化跨图像偏好排序
- 联合优化 → 既准又"懂"审美比较

## 与图像编辑的关系

- 审美评估 → 可以指导编辑方向（什么编辑"更美"）
- 与 [[Wiki/Sources/EmoEdit]] 的 Aesthetic score 筛选有交集
- 审美推理能力可以集成到编辑模型中做"美化编辑"

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 为图像编辑提供"审美方向"的量化指导
- RAPO 的联合优化思路可推广到其他双目标优化场景

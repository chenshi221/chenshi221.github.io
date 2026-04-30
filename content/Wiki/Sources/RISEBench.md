---
type: source
status: processed
source_file: "[[Clippings/Envisioning Beyond the Pixels Benchmarking Reasoning-Informed Visual Editing.md]]"
title: "Envisioning Beyond the Pixels: Benchmarking Reasoning-Informed Visual Editing (RISEBench)"
site: "arXiv (2504.02826v4)"
author: "Xiangyu Zhao, Peiyuan Zhang, et al."
published: "2025-04"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [image-editing, reasoning, benchmark, evaluation, LMM]
aliases: [RISEBench, Envisioning Beyond the Pixels]
---

# RISEBench: Benchmarking Reasoning-Informed Visual Editing

上海交大 + 上海 AI Lab + 武大 + 同济 + 普林斯顿，2025。

## 核心结论

- 提出 **RISEBench**，首个 **推理信息视觉编辑（RISE）** benchmark。
- 覆盖四大推理类别：**Temporal（时间）、Causal（因果）、Spatial（空间）、Logical（逻辑）**。
- 评估框架：Instruction Reasoning + Appearance Consistency + Visual Plausibility（人类评判 + LMM-as-a-judge）。
- 测试了 8 个主流模型（开源+闭源），发现现有模型在推理类编辑上存在显著困难。

## 四大推理类别

| 类别 | 示例任务 |
|------|----------|
| Temporal | "把这个场景变成 10 年后的样子" |
| Causal | "因为下雨了，让地面变湿" |
| Spatial | "把左边的树移到房子后面" |
| Logical | "今天是圣诞节，请添加相应装饰" |

## 关键发现

- 当前模型（包括 GPT-4o）在推理类编辑上表现显著弱于简单编辑
- Causal 和 Temporal 推理是最难的类别
- 这解释了为什么 [[Wiki/Sources/EditWorld]] 提出的 world-instructed editing 具有挑战性

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 为 EditWorld、GoT 等推理类编辑方法提供了标准化的评估框架
- 填补了从"简单编辑"到"推理编辑"的评估空白

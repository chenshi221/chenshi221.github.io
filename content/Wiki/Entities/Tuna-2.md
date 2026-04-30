---
type: entity
status: active
created: "2026-04-30"
updated: "2026-04-30"
aliases: [Tuna-2, Tuna-R]
tags: [unified-multimodal, encoder-free, pixel-space, meta-ai]
sources:
  - "[[Wiki/Sources/Tuna-2]]"
confidence: high
---

# Tuna-2（Meta AI）

**Tuna-2** 是 Meta AI、香港大学和 Waterloo 大学在 2026 年 4 月发布的无编码器统一多模态模型。

- **机构**：Meta AI、The University of Hong Kong、University of Waterloo
- **架构**：Qwen2.5-7B-Instruct + patch embedding（无 VAE、无 SigLIP 等编码器）
- **关键创新**：首次证明统一多模态模型不需要任何预训练视觉编码器，直接在像素空间端到端学习
- **对比实验**：Tuna-2（无编码器）vs Tuna-R（有 SigLIP2 编码器）→ 理解任务上无编码器更优
- **Benchmark**：GQA 65.0、MMMU 50.7、GenEval 等多项超越同规模模型
- **开源**：项目页面 tuna-ai.org/tuna-2

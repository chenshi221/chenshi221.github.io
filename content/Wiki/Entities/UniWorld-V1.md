---
type: entity
status: active
created: "2026-04-30"
updated: "2026-04-30"
aliases: [UniWorld]
tags: [unified-multimodal, semantic-encoder, pku, image-editing]
sources:
  - "[[Wiki/Sources/UniWorld-V1]]"
confidence: high
---

# UniWorld-V1（Peking University）

**UniWorld-V1** 是北京大学（PKU）在 2025 年 6 月发布的统一生成框架。

- **机构**：Peking University、Peng Cheng Laboratory、Rabbitpre AI
- **架构**：Qwen2.5-VL-7B + SigLIP2 + FLUX DiT
- **关键创新**：用语义编码器（SigLIP2）替代 VAE，仅 270 万样本即达到顶尖编辑性能
- **能力范围**：理解 + 生成 + 编辑 + 感知（检测、分割、深度估计）
- **Benchmark**：ImgEdit-Bench 3.26（开源第一）、WISE 0.55
- **开源**：代码、权重、数据、训练评估脚本完整开源

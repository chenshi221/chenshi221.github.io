---
type: entity
status: active
created: "2026-04-30"
updated: "2026-04-30"
aliases: [Show-o2]
tags: [unified-multimodal, show-lab, nus, flow-matching]
sources:
  - "[[Wiki/Sources/Show-o2]]"
confidence: high
---

# Show-o2（NUS Show Lab）

**Show-o2** 是新加坡国立大学 Show Lab 和字节跳动在 2025 年 6 月发布的改进版原生统一多模态模型。

- **机构**：National University of Singapore、ByteDance
- **架构**：3D Causal VAE + 双路径空间融合 + LLM backbone（AR for text, Flow for image/video）
- **关键创新**：3D Causal VAE 支持图像和视频统一建模；双路径融合结合语义和低层特征
- **能力范围**：文本、图像、视频的理解和生成
- **扩展策略**：1.5B 预训练 flow head 可恢复到大模型（7B）
- **与 Show-o 的关系**：从 Show-o 的离散 token + AR+Diffusion 升级到 3D VAE + 连续 Flow Matching

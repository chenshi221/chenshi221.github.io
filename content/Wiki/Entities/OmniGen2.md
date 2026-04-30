---
type: entity
status: active
created: "2026-04-30"
updated: "2026-04-30"
aliases: [OmniGen2]
tags: [unified-multimodal, instruction-alignment, baai]
sources:
  - "[[Wiki/Sources/OmniGen2]]"
confidence: high
---

# OmniGen2（BAAI）

**OmniGen2** 是北京智源人工智能研究院（BAAI）在 2025 年 6 月发布的统一多模态生成模型。

- **机构**：BAAI、USTC、CASIA、Zhejiang University
- **架构**：Qwen2.5-VL-3B（理解）+ Diffusion Transformer（生成），通过 VLM 变长隐状态连接
- **关键创新**：Omni-RoPE 三维位置编码 + 渐进式 GRPO 多任务指令对齐
- **能力范围**：T2I、图像编辑、上下文生成（in-context generation）
- **OmniContext 基准**：提出标准化的上下文生成评估基准
- **开源**：模型、代码、基准、训练数据均开源

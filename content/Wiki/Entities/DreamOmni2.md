---
type: entity
status: active
created: "2026-04-29"
updated: "2026-04-29"
aliases: [DreamOmni2]
tags: [model, diffusion, image-editing, image-generation, DiT]
sources:
  - "[[Wiki/Sources/DreamOmni2]]"
confidence: high
---

# DreamOmni2

## 概要

DreamOmni2 是由 CUHK（香港中文大学）与 ByteDance（字节跳动）联合提出的多模态指令图像编辑与生成模型。它是 [[DreamOmni]] 的扩展，将统一的图像生成/编辑能力从单图像 + 文本指令扩展到多图像 + 多模态指令。

## 基本信息

- **论文**: DreamOmni2: Multimodal Instruction-based Editing and Generation (arXiv 2510.06679)
- **机构**: CUHK, HKUST, HKU, ByteDance Inc.
- **代码**: [GitHub - dvlab-research/DreamOmni2](https://github.com/dvlab-research/DreamOmni2)
- **基座模型**: Flux Kontext（[[DiT]] 架构）
- **VLM**: Qwen2.5-VL 7B

## 核心贡献

1. 提出两个新任务：多模态指令编辑和多模态指令生成
2. 三阶段数据合成管线（feature mixing → 提取模型 → 编辑/生成数据）
3. 索引编码 + 位置编码偏移方案（处理多图像输入）
4. 生成/编辑模型与 VLM 的联合训练
5. DreamOmni2 benchmark（205 编辑 + 114 生成测试）

## 消融实验结论

- 数据和 VLM 联合训练各自独立贡献提升，联合最优
- 索引编码 + 位置编码偏移缺一不可
- LoRA 训练保留基座模型原有指令编辑能力

## 相关模型/工作

- [[DreamOmni]] — 前身工作（CVPR 2025），单图像 + 文本指令的统一生成与编辑
- [[DreamVE]] — 同团队的图像/视频编辑统一模型
- [[Flux Kontext]] — 基座模型
- [[Qwen2.5-VL]] — VLM 组件
- GPT-4o / Nano Banana — 对比的商业闭源模型

## 资料来源

- [[Wiki/Sources/DreamOmni2]]

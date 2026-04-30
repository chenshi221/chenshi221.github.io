---
type: source
status: processed
source_file: "[[Clippings/Emerging Properties in Unified Multimodal Pretraining.md]]"
title: "BAGEL: Emerging Properties in Unified Multimodal Pretraining"
site: "arXiv (ByteDance Seed)"
author: "Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li et al."
published: "2025-05"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [unified-multimodal, bagel, mot, emergent-properties]
aliases: [BAGEL, ByteDance BAGEL]
---

# BAGEL: Emerging Properties in Unified Multimodal Pretraining

## 核心结论

- BAGEL 是 ByteDance Seed 开源的统一多模态基础模型，采用 **MoT（Mixture-of-Transformers）** 架构，7B 激活参数 / 14B 总参数。
- 在大规模交错多模态数据（文本、图像、视频、网页）上预训练后，展现出**涌现能力**：复杂多模态推理、自由形式图像编辑、未来帧预测、3D 操作、世界导航。
- 关键洞察：**交错多模态数据 + 无瓶颈架构**是涌现能力的关键，而非单纯扩大模型。

## 关键事实

- **架构**：两种 Transformer Expert（理解 + 生成），共享 self-attention，无瓶颈连接。理解用 SigLIP2 ViT，生成用 FLUX VAE。
- **交错生成**：✅ 支持图文交错生成（文本→图像→文本在单次推理中）。使用 Generalized Causal Attention + diffusion forcing 策略处理多图像交错序列。
- **数据**：训练了约 5.1T tokens（包括 400M 文本、500M 理解图文对、1.6B 生成图文对、100M 交错理解、45M 视频交错、20M 网页交错）。
- **训练策略**：四阶段——Alignment → Pre-training (2.5T) → Continued Training (2.6T) → SFT (72.7B)。生成与理解数据采样比约 4:1。
- **涌现模式**：基本理解和生成在 0.18T/0.68T tokens 时接近饱和，编辑任务在 2.64T 达到 85% 性能，而需要复杂推理的 Intelligent Edit 在 3.61T 才开始显著提升。
- **推理增强**：Self-CoT 将 WISE 分数从 0.52 提升至 0.70，IntelligentBench 从 44.9 提升至 55.3。
- **Benchmark 表现**：GenEval 0.88（使用 LLM rewriter），MMMU 55.3，MM-Vet 67.2，超越同期开源统一模型。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 关联：[[Wiki/Concepts/多模态指令编辑与生成]]
- 关联：[[Wiki/Entities/DreamOmni2]]（同属统一多模态生成方向）
- 对比：BAGEL（MoT 集成架构）vs DreamOmni2（三步数据合成管线）

## 可能的矛盾或待核实点

- BAGEL 的数据量（5.1T tokens）远大于 UniWorld-V1（2.7M 样本），但后者在一些编辑任务上表现相当。架构效率差异需要进一步分析。
- Self-CoT 的收益可能部分来自推理时计算，而非模型本身能力增强。

## 后续问题

- MoT 与 MoE 架构的数学本质差异，以及为何 MoT 在生成任务上明显更优？
- 涌现能力是否在更大规模模型上继续保持？

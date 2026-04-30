---
type: source
status: processed
source_file: "[[Clippings/Unified Multimodal Understanding and Generation Models Advances, Challenges, and Opportunities.md]]"
title: "Unified Multimodal Understanding and Generation Models: Advances, Challenges, and Opportunities"
site: "arXiv (Alibaba Group)"
author: "Shanshan Zhao, Xinjie Zhang, Jintao Guo, Qing-Guo Chen et al."
published: "2025-05"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [unified-multimodal, survey, taxonomy]
aliases: [统一多模态综述, UMM Survey]
---

# Unified Multimodal Understanding and Generation Models: Advances, Challenges, and Opportunities

## 核心结论

- 全面综述了统一多模态理解与生成模型（UMMs），提出**三分法分类体系**：Diffusion-based、Autoregressive-based、Fused AR + Diffusion。
- Autoregressive 模型进一步按编码方式分为四类：像素编码、语义编码、可学习查询编码、混合编码。
- 系统整理了 UMM 相关的数据集、基准和评估方法。

## 分类体系

| 类别 | 子类 | 代表模型 |
|------|------|----------|
| **Diffusion** | a. 纯扩散 | Dual Diffusion, UniDisc, FUDOKI, Muddit, MMaDA, UniModel |
| **AR (MLLM)** | b-1. 像素编码 | Chameleon, Emu3, LWM, OneCat, Show-o |
| | b-2. 语义编码 | Emu2, LaVIT, VILA-U, OmniGen2, Qwen-Image, UniWorld |
| | b-3. 可学习查询 | SEED, MetaQueries, BLIP3-o, Nexus-Gen |
| | b-4. 伪混合编码 | Janus, Janus-Pro, Unifluid, MindOmni |
| | b-5. 联合混合编码 | TokenFlow, VARGPT, Show-o2, ILLUME+ |
| **Fused AR+Diff** | c-1. 像素编码 | Transfusion, Show-o, MonoFormer, TUNA |
| | c-2. 混合编码 | Janus-Flow, Mogao, BAGEL, LightFusion, EMMA |

## 关键洞察

- **扩散 vs AR 之争**：扩散模型在图像质量上有优势，AR 与 LLM 的结构一致性使其更适合统一建模。
- **编码策略权衡**：像素编码保留细节但缺乏语义；语义编码对齐好但缺少像素级控制；混合编码结合二者但增加复杂度。
- **关键挑战**：tokenization 策略、跨模态注意力、数据构建、模型评估。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Sources/BAGEL]]（BAGEL 属于 c-2 混合编码 Fused AR+Diff）
- 关联：[[Wiki/Sources/UniWorld-V1]]（b-2 语义编码 AR）
- 关联：[[Wiki/Sources/Tuna-2]]（b-1/连续像素 AR）
- 关联：[[Wiki/Sources/Lumina-DiMOO]]（a 类纯扩散，但独立于该分类）
- 关联：[[Wiki/Sources/OmniGen2]]（b-2 语义编码 AR）
- 关联：[[Wiki/Sources/Show-o2]]（b-5 联合混合编码 AR）
- 关联：[[Wiki/Concepts/多模态指令编辑与生成]]

## 后续问题

- 该分类体系是否覆盖了所有 UMM？离散扩散模型（如 Lumina-DiMOO）是否应构成第四大类？
- 各分类之间的性能边界是否随着模型规模增大而模糊？

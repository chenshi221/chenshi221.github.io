---
type: topic
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - Benchmark 主题
tags:
  - benchmark
  - OCR
  - document-parsing
  - evaluation
sources:
  - Wiki/Sources/OCRBench v2
  - Wiki/Sources/OmniDocBench
  - Wiki/Sources/olmOCR
---
# 多模态 Benchmark 与评估

## 概述

多模态模型的评估需要覆盖从基础感知到复杂推理的多个维度。本主题涵盖 OCR 能力评测、PDF 文档解析评测和 PDF OCR 工具链评测，形成从"测什么"到"怎么测"到"用什么工具"的完整链条。

## 核心来源

- [[Wiki/Sources/OCRBench v2]]：8 大 OCR 核心能力 x 23 任务 x 31 场景，10K 人工验证 QA 对，揭示 LMM 在文本定位、手写体、逻辑推理上的不足。
- [[Wiki/Sources/OmniDocBench]]：9 种文档类型 x 三级评估（端到端/任务级/属性级），981 页高质量标注，对比 Pipeline 和 VLM 方法的优劣势。
- [[Wiki/Sources/olmOCR]]：开源 PDF OCR 工具包（Qwen2-VL 7B 微调），$176/百万页，性能超越 GPT-4o，配套 olmOCR-Bench 评测基准。

## 关键概念

- [[Wiki/Concepts/PDF 文档解析]]：PDF 文本提取的两大范式（Pipeline vs End-to-End VLM）及评估标准。
- OCR 评估维度：text recognition、text localization、layout analysis、formula/table recognition、reading order。
- 多层级评估设计：从整体端到端得分到细粒度属性级分析的评估体系。

## 评估体系全景

| Benchmark | 覆盖范围 | 核心指标 | 规模 |
|-----------|---------|---------|------|
| OCRBench v2 | 31 场景 8 能力 | 6 类指标 | 10K QA 对 |
| OmniDocBench | 9 文档类型 | 3 级评估 19 布局类 | 981 页 |
| olmOCR-Bench | 多种 PDF 类型 | 7K+ unit tests | 1,400 页 |

## 开放问题

- 多页文档（跨页表格、引用）的评估方法？
- 如何设计"对抗性"测试集来探测模型鲁棒性边界？
- 工业级文档解析的实时性要求与精度的平衡？

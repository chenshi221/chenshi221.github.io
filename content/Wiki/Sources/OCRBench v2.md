---
type: source
status: processed
source_file: >-
  [[Clippings/OCRBench v2 An Improved Benchmark for Evaluating Large Multimodal
  Models on Visual Text Localization and Reasoning]]
title: 'OCRBench v2: An Improved Benchmark for Evaluating LMMs on Visual Text'
site: arXiv 2501.00321
author: 'Ling Fu et al. (HUST, ByteDance)'
published: '2025'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - benchmark
  - OCR
  - LMM
  - text-recognition
aliases:
  - OCRBench v2 论文
---
# OCRBench v2: An Improved Benchmark for Evaluating LMMs on Visual Text

## 核心结论

- 提出 OCRBench v2，目前最大规模的双语文本中心 benchmark，8 大核心 OCR 能力 x 23 个具体任务 x 31 种场景。
- 包含 10,000 条人工验证 QA 对 + 1,500 张私有测试集图片，评估 LMM 的 text localization、handwritten extraction 和 logical reasoning。

## 关键事实

- 来源：华中科技大学、华南理工大学、Adelaide、字节跳动，2024。
- 8 大核心能力：text recognition、text referring、text spotting、relation extraction、element parsing、mathematical calculation、visual text understanding、knowledge reasoning。
- 发现：大多数 LMM 得分低于 50/100，存在五大类不足：罕见文字识别、细粒度感知、布局感知、复杂元素解析、逻辑推理。
- 与 OmniDocBench、MMLONGBENCH-DOC 等形成互补，覆盖更偏向 OCR 基础能力的评估。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/多模态 Benchmark 与评估]]
- 关联：[[Wiki/Concepts/PDF 文档解析]]
- OCRBench v2 聚焦 OCR 全能力，OmniDocBench 聚焦文档解析端到端评估。

## 后续问题

- 中文和英文性能差距的系统分析？
- 如何设计针对特定应用场景的 OCR subset evaluation？

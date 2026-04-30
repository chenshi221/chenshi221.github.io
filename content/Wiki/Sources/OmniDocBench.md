---
type: source
status: processed
source_file: >-
  [[Clippings/OmniDocBench Benchmarking Diverse PDF Document Parsing with
  Comprehensive Annotations]]
title: 'OmniDocBench: Benchmarking Diverse PDF Document Parsing'
site: arXiv 2412.07626
author: Linke Ouyang et al. (Shanghai AI Lab)
published: '2024'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - benchmark
  - document-parsing
  - PDF
  - VLM
aliases:
  - OmniDocBench 论文
---
# OmniDocBench: Benchmarking Diverse PDF Document Parsing

## 核心结论

- 提出 OmniDocBench，首个高质量、多文档类型、多层级评估的文档解析 benchmark。
- 覆盖 9 种文档类型（学术论文、教科书、手写笔记、报纸等），支持端到端、单任务、属性三级评估。

## 关键事实

- 来源：上海 AI Lab（Shanghai AI Laboratory）、Abaka AI、2077AI，2024。
- 981 页高质量标注，19 种布局类别标签，15 种属性标签。
- 支持三级评估：(1) End-to-End：全页解析质量；(2) Task-Specific：布局检测、OCR、表格识别、公式解析分项评估；(3) Attribute-Based：按文档类型、页面属性的细粒度分析。
- 同时评估了 Pipeline 方法（MinerU、Marker）和端到端 VLM 方法（GPT-4o、Qwen2-VL），揭示不同方法的优劣势分布。
- 解决了已有 benchmark 的三个问题：文档类型单一、评估指标不统一、缺乏细粒度分析。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/多模态 Benchmark 与评估]]
- 关联：[[Wiki/Concepts/PDF 文档解析]]
- 与 [[Wiki/Sources/OCRBench v2]] 互补：前者侧重 OCR 全能力，OmniDocBench 侧重 PDF 端到端解析。

## 后续问题

- VLM 在处理手写体和罕见排版时的鲁棒性如何改善？
- 多页文档（跨页表格、引用）解析的评估如何设计？

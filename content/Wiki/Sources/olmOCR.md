---
type: source
status: processed
source_file: >-
  [[Clippings/pipeline Unlocking Trillions of Tokens in PDFs with Vision
  Language Models]]
title: 'olmOCR: Unlocking Trillions of Tokens in PDFs with VLMs'
site: arXiv 2502.18443
author: Jake Poznanski et al. (Allen AI)
published: '2024'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - PDF
  - OCR
  - VLM
  - toolkit
  - open-source
aliases:
  - olmOCR 论文
---
# olmOCR: Unlocking Trillions of Tokens in PDFs with VLMs

## 核心结论

- 提出 olmOCR，一个开源 PDF 文本提取与线性化工具包，基于微调的 Qwen2-VL 7B VLM。
- 处理百万页 PDF 仅需 $176 USD，性能超越 GPT-4o、Gemini Flash 2 等商用 API。

## 关键事实

- 来源：Allen Institute for AI (AI2)，2024。
- 训练数据 olmOCR-mix-0225：从 10 万份爬取 PDF 中采样 26 万页，由 GPT-4o 生成 OCR 标注。
- 评测基准 olmOCR-Bench：1,400 页 PDF，7,000+ 单元测试（unit-test 风格），覆盖公式、表格、小字体、旧扫描件等挑战内容。
- 支持 vLLM 和 SGLang 推理引擎，可扩展至数百 GPU。
- 下游验证：处理 790 万页 peS2o 科学文献 PDF，替换原有提取文本后训练 LM，下游 benchmark 性能提升。
- 与传统 pipeline 工具（MinerU、Marker）和端到端 VLM 的对比中均表现最优。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/多模态 Benchmark 与评估]]
- 关联：[[Wiki/Concepts/PDF 文档解析]]
- 与 [[Wiki/Sources/OmniDocBench]] 互补：olmOCR 提供 PDF 提取工具链，OmniDocBench 提供评估标准。

## 后续问题

- 对中文、多语言 PDF 的支持程度？
- 扫描版 PDF 中手写体识别的准确率如何？

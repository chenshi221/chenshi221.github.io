---
type: concept
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - PDF Parsing
  - 文档解析
tags:
  - PDF
  - OCR
  - document-parsing
  - VLM
sources:
  - Wiki/Sources/OmniDocBench
  - Wiki/Sources/olmOCR
confidence: high
---
# PDF 文档解析

## 定义

PDF 文档解析是指从 PDF 格式的文档中提取结构化、可机读内容（文本、表格、公式、阅读顺序）的技术任务。是 LLM 训练数据构建和 RAG 系统的关键基础设施。

## 两大技术范式

### 1. Pipeline 方法

将文档解析分解为多个独立模块串联：
- 布局检测（Document Layout Analysis）
- OCR 文字识别
- 公式识别（LaTeX/HTML）
- 表格识别（HTML/Markdown）
- 阅读顺序估计
- 代表工具：MinerU、Marker、Grobid
- 优点：模块化、可并行、效率高
- 缺点：模块间错误累积，跨领域泛化有限

### 2. End-to-End VLM 方法

使用视觉语言模型直接从 PDF 页面图像输出结构化文本：
- 代表：Nougat、GOT-OCR、olmOCR、GPT-4o
- 优点：端到端简化流程，泛化能力强
- 挑战：计算成本高（GPT-4o ~$6,240/百万页），封闭 API 可能不可用

## 关键挑战

1. **PDF 格式的特殊性**：PDF 存储的是字符坐标而非逻辑文本流，缺乏阅读顺序信息
2. **文档多样性**：学术论文 vs 教科书 vs 手写笔记 vs 报纸 vs 财务报表
3. **复杂元素**：多栏布局、浮动图表、跨页表格、行内公式、脚注
4. **质量差异**：扫描质量、字体大小、语言混合

## 评估方法

- Evals 需覆盖：文本准确性、结构保真度、阅读顺序正确性、表格/公式完整性
- [[Wiki/Sources/OmniDocBench]]：9 种文档类型三级评估
- [[Wiki/Sources/olmOCR]]：unit-test 风格评估，7K+ 测试用例
- 关键指标：Edit Distance、BLEU、表格 TEDS、公式 CDM

## 相关来源

- [[Wiki/Sources/OmniDocBench]]
- [[Wiki/Sources/olmOCR]]
- [[Wiki/Topics/多模态 Benchmark 与评估]]

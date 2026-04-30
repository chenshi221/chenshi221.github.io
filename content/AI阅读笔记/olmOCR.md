---
title: 'olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models'
authors:
  - Jake Poznanski
  - Aman Rangapur
  - Jon Borchardt
  - Jason Dunkelberger
  - Regan Huff
  - Daniel Lin
  - Christopher Wilhelm
  - Kyle Lo
  - Luca Soldaini
institutions: Allen Institute for AI (Ai2)
aliases:
  - olmOCR
  - olmOCR pipeline
date: '2026-04-30'
publish_date: 2025-02
venue: 'arXiv:2502.18443'
tags:
  - 论文
  - OCR
  - 文档解析
  - VLM
  - PDF
  - 开源
  - 语言模型训练数据
url: 'https://arxiv.org/abs/2502.18443'
code: 已开源 (https://github.com/allenai/olmocr)
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：Allen AI 提出 olmOCR，一个基于 7B VLM（Qwen2-VL 微调）的开源 PDF 文档线性化工具包。通过 document-anchoring（利用 PDF 内部元数据锚定提示）和 olmOCR-Bench（7,010 个 pass/fail 单元测试），olmOCR 超越了 GPT-4o 等闭源 VLM，同时成本仅为 GPT-4o 的 1/32（每百万页 $176 vs $6,240）。用 olmOCR 重新处理 peS2o 学术语料训练 OLMo-2 时，下游 benchmark 平均提升 1.3%。

---

## Intro

### Motivation

PDF 文档是 LLM 训练数据的巨大未充分利用资源——全球有数万亿 PDF 文档。但 PDF 格式的核心问题是：它存储的是渲染信息（单个字符 + 位置坐标），而非逻辑结构（段落、标题、阅读顺序）。

**现有方案的问题**：
1. **Pipeline 工具**（MinerU, Marker）：由多个 ML 模块拼接，在常见文档上效果好，但对特殊格式泛化差
2. **商业 API**（GPT-4o, Mistral OCR）：质量高但成本惊人——GPT-4o 处理百万页需要 $6,240+
3. **开源 VLM**：直接提示效果差，容易产生幻觉和遗漏内容

### 核心主张

通过微调 7B VLM + document-anchoring 技术，可以在极低成本下实现超越顶级商业 API 的 PDF 线性化质量。

### 贡献

1. **olmOCR-7B**：基于 Qwen2-VL-7B 微调的高质量 PDF 线性化模型
2. **olmOCR-mix-0225**：260,000 页 GPT-4o 生成的银标训练数据
3. **olmOCR-Bench**：7,010 个 pass/fail 单元测试的 benchmark
4. **document-anchoring**：利用 PDF 内部元数据提升提取质量
5. **开源工具包**：支持 vLLM/SGLang 的高效推理 pipeline

---

## Method 核心方法

### 1. Document-Anchoring

![](https://arxiv.org/html/2502.18443/x1.png)

*Figure 1: olmOCR 的 document-anchoring 流程——利用 PDF 内部元数据（文本块+位置坐标）作为锚定信息注入 VLM prompt，VLM 同时参考页面截图和锚定文本，生成自然阅读顺序的线性化文本。此设计显著降低了幻觉和内容遗漏。*

**核心思想**：PDF 大多数是 born-digital 文档，内部已经包含文本和位置元数据——应该利用它们，而非仅依赖页面截图。

**流程**：
1. 用 pypdf 库提取 PDF 的底层结构（文本块 + 图片 + 位置坐标）
2. 按相关性排序（页面开头和结尾的内容优先）
3. 将提取的文本以 "RAW_TEXT_START ... RAW_TEXT_END" 格式插入 VLM 的 prompt
4. VLM 同时看到页面截图和锚定文本，生成自然阅读顺序的纯文本

**为什么有效**：
- 锚定文本减少了 VLM 的幻觉（模型不必"想象"模糊区域的文字）
- 位置信息帮助模型理解布局结构
- 即使锚定文本（来自 PDF 内部）噪声大，VLM 也能用它做参考

### 2. 训练数据：olmOCR-mix-0225

**数据来源**：
- 从 2.4 亿公开网页 PDF 中随机采样 96,929 个文档
- 从 Internet Archive 采样 5,896 本公共领域书籍
- 过滤：非英文、表单、垃圾内容
- 每篇最多采样 3 页，共约 260,000 页

**银标生成**：用 GPT-4o + document-anchoring 为每一页生成 JSON 格式的结构化输出（包含语言、旋转角度、表格信息、自然文本）

**文档类型分布**：
- 学术论文 55.9%
- 宣传册 11.2%
- 法律文件 10.2%
- 书籍 6.8%
- 表格/图表/幻灯片 ~12%

### 3. 模型微调

- 基于 Qwen2-VL-7B-Instruct
- 全参数微调（LoRA 效果更差）
- Learning rate 1e-6, batch size 4, 10,000 步（~1.2 epoch）
- 单节点 8xH100, 16 node-hours
- 最大分辨率 1024px, 总输入约 3,000 tokens

### 4. olmOCR-Bench

**设计理念**：不用软指标（BLEU/Edit Distance），不用 LLM-as-Judge（有自偏好偏误），而是设计可机器验证的 pass/fail 单元测试。

**7 种文档类型，5 种测试类别**：

| 类别 | 测试数 | 说明 |
|------|--------|------|
| ArXiv Math | 2,927 | 数学公式正确性 |
| Old Scans Math | 458 | 旧扫描书数学公式 |
| Tables | 1,020 | 表格单元格关系 |
| Old Scans | 526 | 历史文献阅读顺序 |
| Headers Footers | 753 | 页眉页脚排除 |
| Multi Column | 884 | 多栏阅读顺序 |
| Long Tiny Text | 442 | 密集小字识别 |

**测试类型**：
- Text Presence：某段文字是否出现在输出中
- Text Absence：页眉/页脚/页码是否正确排除
- Natural Reading Order：两段文字的先后顺序是否正确
- Table Accuracy：表格单元格值及其邻居关系
- Math Formula Accuracy：公式符号的相对空间关系（通过 KaTeX 渲染验证）

**评分**：每种文档类型的通过率取平均（macro-average），防止高频类型主导分数。

### 5. 推理 Pipeline

- 支持 SGLang 和 vLLM 后端
- 每工作项约 500 页，GPU 队列调度
- 自动旋转检测和纠正
- 最大 3 次重试 + 回退到纯文本提取
- L40S: 906 tok/s, H100: 3,050 tok/s
- 成本：L40S 下每百万页 $176

---

## 实验/评估/结果

### olmOCR-Bench

| 方法 | Overall Score |
|------|--------------|
| **olmOCR (Anchored)** | **75.5** |
| olmOCR (No Anchor) | 74.7 |
| Mistral OCR API | 72.0 |
| Marker v1.7.5 | 70.1 |
| GPT-4o (Anchored) | 69.9 |
| GPT-4o (No Anchor) | 68.9 |
| Qwen 2.5 VL | 65.5 |
| MinerU v1.3.10 | 61.5 |

- olmOCR 在所有 benchmark 维度上都超越了 GPT-4o
- document-anchoring 带来的改善在公式（ArXiv Math）和旧扫描文档（Old Scans）上最显著

### 下游语言模型训练

- 在 OLMo-2-7B checkpoint 上继续预训练 50B tokens
- 对比用 Grobid + 规则提取的 peS2o vs olmOCR 重新处理的 peS2o
- **主要 benchmark 平均提升 +1.3%**：
  - MMLU 61.1 → 61.1 (持平)
  - ARC-C 75.0 → 76.4 (+1.4)
  - HellaSwag 57.4 → 62.6 (+5.2)
  - DROP 42.3 → 43.7 (+1.4)

### 成本对比

| 方法 | 每百万页成本 |
|------|------------|
| GPT-4o (API) | $12,480 |
| Mistral OCR | $1,000 |
| Marker (H100) | $1,484 |
| MinerU (L40S) | $596 |
| Gemini Flash 2 | $249 |
| **olmOCR (L40S)** | **$176** |

### ELO 人类评估

在 452 个成对比较中：
- olmOCR vs Marker: 61.3% 胜率
- olmOCR vs GOT-OCR: 58.6% 胜率
- olmOCR vs MinerU: 71.4% 胜率
- ELO 评分超过 1800，远超其他工具

---

## 结论

olmOCR 证明了微调一个小型 VLM（7B）+ document-anchoring 可以在 PDF 线性化任务上超越 GPT-4o 等大型商业 API，同时成本降低 30 倍以上。下游语言模型训练实验表明，高质量的 PDF 线性化带来的训练数据改进可以体现在最终的 benchmark 表现中。

---

## 思考

### 优点

1. **全面开源**：训练数据、模型权重、推理代码、benchmark 全部开源——这是可复现研究的典范
2. **Benchmark 设计创新**：pass/fail 单元测试避免了传统 OCR 评估中 fuzzy matching 和 LLM-as-judge 的各种陷阱；公式的空间关系验证尤其巧妙
3. **实用主义**：用 GPT-4o 做 teacher 蒸馏 + 成本只有 1/32——在学术预算内做到了高质量
4. **下游验证**：语言模型训练实验把 PDF 线性化和 LM 性能直接关联，证明了"更好的数据"的真实影响

### 缺点与待解决问题

1. **Teacher 模型的局限**：训练数据来自 GPT-4o，模型的上限受 teacher 限制。GPT-4o 也做不到完美线性化，学生的天花板不高
2. **非英文文档的支持**：训练数据以英文为主（非英文被过滤），中文/日文等多语言支持未验证
3. **document-anchoring 对扫描件的无效**：扫描版 PDF 没有内部元数据，document-anchoring 退化为无锚定模式
4. **图表和图片描述**：当前不生成图片的 caption/alt text，这在某些应用场景下是缺失信息
5. **退化为无锚定模式时的回退质量**：纯扫描件（如旧书扫描）的质量可能大幅下降

### 与已有 Wiki 的连接

- 关联概念：[[Wiki/Concepts/知识蒸馏]]、[[Wiki/Concepts/VLM]]、[[Wiki/Concepts/OCR]]
- 关联实体：[[Wiki/Entities/Qwen2-VL]]、[[Wiki/Entities/Ai2]]
- 关联比较：[[OmniDocBench]]（更细粒度的文档解析评估）、[[OCRBench v2]]（OCR 能力评估）

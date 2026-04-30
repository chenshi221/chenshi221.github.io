---
title: 'OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations'
authors:
  - Linke Ouyang
  - Yuan Qu
  - Hongbin Zhou
  - Jiawei Zhu
  - Rui Zhang
  - Qunshu Lin
  - Bin Wang
  - Zhiyuan Zhao
  - Man Jiang
  - Xiaomeng Zhao
  - Jin Shi
  - Fan Wu
  - Pei Chu
  - Minghao Liu
  - Zhenxiang Li
  - Chao Xu
  - Bo Zhang
  - Botian Shi
  - Zhongying Tu
  - Conghui He
institutions: Shanghai AI Laboratory, Abaka AI, 2077AI
aliases:
  - OmniDocBench
  - OmniDocBench
date: '2026-04-30'
publish_date: 2024-12
venue: 'arXiv:2412.07626'
tags:
  - 论文
  - 文档解析
  - Benchmark
  - OCR
  - PDF
  - 多模态
url: 'https://arxiv.org/abs/2412.07626'
code: 已开源 (https://github.com/opendatalab/OmniDocBench)
rating: ⭐⭐⭐⭐
status: Read
---
**一句话总结**：上海 AI Lab 提出 OmniDocBench，一个专为文档解析任务设计的高质量 benchmark——涵盖 9 种 PDF 文档类型（包括学术论文、教材、考试卷、手写笔记、报纸等）、981 页、19 种布局类别和 15 种属性标签，支持端到端、任务级和属性级三层评估。系统评估了 pipeline 工具和端到端 VLM 在文档解析上的表现差异。

---

## Intro

### Motivation

文档解析（Document Parsing）——从 PDF 中提取结构化的、机器可读的内容——是 LLM 训练数据和 RAG 系统的关键上游任务。

**现有 benchmark 的三大局限**：
1. **文档多样性不足**：主要集中在学术论文，忽略了教材、财务报表、报纸、手写笔记等
2. **评估指标不一致**：仅靠编辑距离或 BLEU 无法评估公式和表格的准确性（LaTeX/HTML 有多种等价表达）
3. **缺乏细粒度评估**：只给总体分，无法定位模型在"公式 vs 表格 vs 文字"或"教科书 vs 报纸"等子维度上的表现

### 核心主张

需要一个包含多样文档类型、支持多层评估（端到端/任务级/属性级）、对公式和表格有专门指标的 benchmark。

### 贡献

1. **OmniDocBench**：981 页，9 种文档类型，超过 20,000 个块级标注 + 70,000 个片段级标注
2. **三层评估**：端到端 + 任务级（OCR/表格/公式/阅读顺序分别评估）+ 属性级（按文档类型、布局复杂度等分组）
3. **系统对比**：评估了 pipeline 工具（MinerU, Marker, Mathpix）、专家 VLM（GOT-OCR, Nougat）和通用 VLM（GPT-4o, Qwen2-VL, InternVL2）三类方法

---

## Method 核心方法

### 1. 数据集构建

**数据采集**：
- 从 Common Crawl、Google、Baidu 和内部数据收集 200,000 份 PDF
- 用 ResNet-50 提取视觉特征 + Faiss 聚类，采样 6,000 个视觉多样页面
- 人工进一步筛选和平衡到 981 页最终数据集

**9 种文档类型**：
- 书籍 (Books)
- PPT 转 PDF (Slides)
- 研究报告 (Financial Report)
- 彩色教科书 (Textbook)
- 考试卷 (Exam Paper)
- 杂志 (Magazine)
- 学术论文 (Academic Papers)
- 手写笔记 (Notes)
- 报纸 (Newspaper)

### 2. 标注体系

**布局检测标注** (block-level, >20,000)：
- 19 种区域类别：标题、正文段落、图片、表格、公式、代码块等
- 阅读顺序标注：>16,000 个区域标注了逻辑阅读顺序
- 附属关系标注：图片/表格的标题、跨页段落的上下文关系

**内容识别标注** (span-level, >70,000)：
- 纯文本标注
- 公式标注（LaTeX 格式）
- 表格标注（HTML + LaTeX 双格式）

**属性标注**：
- 文本属性：语言（中/英/混合）、背景颜色、旋转角度
- 表格属性：边框类型、合并单元格、含公式、彩色背景、旋转
- 页面属性：模糊扫描、水印、彩色背景

### 3. 标注流程

三步质量保证：
1. **自动预标注**：LayoutLMv3（布局）+ PaddleOCR（文字）+ UniMERNet（公式）+ GPT-4o（表格）
2. **人工校正**：修正检测框、验证文字内容、用 Tables Generator 和 latexlive 检查和修正复杂标注
3. **专家质量检查**：用 CDM 渲染技术找出不可渲染的 LaTeX 公式，由 3 位研究员审查修复

### 4. 三层评估框架

**Layer 1 - 端到端评估**：
- 输入：PDF 页面 → 输出：Markdown
- 评估：通过 Adjacency Search Match 算法将预测段落与真值匹配
- 指标：整体编辑距离、各元素得分

**Layer 2 - 任务级评估**：
- OCR（文字识别）：编辑距离
- 表格识别：TEDS + 编辑距离（HTML 格式对齐）
- 公式识别：CDM + 编辑距离 + BLEU
- 布局检测：mAP
- 阅读顺序：归一化编辑距离

**Layer 3 - 属性级评估**：
- 按文档类型分组
- 按属性分组（语言、列布局、特殊问题如模糊/水印等）

---

## 实验/评估/结果

### 端到端评估主要结果

![](https://arxiv.org/html/2412.07626/x1.png)

*Figure 1: OmniDocBench 端到端文本识别结果——9 种 PDF 页面类型（学术论文、教材、考试卷、手写笔记、报纸等）上各方法的编辑距离对比，越低越好。MinerU 和 Qwen2-VL-72B 在学术论文和财务报表上表现最好，但报纸对所有工具都是挑战。*

（编辑距离，越低越好）

| 方法 | English | Chinese | Overall |
|------|---------|---------|---------|
| MinerU (pipeline) | 0.150 | 0.357 | 0.206 |
| Qwen2-VL-72B (通用VLM) | 0.252 | 0.327 | 0.179 |
| GOT-OCR (专家VLM) | 0.287 | 0.411 | 0.267 |
| Marker (pipeline) | 0.336 | 0.556 | 0.274 |
| GPT-4o (通用VLM) | 0.233 | 0.399 | 0.316 |
| Nougat (专家VLM) | 0.452 | 0.973 | 0.806 |

### 关键发现

1. **Pipeline 工具在常见文档上最优秀**：MinerU 和 Qwen2-VL 在学术论文、财务报表上表现最好
2. **通用 VLM 在不常见文档上泛化更好**：手写笔记、幻灯片等——通用 VLM 的训练数据更广泛
3. **报纸是所有工具的噩梦**：高密度多栏布局，几乎所有方法表现差，MinerU 的布局分割有明显优势
4. **视觉退化场景下 VLM 更鲁棒**：模糊扫描、水印、彩色背景——VLM 比 pipeline 工具更稳定
5. **多栏和复杂布局下阅读顺序是瓶颈**：所有模型在双栏/三栏/复杂布局上的阅读顺序得分显著下降

### 单任务评估

**表格识别**：
- 基于 OCR 的专用模型（PaddleOCR, RapidTable）在整体上仍优于 VLM
- 但 VLM 在无边框表格上表现更好

**公式识别**：
- GPT-4o (86.8% CDM) > Mathpix (86.6%) > UniMERNet (85.0%)
- GPT-4o 在严格匹配条件下召回率最高（65.5%）

**布局检测**：
- DocLayout-YOLO（47.38 mAP）显著优于 LayoutLMv3（28.84 mAP）
- DocLayout-YOLO 是 MinerU 的核心布局模块

---

## 结论

OmniDocBench 设置了文档解析评估的新标准：多样化的文档类型、精细的标注体系、三层评估框架。系统对比揭示了 pipeline 工具和 VLM 的不同优劣势——pipeline 在常见文档上精确，VLM 在长尾文档上鲁棒。最佳方案可能是两者的结合：pipeline 做精确提取 + VLM 做泛化和错误修复。

---

## 思考

### 优点

1. **标注质量高**：自动预标注 + 人工校正 + 专家质检三环保证罕见的高质量
2. **文档类型覆盖广**：手写笔记、考试卷、报纸等真实场景文档被现有 benchmark 严重忽略
3. **各层级评估互补**：端到端分数给大致排名，任务级分数定位弱点，属性级分数揭示原因
4. **公平的公式/表格评估**：CDM 验证公式渲染正确性、TEDS 测量表格结构相似性——比简单编辑距离更合理
5. **Ignore 逻辑巧妙**：忽略页眉/页脚等模型输出标准不统一的内容，让评估更公平

### 缺点与待解决问题

1. **数据量偏少**：981 页用于 9 种文档类型，每种约 100 页，统计置信度有限
2. **中文侧重**：中文文档占比高（612/981），英文评估结论的代表性需要更多数据验证
3. **不支持多页文档**：所有评估都是单页级别，无法评估跨页表格/跨页段落的一致性
4. **缺乏多模态内容评估**：不评估图片内容理解（图表中的数据、照片中的信息等）
5. **Pipeline vs VLM 的公平对比受限**：不同方法输出格式不同（有些输出 Markdown，有些只有纯文本），预处理有一定主观性

### 与已有 Wiki 的连接

- 关联概念：[[Wiki/Concepts/文档解析]]、[[Wiki/Concepts/OCR]]、[[Wiki/Concepts/Benchmark]]
- 关联实体：[[Wiki/Entities/上海AI实验室]]、[[Wiki/Entities/GPT-4o]]
- 关联比较：[[OCRBench v2]]（更侧重 OCR 文字能力评估）、[[olmOCR]]（PDF 线性化工具）

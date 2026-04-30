---
title: 'OCRBench v2: An Improved Benchmark for Evaluating Large Multimodal Models on Visual Text Localization and Reasoning'
authors:
  - Ling Fu
  - Zhebin Kuang
  - Jiajun Song
  - Mingxin Huang
  - Biao Yang
  - Yuzhe Li
  - Linghao Zhu
  - Qidi Luo
  - Xinyu Wang
  - Hao Lu
  - Zhang Li
  - Jingqun Tang
  - Wei Chen
  - Lianwen Jin
  - Yuliang Liu
  - Xiang Bai
institutions: Huazhong University of Science and Technology, South China University of Technology, ByteDance
aliases:
  - OCRBench v2
  - OCRBench-v2
date: '2026-04-30'
publish_date: 2025-01
venue: 'arXiv:2501.00321'
tags:
  - 论文
  - OCR
  - Benchmark
  - 多模态
  - LMM
  - 文字识别
  - 文本定位
  - 推理
url: 'https://arxiv.org/abs/2501.00321'
code: 已开源（数据+代码）
rating: ⭐⭐⭐⭐
status: Read
---
**一句话总结**：OCRBench v2 是目前覆盖面最广的 OCR 能力评估 benchmark——23 个任务、31 种场景、10,000 个人工验证的 QA 对，系统评估 LMM 在文本识别、文本定位、文本检测、关系抽取、元素解析、数学计算、视觉文本理解和知识推理 8 项核心能力的表现。额外提供 1,500 张隐私测试集防止数据污染。当前大多数 LMM 在 benchmark 上总得分不超过 50（满分 100）。

---

## Intro

### Motivation

现有 OCR benchmark 的局限性：
1. **任务单一**：DocVQA、ChartQA 等只评估特定场景（文档/图表）
2. **能力饱和**：Qwen2.5-VL 在 DocVQA 上已达 96.4%（接近人类 98.1%），OCRBench v1 上达 88.8%
3. **缺少文本定位和推理评估**：大多数 benchmark 只评估"能不能读到文字"，不评估"能不能定位文字位置"和"能不能基于文字推理"
4. **缺少私有测试集**：公开数据可能已被泄露到 LMM 预训练数据中

### 核心主张

当前 LMM 在文字相关任务上的能力被高估了——它们只是擅长简单的文字识别，但在定位、解析、推理等更困难的任务上表现很差。需要一个更全面、更有难度、有隐私保护的 benchmark 来揭示这些差距。

### 贡献

1. **OCRBench v2**：8 项核心能力、23 个任务、31 种场景、10,000 QA 对
2. **私有测试集**：额外 1,500 张人工标注图像（不公开）
3. **6 种评估指标**：针对不同任务类型的专项指标
4. **18 个 LMM 的系统评估**：揭示 5 类关键局限

---

## Method 核心方法

### 1. 8 项核心 OCR 能力

![](https://arxiv.org/html/2501.00321/x5.png)

*Figure 5: OCRBench v2 的 8 项核心 OCR 能力和 23 个子任务总览——每种颜色代表一个能力类型，覆盖从基础文字识别到高阶知识推理的完整 OCR 能力谱系。*

| 能力 | 子任务 | 核心评估目标 |
|------|--------|------------|
| Text Recognition | 标准识别、细粒度识别、全文 OCR | 能否在不同场景下准确识别文字 |
| Text Referring | 文本定位、带坐标 VQA | 能否在图中找到指定文字的位置 |
| Text Spotting | 文本检测+识别 | 能否同时给出所有文字的位置和内容 |
| Relation Extraction | 关键信息抽取、信息映射、手写提取 | 能否从密集排列的文字中提取结构化关系 |
| Element Parsing | 表格解析、图表解析、文档解析、公式识别 | 能否将复杂元素输出为 Markdown/JSON/LaTeX |
| Mathematical Calculation | 数学 QA、文本计数 | 能否对文字中的数字进行正确推理 |
| Visual Text Understanding | 文档分类、图表 QA、认知 VQA | 能否理解文字视觉语义 |
| Knowledge Reasoning | 科学 QA、APP 代理、ASCII 艺术、翻译、推理 VQA | 需要常识和专业知识的高阶推理 |

任务量为已有最大 benchmark（OCRBench v1）的 4 倍。

### 2. 31 种场景覆盖

涵盖：示意图、科学论文、Word 文档、表格、图表、收据、试题、数学公式、产品标签、手机截图、室内场景、行业报告、海报、街景、ASCII 艺术、店招、财报、化学式、教科书、杂志、邮件、网页截图、详情页、验证码、简历、插画、报纸、路标、菜单、通知、问卷

### 3. 6 种评估指标

- **Parsing Type**：TEDS（Tree Edit Distance Similarity）测量结构化输出与真值的树编辑距离
- **Localization Type**：IoU 测量定位精度
- **Extraction Type**：F1 score 测量信息抽取的精准率和召回率
- **Long Reading Type**：BLEU + METEOR + F1 + NED（Normalized Edit Distance）综合评估长文本阅读
- **Counting Type**：归一化 L1 距离
- **Basic VQA Type**：精确匹配（短答案）或 ANLS（长答案）

### 4. 私有测试集

- 1,500 张手动收集和标注的文本丰富图像
- 来源：印刷书籍、电子书、扫描文档、网页内容
- 不公开，只维护定期更新的 leaderboard
- 公开和私有测试集上的模型排名高度一致，验证了 benchmark 的设计有效性

---

## 实验/评估/结果

### 英文公开数据上的主要结果

| Model | Average |
|-------|---------|
| InternVL3-14B | **52.6** |
| Gemini 1.5 Pro | 51.9 |
| Ovis2-8B | 47.7 |
| Step-1V | 46.7 |
| Qwen2.5-VL-7B | 46.7 |
| GPT-4o | 46.5 |
| Pixtral-12B | 40.3 |

**关键发现**：目前最好的模型也仅达到 ~53 分（满分 100）。

### 5 个关键发现

**Finding 1 - 低频文字识别困难**：LMM 对点阵字体、手写体、遮挡文字、验证码等低频文本的识别率比高频文字低 20-30%。

**Finding 2 - 空间感知弱**：InternVL3-14B 在"带坐标 VQA"任务中回答准确率达 78.3%，但定位 IoU 仅 12.9%。模型能大致判断答案位置，但不能精确输出坐标。

**Finding 3 - 复杂布局困难**：GPT-4o 无法识别重叠手写文字中的字符，旋转 90 度后的数字识别错误。DocVQA 图像旋转 90 度后 InternVL3-14B 准确率从 90.9% 骤降至 35.2%。

**Finding 4 - 结构化解析远弱于识别**：InternVL3-14B 在非配对实体匹配上达 94.4%，但在关键信息抽取（给定 entity 找 value）上仅 84.9%，在文档解析（需输出 Markdown/HTML）上更低。

**Finding 5 - 推理能力差异显著**：常识推理 72.9%、视觉文本理解 83.0%、模式识别 69.2%、计算 56.5%、专业知识 71.8%。

### 中文任务上的结果

| Model | Average |
|-------|---------|
| InternVL3-14B | **55.7** |
| Qwen2.5-VL-7B | 55.6 |
| Ovis2-8B | 49.2 |
| Gemini 1.5 Pro | 43.1 |
| Deepseek-VL2-Small | 42.7 |

### 与专用模型的对比

- **文字识别**：Qwen2.5-VL-7B (73.0%) > 传统 OCR 模型 SVTR (57.8%)
- **文本定位**：传统模型 TESTR (51.8 F1) >> 最好的 LMM Gemini-1.5-Pro (13.5 F1) —— 这是 LMM 最大的弱点之一

---

## 结论

OCRBench v2 揭示了当前 LMM 在 OCR 相关任务上的系统性弱点：它们擅长"读文字"，但严重缺乏"定位文字位置"、"解析复杂结构"和"基于文字推理"的能力。私有测试集的设计保证了评估的可靠性。该 benchmark 为 LMM 的文字理解能力开发提供了明确的方向指引。

---

## 思考

### 优点

1. **任务覆盖广**：8 项能力 x 23 个任务，是目前最全面的 OCR 评估框架
2. **难度设计合理**：大多数 LMM 得分在 50 以下，避免了 benchmark 饱和问题
3. **5 个发现深刻**：每个发现都有数据和案例支撑，对模型开发有直接指导意义
4. **私有测试集值得赞赏**：在数据污染问题普遍存在的当下，这是负责任的做法

### 缺点与待解决问题

1. **中文任务覆盖不均**：中文只有 5 项能力（无 referring/spotting/calculation），评估不完整
2. **L1 定位指标标注成本高**：带坐标的精确标注在扩展 benchmark 时成本大
3. **未涉及多页/长文档**：所有测试基于单张图片，不能评估多页文档理解能力
4. **评估指标的理解成本**：6 种指标让结果解释变得复杂

### 与已有 Wiki 的连接

- 关联概念：[[Wiki/Concepts/OCR]]、[[Wiki/Concepts/LMM]]、[[Wiki/Concepts/Benchmark]]
- 关联实体：[[Wiki/Entities/GPT-4o]]、[[Wiki/Entities/Qwen2.5-VL]]、[[Wiki/Entities/InternVL]]
- 关联比较：[[OmniDocBench]]（文档解析专项 benchmark）、[[olmOCR]]（PDF 线性化工具）

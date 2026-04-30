---
title: Language Models are Few-Shot Learners
authors:
  - Tom B. Brown
  - Benjamin Mann
  - Nick Ryder
  - Melanie Subbiah
  - Jared Kaplan
  - Prafulla Dhariwal
  - Arvind Neelakantan
  - Pranav Shyam
  - Girish Sastry
  - Amanda Askell
  - Sandhini Agarwal
  - Ariel Herbert-Voss
  - Gretchen Krueger
  - Tom Henighan
  - Rewon Child
  - Aditya Ramesh
  - Daniel M. Ziegler
  - Jeffrey Wu
  - Clemens Winter
  - Christopher Hesse
  - Mark Chen
  - Eric Sigler
  - Mateusz Litwin
  - Scott Gray
  - Benjamin Chess
  - Jack Clark
  - Christopher Berner
  - Sam McCandlish
  - Alec Radford
  - Ilya Sutskever
  - Dario Amodei
institutions: OpenAI
aliases:
  - GPT-3
  - GPT-3论文
date: '2026-04-30'
publish_date: 2020-05
venue: NeurIPS 2020
tags:
  - 论文
  - NLP
  - 大语言模型
  - 少样本学习
  - In-context Learning
  - Scaling Law
url: 'https://arxiv.org/abs/2005.14165'
code: 通过 API 提供
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：本文提出了 GPT-3，一个 175B 参数的自回归语言模型，通过海量数据上的预训练，在不进行任何梯度更新（仅通过上下文中的 few-shot 示例）的情况下，在众多 NLP 任务上展现出强大的性能，证明了"规模本身就是一种涌现"。

---

## Intro

### Motivation

近年来 NLP 的预训练+微调范式虽然有显著效果，但存在两个问题：
1. **需要大量标注数据**：每个下游任务都需要数千到数十万标注样本
2. **泛化能力有限**：微调后的模型往往在分布外数据上表现不佳

人类智能的特点是从极少量示例中即可学习新任务。作者希望通过扩大模型规模和训练数据量，让语言模型获得类似的能力——在推理时仅通过上下文中的少量示例（in-context learning）即可执行新任务，而无需任何梯度更新。

### 贡献

1. **训练了 175B 参数的 GPT-3**：比当时最大的模型大 10 倍
2. **系统研究 in-context learning**：在 zero-shot、one-shot、few-shot 三种设定下评估，无需微调
3. **在多个 NLP 任务上达到 SOTA**：包括翻译、QA、数学推理、新闻生成等 42 个 benchmark
4. **验证了 scaling 趋势**：随着模型从 125M 增大到 175B，few-shot 性能持续提升

---

## Method 核心方法

### 1. 模型架构

GPT-3 使用与 GPT-2 相同的架构（decoder-only Transformer），主要修改是 Sparse Attention 的替代方案：
- 层数：96
- d_model：12,288
- 注意力头数：96
- 上下文窗口：2,048 tokens
- 总参数：175B

8 种模型规模用于研究 scaling：

| 名称 | 参数量 | 层数 | d_model | 头数 |
| --- | --- | --- | --- | --- |
| GPT-3 Small | 125M | 12 | 768 | 12 |
| GPT-3 Medium | 350M | 24 | 1,024 | 16 |
| GPT-3 Large | 760M | 24 | 1,536 | 16 |
| GPT-3 XL | 1.3B | 24 | 2,048 | 24 |
| GPT-3 2.7B | 2.7B | 32 | 2,560 | 32 |
| GPT-3 6.7B | 6.7B | 32 | 4,096 | 32 |
| GPT-3 13B | 13B | 40 | 5,120 | 40 |
| GPT-3 175B | 175B | 96 | 12,288 | 96 |

### 2. 训练数据

总规模约 500B tokens（约 570GB 文本），来自 5 个来源：

| 数据集 | 占比 | 说明 |
| --- | --- | --- |
| Common Crawl（过滤后） | 60% | 使用高质量参考语料训练分类器过滤低质量页面 |
| WebText2 | 22% | 来自 Reddit 高赞链接的网页文本 |
| Books1 + Books2 | 8% + 8% | 互联网书籍语料 |
| Wikipedia | 3% | 英文维基百科 |

关键创新：用 Common Crawl 与高质量语料（WebText/Wikipedia/Books）的重叠度训练分类器，保留高质量文档，得到过滤后的 Common Crawl 语料。

### 3. Training Setup

- 跨 GPU 的模型并行 + 数据并行
- 使用与 GPT-2 类似的训练方法（Adam 优化器）
- 上下文长度：2,048 tokens
- 仅在预训练数据上训练，不做任何微调

### 4. 评估范式：In-context Learning

不进行梯度更新，通过上下文中的示例引导模型行为：

- **Zero-shot**：仅提供任务的自然语言描述，无示例
- **One-shot**：提供任务描述 + 1 个示例
- **Few-shot**：提供任务描述 + K 个示例（通常 K=10-100）

所有示例以纯文本形式拼接在 prompt 中，模型基于条件概率自回归生成答案。

---

## 实验/评估/结果

### 语言建模与 Cloze Tasks

- LAMBADA（zero-shot）：GPT-3 86.4%，超越之前 SOTA（68%）超过 18 个百分点
- HellaSwag：79.3%
- Penn Tree Bank：PPL 远超之前方法

### 问答

- TriviaQA（zero-shot）：64.3%，few-shot：**71.2%**（超越 fine-tuned SOTA）
- WebQuestions（few-shot）：41.5%
- Natural Questions（few-shot）：29.9%

### 翻译

- 所有规模下 few-shot 性能随模型增长持续提升
- 英译法、英译德、英译罗等方向表现最好
- 反向翻译（XX→EN）表现较弱：175B 模型反向翻译不敌监督学习小模型

### 常识推理

- PIQA：82.8%（few-shot）
- ARC（Easy）：68.8%，ARC（Challenge）：51.4%
- WinoGrande：70.2%

### 数学推理

- 在加/减法（2-5 位数）上随模型增大持续提升，175B 达到约 90%
- 但模型偶尔会犯非常 basic 的错误

### 新闻文章生成

- 人类评估中，175B GPT-3 生成的 ~500 词新闻文章，人类仅 52% 能准确区分（接近随机 50%）

### 局限与失败案例

- **自然语言推理（NLI）**：ANLI 上 GPT-3 几乎随机（约 40%），可能因为 NLI 需要推理两个句子的关系，in-context learning 难以掌握
- **部分阅读理解**：RACE-h 上 few-shot 仅 ~46.8%，远低于 SOTA
- **合成任务的不稳定性**：translation、算术等任务在部分设定下表现不稳定
- **生成样本存在重复**：上下文越长越容易重复

### 数据污染的讨论

论文专门讨论了训练数据与测试数据的重叠问题，发现某些数据集的 n-gram 重叠率较高，但去除重叠样本后性能下降通常较小，主要结论不受影响。

---

## 结论

GPT-3 证明了：将语言模型的规模提升至 175B 参数并在 ~500B tokens 上训练后，模型在不进行任何微调的情况下，通过 few-shot in-context learning 即可在广泛 NLP 任务上展现出强大的能力。这挑战了"必须为每个任务收集标注数据微调"的固有范式。同时也暴露了在需要深度推理的任务上的明显短板。

---

## 思考

### 优点

1. **规模驱动涌现的实证里程碑**：GPT-3 是第一个系统证明"规模本身就是一种涌现"的工作。从 125M 到 175B 的 8 种规格的 scaling 曲线，展示了 in-context learning 能力是如何随规模平滑增长的。

2. **In-context Learning 范式的开创**：在 GPT-3 之前，NLP 主流是 pre-train + fine-tune。GPT-3 证明了不做微调也能执行任务——仅靠 prompt 中的示例即可。这催生了整个 prompt engineering 和 in-context learning 研究领域。

3. **训练数据工程的先驱**：从 Common Crawl 这一"脏数据"中提取高质量训练数据的方法（用分类器过滤），成为后续大模型训练数据工程的标准做法。

4. **对数据污染问题的坦诚**：论文公开讨论了训练-测试重叠问题并做了细致的分析（n-gram 重叠、性能影响），这在当时并不多见。

### 缺点与局限

1. **推理能力弱**：在 NLI、复杂阅读理解等需要深度推理的任务上表现差。这说明"模式匹配"和"推理"之间存在鸿沟，仅靠规模化不能完全弥合。

2. **弱在反向任务**：翻译（XX→EN 比 EN→XX 弱得多）暴露了模型本质上是"看多了再复述"，不是真正的双语理解。

3. **生成质量与可控性**：新闻文章生成虽然难以分辨，但模型缺乏对事实的保证，生成内容可能包含虚假信息。

4. **成本极高**：175B 参数的训练和推理成本对大多数学术机构不可承受，加剧了 AI 研究的不平等。

5. **Ethics/偏见**：论文讨论了模型可能包含的偏见和潜在的滥用风险，但讨论深度有限。

### 与已有 Wiki 的连接

- 关联概念：[[In-context Learning]]、[[Few-shot Learning]]、[[Scaling Law]]、[[Decoder-only Transformer]]
- 关联论文：[[AI阅读笔记/BERT]]（双向 vs 单向架构之争）、[[AI阅读笔记/InstructGPT]]（GPT-3 的指令跟随改进）、[[AI阅读笔记/Chinchilla 缩放定律]]（对 scaling law 的修正）
- 后续演进：GPT-3 → InstructGPT/GPT-3.5 → GPT-4
- 关联问题：[[Wiki/Questions/为什么 decoder-only 架构最终胜出]]

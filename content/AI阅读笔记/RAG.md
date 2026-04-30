---
title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks'
authors:
  - Patrick Lewis
  - Ethan Perez
  - Aleksandra Piktus
  - Fabio Petroni
  - Vladimir Karpukhin
  - Naman Goyal
  - Heinrich Küttler
  - Mike Lewis
  - Wen-tau Yih
  - Tim Rocktäschel
  - Sebastian Riedel
  - Douwe Kiela
institutions: 'Facebook AI Research / University College London / New York University'
aliases:
  - RAG
  - RAG论文
  - 检索增强生成
date: '2026-04-30'
publish_date: 2020-05
venue: NeurIPS 2020
tags:
  - 论文
  - NLP
  - 检索增强
  - QA
  - 知识密集型
url: 'https://arxiv.org/abs/2005.11401'
code: 已开源（https://github.com/huggingface/transformers 中集成）
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：本文提出了 RAG（Retrieval-Augmented Generation）模型，将预训练的 seq2seq 模型（parametric memory）与基于 DPR 的 Wikipedia 向量索引（non-parametric memory）相结合，端到端训练，在多个开放域 QA 任务上达到 SOTA，并证明了 RAG 生成的回答比纯参数模型更具体、多样和事实准确。

---

## Intro

### Motivation

大语言模型虽然能在参数中存储知识，但存在根本性局限：
1. **知识难以更新和修正**：参数化的知识嵌入在权重中，无法灵活更新
2. **缺乏可解释性**：无法追溯预测的知识来源
3. **容易产生幻觉（hallucination）**：尤其是对长尾知识
4. **知识密集型任务上的表现不如 task-specific 架构**

作者认为，将预训练的生成模型与可显式访问的非参数化记忆相结合，可以同时解决上述问题。

### 贡献

1. **提出 RAG**：将预训练 seq2seq model（BART）+ 预训练 retriever（DPR）+ Wikipedia 向量索引组合成统一的概率模型，端到端训练
2. **两种变体**：RAG-Sequence（整个序列使用同一检索文档）和 RAG-Token（每个 token 可用不同检索文档）
3. **三个开放域 QA 任务 SOTA**：Natural Questions、WebQuestions、CuratedTrec
4. **知识更新能力**：无需重新训练，仅替换文档索引即可更新模型的世界知识
5. **生成质量提升**：RAG 生成的回答比纯 BART 更具体（specific）、更多样（diverse）和更事实（factual）

---

## Method 核心方法

![](https://arxiv.org/html/2005.11401/x1.png)

*Figure 1: RAG 架构概览。将预训练的检索器（Query Encoder + Document Index）与预训练的 seq2seq 生成器（Generator）结合，端到端微调。对于 query x，使用 MIPS 检索 top-K 文档 z_i，最终的预测 y 通过边缘化不同文档下的生成结果得到。*

### 1. 模型架构

RAG 的核心是：**将检索文档作为潜在变量（latent variable），在 seq2seq 生成过程中通过边缘化（marginalization）来使用检索到的知识**。

组件：
- **Retriever**：Dense Passage Retriever (DPR)，基于 BERT_BASE 双编码器，预计算 Wikipedia 文档的向量索引（21M 文档，每个 100 词）
- **Generator**：BART-large（400M 参数），预训练的 seq2seq Transformer
- **Non-parametric Memory**：Wikipedia 全文向量索引，通过 FAISS 的 MIPS（最大内积搜索）实现快速检索

### 2. RAG-Sequence

整个生成序列使用同一个检索文档。概率模型为：

$$p_{\text{RAG-Sequence}}(y|x) \approx \sum_{z \in \text{top-}k} p_{\eta}(z|x) p_{\theta}(y|x,z) = \sum_{z \in \text{top-}k} p_{\eta}(z|x) \prod_{i}^{N} p_{\theta}(y_i|x,z,y_{1:i-1})$$

解码时：对每个 top-K 文档分别做 beam search，然后对 beam 中的假设做边缘化（Thorough Decoding）或近似边缘化（Fast Decoding）。

### 3. RAG-Token

每个目标 token 可以依赖不同的检索文档。概率模型为：

$$p_{\text{RAG-Token}}(y|x) \approx \prod_{i}^{N} \sum_{z \in \text{top-}k} p_{\eta}(z|x) p_{\theta}(y_i|x,z,y_{1:i-1})$$

解码更简单：将每个 token 的生成概率替换为对所有文档的边缘化概率，然后标准 beam search。

### 4. 训练

端到端训练，无检索监督信号（不需要标注"哪个文档是正确的"）：
- 文档编码器 BERT_d 冻结（避免频繁重建索引）
- Query 编码器 BERT_q 和 BART 生成器可训练
- 损失函数：负对数似然，Adam 优化器

### 5. Wiki 知识源更新

非参数记忆可以随时替换：将 Wikipedia 索引从 2018 年版本替换为 2022 年版本，模型无需重新训练即可适应新知识。

---

## 实验/评估/结果

### 开放域 QA（核心结果）

![](https://arxiv.org/html/2005.11401/x3.png)

*Figure 3: （左）NQ 性能随检索文档数的变化；（中）NQ 检索召回率；（右）MS-MARCO Bleu-1 和 Rouge-L 随检索文档数的变化。RAG 仅需少量文档即可达到高召回率。*

| 数据集 | RAG | 之前 SOTA | 说明 |
| --- | --- | --- | --- |
| Natural Questions | **SOTA** | DPR / REALM | 生成式回答首次超越抽取式 |
| TriviaQA | **SOTA** | T5-11B | 无需专门预训练即超越 |
| WebQuestions | **SOTA** | DPR | |
| CuratedTrec | **SOTA** | DPR | |

关键发现：尽管这些任务通常被设计为"抽取式"（从文档中提取答案 span），RAG 的**自由形式生成**反而优于之前的抽取式方法。

### Abstractive QA (MS-MARCO)

- RAG 在 MS-MARCO NLG 任务上生成更具体、更多样的回答
- 比 BART baseline 的回答事实性更高
- 即使某些问题在 Wikipedia 中找不到答案，RAG 仍能依赖参数化知识生成合理回答

### Jeopardy 问题生成

- RAG 生成的 Jeopardy 问题比 BART 更具体、更多样
- 人类评估：RAG 生成的问题比 BART 更"factual"和"specific"

### FEVER 事实验证

- RAG 在 FEVER 上达到与使用强检索监督的 pipeline 模型仅差 4.3% 的性能
- 说明了 RAG 在不需要检索监督的情况下即可接近监督检索系统

### 消融实验

- 同时训练 retriever 和 generator 优于仅训练 generator
- Top-K 从 5 增加到 10 带来小幅提升
- RAG-Sequence vs RAG-Token：性能接近，任务依赖

---

## 结论

RAG 开创了"检索增强生成"的范式——将语言模型的知识输出与外部知识库的检索相结合。这个范式解决了纯参数模型知识难以更新、缺乏可解释性、容易幻觉等问题。RAG 在知识密集型任务上达到了 SOTA，并证明了其生成质量在事实性和多样性方面的优势。

---

## 思考

### 优点

1. **范式的开创性**：RAG 不是对某个具体架构的改进，而是在方法论层面提出了一个新范式："检索+生成"的混合架构。这篇论文成为 retrieval-augmented generation 整个子领域的奠基之作。

2. **端到端训练不依赖检索监督**：retriever 和 generator 联合训练不需要标注"哪个文档相关"——这正是为什么它叫 latent variable model。这在实际中非常重要，因为获取检索标注的成本极高。

3. **模块化的优雅设计**：retriever 和 generator 各自基于已有预训练组件（DPR 和 BART），通过概率公式联结起来。这种"组合已有组件创建新系统"的方法是 modern AI 工程的核心范式。

4. **知识更新的实际价值**：无需重新训练即可更新模型的世界知识——这在知识不断更新的实际应用中价值巨大。

### 缺点与局限

1. **只检索 Wikipedia**：知识源仅限 Wikipedia，覆盖范围有限。现实世界的知识密集型任务需要更多样化的知识源（代码库、企业文档、实时信息等）。

2. **检索器质量的基础限制**：RAG 的性能很大程度上受限于 DPR。如果前 K 个文档不包含正确答案，生成器无论如何也无法产出正确答案——这是所有 RAG 系统的根本瓶颈。

3. **生成器-检索器的联合训练不够深**：虽然论文声称端到端训练，但文档编码器是冻结的。这意味着 retrieval index 不能随着生成器的学习而优化，存在优化 gap。

4. **上下文窗口受限**：BART 的输入窗口有限，只能拼接有限数量的检索文档。后来随着长上下文 LLM 的发展，这个问题得到了缓解。

### 与已有 Wiki 的连接

- 关联概念：[[RAG（检索增强生成）]]、[[DPR（Dense Passage Retrieval）]]、[[Non-parametric Memory]]、[[Latent Variable Model]]
- 关联论文：[[AI阅读笔记/GPT-3]]（RAG 解决的"知识存储"问题在 GPT-3 中非常突出）
- 后续演进：REALM、Atlas、RETRO、LlamaIndex、LangChain 等 RAG 工程框架
- 关联问题：[[Wiki/Questions/RAG 和长上下文 LLM 哪个是知识密集型任务的最优解]]

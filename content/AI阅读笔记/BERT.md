---
title: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding'
authors:
  - Jacob Devlin
  - Ming-Wei Chang
  - Kenton Lee
  - Kristina Toutanova
institutions: Google AI Language
aliases:
  - BERT
  - BERT论文
date: '2026-04-30'
publish_date: 2018-10
venue: NAACL 2019
tags:
  - 论文
  - NLP
  - 预训练
  - 双向Transformer
  - MLM
  - 微调
url: 'https://arxiv.org/abs/1810.04805'
code: 已开源（https://github.com/google-research/bert）
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：本文提出了 BERT，一种基于深度双向 Transformer 编码器的语言表示模型，通过 Masked Language Model（MLM）和 Next Sentence Prediction（NSP）两种无监督预训练任务，在 11 项 NLP 任务上刷新了 SOTA，开创了"预训练+微调"的 NLP 范式。

![](https://arxiv.org/html/1810.04805v2/x1.png)

*Figure 1: BERT 的预训练和微调流程。除输出层外，预训练和微调使用相同的架构。*

---

## Intro

### Motivation

2018 年，语言模型预训练已显示出对 NLP 任务的有效性。当时有两种主要策略：
1. **Feature-based**（ELMo）：将预训练表示作为额外特征注入任务特定模型
2. **Fine-tuning**（OpenAI GPT）：用预训练参数初始化，在目标任务上微调所有参数

但两者共用一个关键局限：都使用**单向**语言模型进行预训练。这限制了表示能力，尤其对 token 级任务（如 QA）而言，双向上下文至关重要。

### 贡献

1. **提出 MLM 预训练目标**：通过随机掩码部分 token 并基于上下文预测，实现了深度双向预训练，克服了标准语言模型只能用单向训练的固有限制
2. **统一的预训练-微调框架**：BERT 在预训练和微调阶段使用几乎相同的架构，仅需一个额外输出层即可适配各种下游任务
3. **在 11 项 NLP 任务上达到 SOTA**：GLUE 80.5%（+7.7%），MultiNLI 86.7%，SQuAD v1.1 F1 93.2，SQuAD v2.0 F1 83.1
4. **证明了模型规模的重要性**：BERT_LARGE（L=24, H=1024, A=16, 340M）在所有任务上显著优于 BERT_BASE（L=12, H=768, A=12, 110M）

---

## Method 核心方法

BERT 的方法论由四个紧密关联的部分组成：双向 Transformer 架构 → 三合一输入表示 → 双任务预训练（MLM+NSP） → 统一微调框架。

### 1. 模型架构：双向 Transformer 编码器

BERT 使用多层双向 Transformer 编码器，与原始 Transformer 的编码器几乎相同（仅调整了层数、宽度和头数）：

| 模型 | L | H | A | 参数量 | FFN 维度 |
|------|---|---|---|--------|---------|
| BERT_BASE | 12 | 768 | 12 | 110M | 3072 |
| BERT_LARGE | 24 | 1024 | 16 | 340M | 4096 |

BERT_BASE 与 OpenAI GPT 参数量完全一致——这是一个精心设计的对比：架构大小相同，唯一差异是**双向 vs 单向**。

**与 GPT/ELMo 的架构对比（原文 Figure 3）**：

| 模型 | 架构 | 注意力方向 | 预训练目标 |
|------|------|-----------|-----------|
| **BERT** | Transformer Encoder | **双向**（全层左右上下文） | MLM + NSP |
| OpenAI GPT | Transformer Decoder | 单向（仅左→右） | 自回归 LM |
| ELMo | 独立 LSTM（左→右 + 右→左） | 浅层拼接（非深层交互） | 双向 LM |

BERT 是**唯一在所有层中联合使用左右双向上下文**的模型——GPT 的单向限制使每个 token 只能基于前文预测自己；ELMo 的左右 LSTM 独立训练、仅在顶层拼接，缺少深层交互。

### 2. 输入表示：三合一 Embedding

使用 WordPiece tokenizer（30K 词表）。每个 token 的输入表示是三个 embedding 的**逐元素求和**：

$$\text{Input} = E_{\text{token}} + E_{\text{segment}} + E_{\text{position}}$$

- **Token Embedding**：子词嵌入（WordPiece 将未知词拆分，如 "playing"→"play"+"##ing"）
- **Segment Embedding**：EA 或 EB，标识属于句子 A 还是 B（对 NSP 任务关键）
- **Position Embedding**：可学习的位置嵌入，支持最大 512 位置

**特殊 token**：

| Token | 位置 | 用途 |
|-------|------|------|
| `[CLS]` | 每个序列首位 | 最终隐藏状态 $C \in \mathbb{R}^H$ 聚合全序列信息，用于分类 |
| `[SEP]` | 句子 A 末 + 句子 B 末 | 分隔句子，同时句子分隔信号送达所有层 |
| `[MASK]` | 被遮位置（仅预训练） | MLM 目标 |

输入序列结构：`[CLS] Tok_A ... [SEP] Tok_B ... [SEP]`。对单句任务，仅用句子 A。

### 3. 预训练任务 1：Masked Language Model（核心创新）

**问题**：标准自回归 LM 天然是单向的（每个 token 预测下一个）。要实现双向，简单的"双向预测"不可行——每个 token 在多层的间接路径中会"看到自己"。

**解决**：随机 mask 15% 的输入 token，基于未 mask 的上下文预测被 mask 位置的原 token。由于 mask token 被替换为 `[MASK]`，模型无法"看到自己"。

**80-10-10 策略**（解决 pretrain-finetune mismatch）：

| 比例 | 操作 | 目的 |
|------|------|------|
| 80% | 替换为 `[MASK]` | 主要训练信号 |
| 10% | 替换为**随机** token | 教会模型"纠错"——不仅是填空 |
| 10% | **保持不变** | 偏向真实 token 分布（微调时无 [MASK]） |

仅对选中的 15% token 做以上操作——实际被替换为 [MASK] 的仅约 12%（15%×80%）。损失仅计算被 mask 位置（非全部 token），使用交叉熵。

### 4. 预训练任务 2：Next Sentence Prediction

构造句子对 (A, B)：50% B 是真实的下一句（IsNext），50% B 是随机句子（NotNext）。`[CLS]` 的最终向量 C 通过二分类器预测。NSP 教会模型理解**句子间关系**——对 QA（问题-段落）和 NLI（前提-假设）任务至关重要。

**消融证明**（原文 Table 5）：去掉 NSP 导致 QNLI（-3.5）和 SQuAD（-0.6 F1）显著下降。NSP 不是 universally beneficial——后来 RoBERTa 证明去掉 NSP 在某些任务上更好，但 BERT 首次建立了这个方向。

### 5. 预训练设置

| 配置 | 值 |
|------|-----|
| 数据 | BooksCorpus（800M 词）+ English Wikipedia（2500M 词）——**仅文档级**语料 |
| 优化器 | Adam（β₁=0.9, β₂=0.999, ε=1e-6），L2 weight decay 0.01 |
| 学习率 | warmup 10K steps → peak 1e-4 → linear decay |
| Batch | 256 序列（BASE：16 TPU 4 天；LARGE：64 TPU 4 天） |
| Dropout | 0.1（所有层） |
| 激活函数 | GELU（非 ReLU） |
| 序列长度 | 512 tokens（90% 步）→ 128 tokens（剩余 10%，加速） |

### 6. 微调：统一框架适配所有任务

微调成本极低——所有实验可在单 Cloud TPU 上 1 小时内复现，或在 GPU 上几小时完成。

| 任务类型 | 输入 | 输出头 | 代表任务 |
|---------|------|--------|---------|
| **单句分类** | [CLS] + 句子 | C → 线性分类器 | SST-2, CoLA |
| **句对分类** | [CLS] + A + [SEP] + B | C → 线性分类器 | MNLI, QQP, RTE, MRPC |
| **QA** | [CLS] + 问题 + [SEP] + 段落 | 每个 token 向量 → start/end 点积+softmax | SQuAD v1.1/v2.0 |
| **序列标注** | [CLS] + 句子 | 每个 token 向量 → 分类器 | CoNLL-2003 NER |

**超参数**：微调 epoch 数少（2-4 epochs），学习率小（2e-5 到 5e-5），batch 16 或 32。微调对超参数选择有一定敏感性，建议多试几组。

---

## 实验/评估/结果

### GLUE（通用语言理解评估）

| 系统 | MNLI | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pre-OpenAI SOTA | 80.6 | 66.1 | 82.3 | 93.2 | 35.0 | 81.0 | 86.0 | 61.7 | 74.0 |
| OpenAI GPT | 82.1 | 70.3 | 87.4 | 91.3 | 45.4 | 80.0 | 82.3 | 56.0 | 75.1 |
| BERT_BASE | 84.6 | 71.2 | 90.5 | 93.5 | 52.1 | 85.8 | 88.9 | 66.4 | 79.6 |
| BERT_LARGE | **86.7** | **72.1** | **92.7** | **94.9** | **60.5** | **86.5** | **89.3** | **70.1** | **82.1** |

BERT_LARGE 在所有 8 个 GLUE 任务上实现 SOTA，平均提升 7.0%。

### SQuAD（阅读理解）

| 系统 | Dev F1 | Test F1 |
| --- | --- | --- |
| Human | - | 91.2 |
| #1 Ensemble (nlnet) | - | 91.7 |
| BERT_LARGE (Single) | 90.9 | - |
| BERT_LARGE (Ensemble) | 91.8 | - |
| BERT_LARGE (Sgl.+TriviaQA) | 91.1 | 91.8 |
| BERT_LARGE (Ens.+TriviaQA) | 92.2 | **93.2** |

- SQuAD v2.0：单模型 Test F1 83.1，比之前最佳系统提升 +5.1 F1

### SWAG（常识推理）

- BERT_LARGE：Dev 86.6%，Test 86.3%（超越人类专家 85.0%）
- 比 OpenAI GPT 提升 +8.3%

### 消融实验

#### 预训练任务的重要性

| 任务变体 | MNLI-m | QNLI | MRPC | SST-2 | SQuAD F1 |
| --- | --- | --- | --- | --- | --- |
| BERT_BASE | 84.4 | 88.4 | 86.7 | 92.7 | 88.5 |
| 无 NSP | 83.9 | 84.9 | 86.5 | 92.6 | 87.9 |
| LTR + 无 NSP（类 GPT） | 82.1 | 84.3 | 77.5 | 92.1 | 77.8 |
| LTR + 无 NSP + BiLSTM | 82.1 | 84.1 | 75.7 | 91.6 | 84.9 |

- 去掉 NSP 对 QNLI、SQuAD 影响显著
- 双向 MLM 相比单向 LM：MRPC +9.2，SQuAD +10.7 的巨大优势
- 即使给单向模型加 BiLSTM，SQuAD F1 仍比 BERT 低 3.6

#### 模型规模的效果

| L | H | A | LM PPL | MNLI-m | MRPC | SST-2 |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 768 | 12 | 5.84 | 77.9 | 79.8 | 88.4 |
| 6 | 768 | 12 | 4.68 | 81.9 | 84.8 | 91.3 |
| 12 | 768 | 12 | 3.99 | 84.4 | 86.7 | 92.9 |
| 12 | 1024 | 16 | 3.54 | 85.7 | 86.9 | 93.3 |
| 24 | 1024 | 16 | 3.23 | 86.6 | 87.8 | 93.7 |

结论：增大模型规模在所有数据集上都有严格的正向收益，即使在 MRPC（仅 3,600 个训练样本）上也是如此。

#### Feature-based vs Fine-tuning（NER on CoNLL-2003）

- Fine-tuning BERT_LARGE：Test F1 92.8
- 拼接 BERT_BASE 最后 4 层的固定特征：Dev F1 96.1（仅比微调低 0.3 F1）

证明 BERT 在 feature-based 和 fine-tuning 两种方法下都很有效。

---

## 结论

BERT 证明了深度双向预训练表示可以大幅提升 NLP 任务性能。通过 MLM 解决了双向预训练的"自己看到自己"问题，NSP 让模型理解句子间关系。统一的预训练-微调框架让 BERT 可以轻易适配各种任务。关键洞察是：在足够充分的预训练下，即使是数据量很小的下游任务也能从更大的模型中获益。

---

## 思考

### 优点

1. **范式级创新**：BERT 不是简单地改架构，而是重新定义了 NLP 的工作流程——"在海量无标注文本上预训练一个大模型，然后在各种任务上微调"。这个"预训练+微调"范式统治了 NLP 数年。

2. **MLM 的巧妙设计**：用 80/10/10 的 mask 策略解决 pre-train/fine-tune mismatch，看似简单的 trick 但对双向训练至关重要。

3. **实验极其系统和全面**：消融实验从预训练任务、模型规模、feature-based vs fine-tuning 三个维度深刻揭示了 BERT 为什么有效。

4. **规模重要性的早期证明**：Table 6 是先驱性的——它在 2018 年就证明了"更大模型 = 更好效果"的 scaling law 趋势，预示了后来 GPT-3 等超大规模模型的路线。

### 缺点与局限

1. **NSP 的有效性后来被质疑**：RoBERTa（2019）等后续研究发现去掉 NSP 在某些情况下反而更好。NSP 可能过于简单，模型并未学到真正的句子间推理。

2. **编码器架构的局限**：BERT 只用了 Transformer 编码器，天生不适合生成任务。后来 GPT 系列的 decoder-only 路线在生成和通用能力上证明了优势。

3. **固定长度的位置编码**：BERT 使用可学习的绝对位置 embedding，无法泛化到比训练时更长的序列。

4. **掩码策略的效率问题**：15% 的 token 参与训练，相比标准 LM（100% token 参与）需要更多训练步数才能收敛。

5. **`[CLS]` 作为聚合表示的局限**：只用第一个 token 表示整个序列可能丢失信息，后续工作提出了更好的池化策略。

### 与已有 Wiki 的连接

- 关联概念：[[MLM（Masked Language Model）]]、[[预训练+微调]]、[[Feature-based vs Fine-tuning]]
- 关联论文：[[AI阅读笔记/Transformer]]（BERT 的架构基础）、[[AI阅读笔记/GPT-3]]（与 BERT 不同方向的路线选择）
- 后续演进：RoBERTa、ALBERT、ELECTRA 等对 BERT 的改进

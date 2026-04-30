---
type: source
status: processed
site: arXiv
source_file: >-
  [[Clippings/BERT Pre-training of Deep Bidirectional Transformers for Language
  Understanding.md]]
title: >-
  BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding
author: Jacob Devlin et al. (Google AI Language)
published: '2018'
processed: '2026-04-30'
tags:
  - BERT
  - pretraining
  - NLP
  - bidirectional
  - MLM
---
# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## 核心结论

- 提出 **BERT**（Bidirectional Encoder Representations from Transformers），通过在所有层中联合左右上下文来预训练深度双向表示。
- 11 项 NLP 任务上达到新 SOTA：GLUE 80.5%（+7.7%），MultiNLI 86.7%（+4.6%），SQuAD v1.1 F1 93.2，SQuAD v2.0 F1 83.1。
- 预训练 + 微调范式：一个预训练模型只需增加一个额外的输出层即可适配各种下游任务，无需任务特定的架构修改。
- BERT 的「双向」特性是关键优势——与 GPT 的左到右自回归方式形成对比。

## 关键方法或创新点

- **Masked Language Model (MLM)**：随机 mask 15% 的 token（80% [MASK] + 10% 随机词 + 10% 保留），让模型从双向上下文预测被 mask 的词。
- **Next Sentence Prediction (NSP)**：预训练时 50% 概率给连续的句子对，50% 概率给随机句子对，模型判断是否为下一句。
- 使用 **Transformer Encoder**（非 Decoder），结构为 L 层 + H 维隐藏层 + A 个注意力头。两个版本：BERT_base（L=12, H=768, A=12, 110M）和 BERT_large（L=24, H=1024, A=16, 340M）。
- 输入表示为 Token Embeddings + Segment Embeddings + Position Embeddings 之和。
- 训练数据：BooksCorpus（800M 词）+ 英文 Wikipedia（2500M 词）。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Entities/Transformer]]、[[Wiki/Topics/大语言模型基础]]
- BERT 开启了 NLP 的「预训练 + 微调」时代，与后来的 [[Wiki/Concepts/GPT 系列模型]] 形成编码器-解码器两大路线。
- MLM 预训练目标成为后续编码器模型的标配（RoBERTa、DeBERTa 等）。

## 局限或注意事项

- [MASK] token 在预训练和微调间存在 mismatch（微调时不会出现 [MASK]），论文通过 10% 随机词替换来缓解。
- NSP 后来在 RoBERTa 中被证明并非必要，可以被移除。
- BERT 不适合生成任务（自回归解码）。
- 输入长度限制 512 token，长文本需要分段处理。

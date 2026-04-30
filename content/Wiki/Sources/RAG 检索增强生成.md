---
type: source
status: processed
source_file: '[[Clippings/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks]]'
title: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
site: arxiv
author: Patrick Lewis et al. (Facebook AI Research)
published: '2020'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - RAG
  - 检索增强生成
  - 知识密集
  - DPR
  - Meta
aliases:
  - RAG
---
# RAG 检索增强生成

## 核心结论

Facebook/Meta AI 提出 RAG（Retrieval-Augmented Generation），将预训练的参数化记忆（seq2seq 模型）与非参数化记忆（Wikipedia 稠密向量索引）结合，用于知识密集型 NLP 任务。RAG 在开放域 QA、事实验证、知识增强生成等任务上达到 SOTA，且具有知识热更新、来源可追溯的优势。

## 关键事实

- 作者：Patrick Lewis、Ethan Perez 等（Facebook AI Research & UCL & NYU），2020
- 两种公式：(1) RAG-Sequence：同一检索文档用于整个解码序列；(2) RAG-Token：每个 token 可依赖不同检索文档
- 检索器：DPR（Dense Passage Retrieval），文档编码器为 BERT-base
- 生成器：BART-large（400M 参数）
- 非参数记忆：Wikipedia dump 的稠密向量索引（~21M 文档片段）
- 知识更新：只替换向量索引即可实现知识热更新，无需重新训练

## 方法或论证路径

- 输入 x + 检索的 top-k 文档 z → 生成输出 y
- 训练：联合微调检索器和生成器（端到端训练检索器是当时的关键贡献）
- 评估：开放域 QA（Natural Questions、TriviaQA、WebQuestions）、事实验证（FEVER）、MS-MARCO
- 消融实验：展示检索器质量、文档数量 k、RAG-Sequence vs RAG-Token 的影响

## 与现有 Wiki 的关系

- 关联：[[Wiki/Concepts/RAG 检索增强生成]]、[[Wiki/Topics/大语言模型基础]]
- 补充：RAG 是当前 LLM Agent 工具使用（检索 API）、知识库问答的核心范式来源
- 对比：与纯生成模型（GPT 系列）的内化知识形成互补——RAG 擅长事实准确性，GPT 擅长流畅性

## 可能的矛盾或待核实点

- 论文仅验证了 QA 和事实验证任务，在代码生成、多模态等场景的效果需后续工作补全
- 端到端训练检索器 vs 冻结检索器的收益在更强大生成器上是否仍然成立

## 后续问题

- RAG 与长上下文模型（128K+ token）的关系：长上下文是否可以取代检索？
- RAG 在多模态场景（RAG+Image Generation）的扩展

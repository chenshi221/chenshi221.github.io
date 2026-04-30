---
type: concept
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - RAG
  - 检索增强生成
  - Retrieval-Augmented Generation
tags:
  - RAG
  - 检索增强
  - 知识密集
  - DPR
  - 幻觉
  - 可追溯性
sources:
  - '[[Wiki/Sources/RAG 检索增强生成]]'
confidence: high
---
# RAG 检索增强生成

## 定义

RAG（Retrieval-Augmented Generation，检索增强生成）是一种将信息检索与文本生成结合的技术范式。模型在生成回答时，不仅依赖自身参数化知识（训练时学到的），还从外部知识库（如 Wikipedia）检索相关文档作为上下文。

由 Facebook/Meta AI 于 2020 年首次系统化提出。

## 核心架构

RAG 包含两个关键组件：

1. **检索器（Retriever）**：从外部知识库中检索相关文档
   - 原始论文使用 DPR（Dense Passage Retrieval），双编码器（query encoder + document encoder）
   - 知识库：Wikipedia 稠密向量索引，约 2100 万个文档片段
   
2. **生成器（Generator）**：基于输入 + 检索文档生成回答
   - 原始论文使用 BART-large（seq2seq 架构）
   - 现代 RAG 使用 LLM（GPT-4、Llama 等）

### 两种公式

- **RAG-Sequence**：同一组检索文档用于整个生成序列（全局决策）
- **RAG-Token**：每个 token 可以依赖不同检索文档（细粒度决策）

## 为什么 RAG 重要

RAG 解决了纯 LLM 的几个核心限制：

| 问题 | 纯 LLM | RAG |
|------|--------|-----|
| 幻觉 | 可能生成不存在的事实 | 事实由检索文档约束 |
| 知识截止 | 训练数据截止日期后无法更新 | 只需更新知识库索引 |
| 可追溯性 | 无法验证信息来源 | 每个回答可附带来源引用 |
| 领域专业知识 | 需要微调 | 注入领域知识库即可 |

## 后续演进

- **REALM**：端到端训练检索器（Google, 2020）
- **RETRO**：大规模检索增强预训练（DeepMind, 2021）
- **Atlas**：少量样本学习的检索增强（Meta, 2022）
- **长上下文 LLM vs RAG**：128K+ token 上下文模型能否取代检索？目前共识是两者互补——长上下文适合单文档分析，RAG 适合海量文档检索
- **RAG + Agent**：Agent 框架中检索作为工具之一（如 ChatGPT with browsing）

## 来源

- [[Wiki/Sources/RAG 检索增强生成]] — 原始论文（Lewis et al., 2020）
- 关联：[[Wiki/Topics/大语言模型基础]]、[[Wiki/Concepts/LLM Agent 架构]]

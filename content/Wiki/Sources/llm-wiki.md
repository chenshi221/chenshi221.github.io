---
type: source
status: processed
source_file: "[[Clippings/llm-wiki.md]]"
title: "llm-wiki"
site: "GitHub Gist (karpathy)"
author: "Andrej Karpathy"
published: ""
processed: "2026-04-29"
updated: "2026-04-29"
tags: [meta, methodology, knowledge-management]
aliases: [LLM Wiki, Karpathy Wiki, llm-wiki]
---

# llm-wiki

> Andrej Karpathy 提出的 LLM 维护个人知识库模式。来源：[[Clippings/llm-wiki.md]]

## 核心结论

- 传统 RAG 在每个查询时重新从原始文档中检索和综合知识，没有积累效应。LLM Wiki 模式通过 LLM **增量构建和维护一个持久的 wiki**，让知识持续复合增长。
- 关键区别在于：wiki 是一个 **持久、可累积的产物**，交叉引用、矛盾标注和综合理解已经预先存在，无需每次重新推导。
- 人类负责策展、提问和判断方向；LLM 负责所有归纳、交叉引用、归档和簿记工作——这些正是导致传统 wiki 被放弃的维护负担。

## 关键事实

### 三层架构

1. **原始来源层 (Raw sources)** — 不可变的源文档集合（文章、论文等），LLM 只读不修改。
2. **Wiki 层** — LLM 生成和维护的 Markdown 文件目录（摘要、实体页、概念页、比较等）。LLM 完全拥有此层。
3. **Schema 层** — 告诉 LLM wiki 结构、约定和工作流的配置文件（如 CLAUDE.md）。人类和 LLM 共同演进此层。

### 三大操作

- **Ingest（摄取）** — 放入新来源 → LLM 读取、讨论关键点、写摘要页、更新索引和相关页面、追加日志。单个来源可能触及 10-15 个 wiki 页面。
- **Query（查询）** — 提问题 → LLM 搜索相关页面、阅读、综合并带引用回答。**好的回答本身可以沉淀为新的 wiki 页面**。
- **Lint（体检）** — 周期性健康检查：矛盾、过时声明、孤立页面、缺失概念、数据缺口。

### 索引与日志

- **index.md** — 内容导向，按类别列出所有 wiki 页面及其一句话摘要。LLM 在每个摄取后更新。在几百页规模下效果良好，无需 embedding 检索基础设施。
- **log.md** — 时间线导向，只追加记录操作历史。使用统一前缀格式（如 `## [YYYY-MM-DD] ingest | 标题`）可用 Unix 工具解析。

## 与现有 Wiki 的关系

- **元层指导**：此文档描述的架构正是本 vault 的 CLAUDE.md 所实现的模式。Claude 作为 LLM 维护者，本 wiki 是产物，CLAUDE.md 是 schema 层。
- **关联**：[[Wiki/Concepts/llm-wiki-pattern]] — 此模式的更详细概念阐述
- **支持**：本 vault 的目录结构和 CLAUDE.md 工作流完全遵循此模式

## 应用场景

Karpathy 列举了多种适用场景，其中与本 vault 最相关的是：

- **研究**（本 vault 主要用途）— 在数周或数月内深入某个话题，阅读论文、文章、报告，逐步构建综合 wiki
- **个人知识管理** — 整理笔记、日志、文章摘录

## 可能的矛盾或待核实点

- Karpathy 认为在 ~100 个来源、~几百页规模下 index 文件足以导航。需要验证随着本 vault 增长是否仍然成立。
- "LLM 从来不会厌倦，不会忘记更新交叉引用"——实际使用中可能存在 LLM 上下文窗口限制、一致性丢失等问题，需要长期验证。

## 后续问题

- 当 wiki 规模超过 index 文件的可管理范围时，是否需要引入搜索工具（如 qmd）？
- 如何确保 LLM 在长期维护中保持 wiki 的一致性，避免概念漂移？
- Karpathy 提到"你可能也想要批量摄取多个来源"，本 vault 的单文件逐个摄取 vs 批量摄取的取舍？

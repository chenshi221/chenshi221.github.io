---
type: concept
status: active
created: "2026-04-29"
updated: "2026-04-29"
aliases: [LLM Wiki, LLM-maintained knowledge base, persistent wiki pattern, Karpathy wiki]
tags: [meta, methodology, knowledge-management, llm]
sources:
  - "[[Wiki/Sources/llm-wiki]]"
confidence: high
---

# LLM Wiki 模式

## 定义

LLM Wiki 模式是一种利用 LLM 作为主要维护者来增量构建和持久化个人知识库的方法论。由 Andrej Karpathy 在其 [[Clippings/llm-wiki.md|llm-wiki gist]] 中正式提出，但理念可追溯到 Vannevar Bush 的 Memex（1945）。

## 核心思想

不同于传统 RAG（在每个查询时从原始文档中检索并重新综合知识），LLM Wiki 模式让 LLM **增量构建和维护一个持久的、结构化的、互相链接的 Markdown 文件集合**——wiki。知识在摄取时被编译一次，然后持续更新，而非在每个查询时重新推导。

## 三层架构

1. **原始来源层** — 不可变的源文档，LLM 只读不修改
2. **Wiki 层** — LLM 生成维护的结构化知识库
3. **Schema 层** — 配置文件（如 CLAUDE.md），定义结构和约定

## 三个操作

- **Ingest（摄取）** — 处理新来源并整合到 wiki
- **Query（查询）** — 基于 wiki 回答问题，好的回答沉淀为新页面
- **Lint（体检）** — 周期性健康检查

## 关键文件

- **index.md** — 内容目录
- **log.md** — 操作时间线

## 与本 vault 的关系

本 vault 的 [[CLAUDE.md]] 是该模式的具体实现，定义了本知识库的目录结构、工作流、命名约定和质量标准。本 wiki 是该模式的产物，Claude 是维护者。

## 资料来源

- [[Wiki/Sources/llm-wiki]] — Karpathy 的原始描述

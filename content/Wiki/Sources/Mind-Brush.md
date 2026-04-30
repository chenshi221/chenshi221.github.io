---
type: source
status: processed
source_file: "[[Clippings/Mind-Brush Integrating Agentic Cognitive Search and Reasoning into Image Generation.md]]"
title: "Mind-Brush: Integrating Agentic Cognitive Search and Reasoning into Image Generation"
site: "arXiv (2602.01756v1)"
author: "Jun He, Junyan Ye, Zilong Huang, et al."
published: "2026-02"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [image-generation, agent, reasoning, knowledge-retrieval, benchmark]
aliases: [Mind-Brush]
---

# Mind-Brush: Integrating Agentic Cognitive Search and Reasoning into Image Generation

2026。ICML 投稿。

## 核心结论

- 现有模型本质上是"静态文本到像素的解码器"，无法理解隐式用户意图。
- 提出 **Mind-Brush**，将图像生成转变为动态知识驱动工作流，模拟人类的 **"think-research-create"** 范式。
- 主动检索多模态证据来接地 out-of-distribution 概念，用推理工具解决隐式视觉约束。
- 提出 **Mind-Bench**（500 样本），覆盖实时新闻、新兴概念、数学和地理推理。

## 关键创新

### Think-Research-Create 范式

1. **Think**：理解用户意图，分解为子问题
2. **Research**：主动检索多模态知识（网络搜索、数据库查询）
3. **Create**：基于检索到的知识生成图像

### 能力突破

- 对 Qwen-Image baseline 在 Mind-Bench 上实现 **zero-to-one 能力飞跃**
- 在 WISE 和 RISE 等 benchmark 上也取得最优

## 与 GoT 的区别

| 维度 | GoT | Mind-Brush |
|------|-----|------------|
| 推理方式 | 内部 CoT 推理 | 外部知识检索 + 推理 |
| 知识来源 | 模型内部先验 | 动态检索外部知识 |
| 适用场景 | 通用生成/编辑 | OOD 概念、实时信息 |

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 代表了从"模型先验"到"外部知识接地"的演进
- Agent + 生成的融合趋势

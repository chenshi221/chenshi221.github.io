---
type: source
status: processed
source_file: '[[Clippings/gpt-oss-120b & gpt-oss-20b Model Card]]'
title: gpt-oss-120b & gpt-oss-20b Model Card
site: arxiv
author: OpenAI
published: '2025'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - gpt-oss
  - OpenAI
  - 开源模型
  - 推理模型
  - Apache 2.0
aliases:
  - gpt-oss
---
# gpt-oss 开源推理模型

## 核心结论

OpenAI 发布 gpt-oss-120b 和 gpt-oss-20b 两个开源权重推理模型（Apache 2.0）。120B 模型在代码和推理任务上达到 GPT-4.1 约 90-95% 性能。支持完整 CoT、结构化输出、工具使用（网络搜索和 Python 执行）和 Agent 工作流，兼容 Responses API。可调整推理努力程度（reasoning effort），为 Agentic 场景优化。

## 关键事实

- 发布方：OpenAI，2025
- 两个规模：gpt-oss-120b 和 gpt-oss-20b
- 许可：Apache 2.0 + gpt-oss usage policy
- 纯文本模型（非多模态）
- 特性：完整 CoT、结构化输出、可调推理努力、工具使用、Agent 工作流
- 安全考量：开源模型风险不同于 API 模型——无法撤销能力、无法后部署安全补丁
- 未达 OpenAI 内部高危评估阈值

## 方法或论证路径

- 模型卡（非系统卡）：因为开源后由第三方构建系统，OpenAI 无法控制完整系统安全
- 推理能力通过可调 reasoning effort 实现——简单任务调低 effort 节省计算
- 工具使用内置：原生支持 web search 和 Python 代码执行
- 安全评估：进行了可扩展能力评估，确定默认模型不达到内部高危阈值

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/大语言模型基础]]、[[Wiki/Concepts/GPT 系列模型]]、[[Wiki/Topics/国产大模型演进]]
- 补充：OpenAI 首次大规模开源模型权重（Apache 2.0），与 DeepSeek/Kimi 的开源策略形成对比但范围不同
- 对比：120B 是 Dense 还是 MoE？文档未详细说明架构——这是信息缺口

## 可能的矛盾或待核实点

- 架构细节未公开：120B 是否是 MoE？与 GPT-4 的技术关系未说明
- "未达高危阈值"的判断标准：OpenAI 的内部评估标准未完全透明

## 后续问题

- gpt-oss 与 DeepSeek-V3/R1 的直接性能对比
- 开源后社区的微调和安全 bypass 实证

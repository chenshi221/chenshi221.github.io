---
type: source
status: processed
source_file: "[[Clippings/VisionCreator A Native Visual-Generation Agentic Model with Understanding, Thinking, Planning and Creation.md]]"
title: "VisionCreator: A Native Visual-Generation Agentic Model"
site: "arXiv (2603.02681v1)"
author: "Jinxiang Lai, Zexin Lu, et al. (Tencent Hunyuan + HKUST)"
published: "2026-03"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [image-generation, agent, MLLM, reinforcement-learning, Tencent]
aliases: [VisionCreator]
---

# VisionCreator: A Native Visual-Generation Agentic Model

Tencent Hunyuan + 港科大，2026。

## 核心结论

- 提出 **VisionCreator**，原生视觉生成 agent 模型，统一 **UTPC**（Understanding, Thinking, Planning, Creation）能力。
- 四大贡献：(1) VisGenData-4k（元认知驱动的创建轨迹数据）；(2) Progressive Specialization Training (PST) + Virtual RL (VRL)；(3) VisGenBench（1.2K 测试样本）；(4) 在多个 benchmark 上 SOTA。

## UTPC 框架

| 阶段 | 能力 | 说明 |
|------|------|------|
| Understanding | 视觉理解 | 分析用户需求、参考图像 |
| Thinking | 思考推理 | 设计创意方案、权衡选项 |
| Planning | 规划布局 | 确定视觉元素和空间关系 |
| Creation | 执行生成 | 端到端生成/编辑图像 |

## 训练方法

### Progressive Specialization Training (PST)
- 分阶段训练：先学会理解 → 再学会思考 → 再学会规划 → 最终学会创建
- 避免一次性训练导致的灾难性遗忘

### Virtual Reinforcement Learning (VRL)
- 在高保真模拟环境中评估生成质量
- 不需要人类反馈即可优化

## 在 Agent 生成模型中的定位

| 模型 | 范式 | 核心能力 |
|------|------|----------|
| GoT | CoT 推理 + 生成 | 语义-空间推理 |
| Mind-Brush | think-research-create | 外部知识检索 |
| VisionCreator | UTPC | 端到端 agent 创建 |

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 代表了图像生成从"模型调用"到"自主 Agent"的演进

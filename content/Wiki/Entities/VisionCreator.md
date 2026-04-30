---
type: entity
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - VisionCreator
tags:
  - image-generation
  - agent
  - MLLM
  - reinforcement-learning
  - Tencent
sources:
  - '[[Wiki/Sources/VisionCreator]]'
---

# VisionCreator

## 基本信息

| 属性 | 值 |
|------|-----|
| 全称 | VisionCreator: A Native Visual-Generation Agentic Model with Understanding, Thinking, Planning and Creation |
| 作者 | Jinxiang Lai, Zexin Lu, et al. |
| 机构 | Tencent Hunyuan + 港科大 |
| 年份 | 2026 (arXiv 2603.02681) |
| 类型 | 原生视觉生成 Agent 模型 |

## 核心创新

提出 **UTPC 框架**，将图像生成统一为四个阶段：理解 → 思考 → 规划 → 创建。这是目前 Agent 生成模型中**最完整的认知流程**。

```
VisionCreator:
  Understanding → Thinking → Planning → Creation
  (理解需求)     (设计方案)   (空间布局)   (执行生成)
```

## UTPC 框架详解

| 阶段 | 核心能力 | 具体做什么 |
|------|----------|-----------|
| **U**nderstanding | 视觉理解 | 分析用户需求、解读参考图像、识别约束条件 |
| **T**hinking | 创意推理 | 设计多种创意方案、权衡风格和可行性 |
| **P**lanning | 空间规划 | 确定视觉元素的位置、大小、层次关系 |
| **C**reation | 执行生成 | 端到端生成/编辑图像 |

## 训练创新

### Progressive Specialization Training (PST)
分阶段训练，逐步解锁能力：
```
阶段1: 只训练 Understanding → 模型先学会"看懂"
阶段2: + Thinking              → 模型学会"想清楚"
阶段3: + Planning              → 模型学会"布局"
阶段4: + Creation              → 模型最终学会"画出来"
```
这避免了 "一次性全训" 导致的灾难性遗忘。

### Virtual Reinforcement Learning (VRL)
- 在高保真模拟环境中自动评估生成质量
- **不需要人类反馈**即可优化生成策略
- 解决了 RLHF 依赖大量人类标注的瓶颈

## 三大 Agent 模型的完整对比

| 维度 | GoT | Mind-Brush | VisionCreator |
|------|-----|------------|---------------|
| **认知流程** | 推理 → 生成 | Think → Research → Create | U → T → P → C |
| **知识获取** | 内部先验 | **外部检索** | 内部先验 |
| **训练方法** | 监督微调 | Agent 工作流 | PST + VRL |
| **强化学习** | 无 | 无 | **VRL（无人类反馈）** |
| **Benchmark** | 无 | Mind-Bench (500) | VisGenBench (1.2K) |
| **优势** | 推理链可解释 | 实时知识接地 | **最完整认知流程** |
| **弱点** | 无法处理 OOD | 检索质量依赖外部 | 流程复杂，训练成本高 |

## 关键判断

VisionCreator 是 Agent 路线中**最激进**的：它试图用一个模型完成从理解到创建的全部工作。但这带来了一个问题：

> **Agent 是必要的复杂度，还是过早的 over-engineering？**

- 对于 90% 的用户需求（"画一只猫"），直接生成就够了
- 对于 9% 的需求（"画一只穿宇航服的猫在月球上"），GoT 级别的推理就够
- 对于 1% 的需求（"设计一个符合唐代建筑规范的寺庙鸟瞰图"），才需要 VisionCreator 的全流程

最优系统可能不是"全 Agent"，而是**路由式**：简单需求走快速通道，复杂需求激活 UTPC。

## 在 Wiki 中的关联

- 来源：[[Wiki/Sources/VisionCreator]]
- 主题：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 比较：[[Wiki/Comparisons/编辑方法能力演进]]
- 相关：[[Wiki/Entities/GoT]], [[Wiki/Entities/Mind-Brush]]

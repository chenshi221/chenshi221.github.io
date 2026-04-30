---
type: comparison
status: active
created: '2026-04-30'
updated: '2026-04-30'
tags:
  - 推理模型
  - DeepSeek
  - Kimi
  - Qwen
  - RL
  - GRPO
  - 训练方法
  - 对比
sources:
  - '[[Wiki/Sources/DeepSeek-R1 强化学习推理]]'
  - '[[Wiki/Sources/Kimi k1.5 强化学习规模化]]'
  - '[[Wiki/Sources/Qwen3 技术报告]]'
confidence: high
---
# 推理模型训练方法比较：DeepSeek-R1 vs Kimi k1.5 vs Qwen3

## 比较概述

2024-2025 年，三家国产大模型团队各自探索了不同的推理模型训练路线。虽然目标一致（让模型具备强大的 CoT 推理能力），但方法论的差异反映了对"推理从何而来"这一根本问题的不同理解。

## 训练流程对比

### DeepSeek-R1：纯 RL 涌现 + 蒸馏

```
阶段1: DeepSeek-R1-Zero
  DeepSeek-V3-Base → GRPO RL（纯 RL，无 SFT）
  → 涌现：aha moment, self-verification, reflection
  （但可读性差、语言混杂）

阶段2: DeepSeek-R1
  1. 冷启动 SFT（数千高质量 CoT 数据）
  2. 推理导向 GRPO RL（语言一致性奖励）
  3. 拒绝采样 → SFT（推理 + 通用）
  4. 全场景 GRPO RL（所有类型的 reward）

阶段3: 蒸馏
  R1 数据 → 蒸馏 6 个模型（1.5B-70B）
```

**核心理念**：RL 是推理能力的来源，SFT 是格式保障。R1-Zero 证明了纯 RL 可以涌现推理，R1 通过冷启动 SFT 解决了可读性问题。

### Kimi k1.5：长上下文 RL + Long2Short

```
阶段1: 长上下文预训练
  128K 上下文窗口（让模型能处理极长的 CoT）

阶段2: 长 CoT RL 训练
  online mirror descent（不是 PPO）
  规则化奖励（不是 reward model）
  128K 长上下文中的多轮推理

阶段3: Long2Short 蒸馏
  四种方法：
  - Model Merging（长短 CoT 权重融合）
  - 最短拒绝采样（选最短正确回答训练）
  - DPO 蒸馏（长 CoT preferred, 短 CoT rejected）
  - 长度惩罚 RL

阶段4（未明确）：可能的 iterative RL
```

**核心理念**：**长上下文是推理的物理基础**。推理需要"空间"——128K 上下文给了模型自由展开思考的空间。Long2Short 是部署优化，不是能力来源。

### Qwen3：双模式 + 思考预算

```
阶段1: Dense + MoE 双架构预训练
  36T tokens（Qwen2.5 基础上升级）
  双架构策略：Dense 版（高效推理）、MoE 版（高吞吐）

阶段2: 四阶段训练管线
  1. SFT（通用能力）
  2. 长 CoT SFT（推理能力初始化）
  3. RL（GRPO-like，规则化奖励 + reward model 混合）
  4. DPO（最终对齐）

阶段3: 统一思考/非思考模式
  通过 /think 和 /no_think token 切换
  thinking budget：控制推理深度（0-4096 token）
  Strong-to-Weak 蒸馏：大模型教小模型切换模式
```

**核心理念**：**推理不是 always-on 的**。大多数用户查询不需要深度推理。核心问题是"对简单问题快速回答，对难问题充分思考"——thinking budget 是产品化的关键创新。

## 方法论差异的根源

### 对"推理从何而来"的不同回答

| 团队 | 回答 | 方法论推论 |
|------|------|-----------|
| DeepSeek | 推理来自 RL 的探索性训练 | 纯 RL（R1-Zero）+ 蒸馏是最优路径 |
| Kimi | 推理来自长上下文的思考空间 | 128K RL + Long2Short 是最优路径 |
| Qwen3 | 推理来自训练，但使用应受控 | 双模式 + thinking budget 是最优路径 |

这三种回答并不矛盾——它们是对同一问题的不同侧面的强调。

### 对开源策略的不同理解

| 团队 | 开源策略 | 影响 |
|------|---------|------|
| DeepSeek | 全开源（权重 + 技术报告） | 社区最活跃，衍生最丰富 |
| Kimi | 权重开源（K2, VL）+ 技术报告 | 关注 Agent 生态 |
| Qwen3 | Apache 2.0 全开源 | 商用最友好 |

## 效果对比

| 维度 | DeepSeek-R1 | Kimi k1.5 | Qwen3 |
|------|------------|-----------|-------|
| **推理天花板** | 最高（671B RL） | 极高（128K 长推理） | 高（双模式） |
| **实用性（简单任务）** | 中（慢） | 中（慢） | **最高**（/no_think 快速） |
| **长上下文推理** | 高 | **最高**（128K 原生） | 高 |
| **Agent 能力** | 中（V3.2 在追赶） | **最高**（K2.5） | 中（非重点） |
| **多语言** | 中（中文强） | 中（中文强） | **最高**（36T 多语言） |
| **部署灵活性** | 中（671B MoE） | 中（1.04T MoE） | **最高**（Dense+MoE 双选） |
| **开源生态** | **最强** | 强 | 强 |

## 关键洞察

### 三家的独特贡献

- **DeepSeek**：证明了 **"纯 RL 可以涌现推理"**，这是对 AI 认知能力的根本性贡献。R1-Zero 的 aha moment 的价值不在工程，在科学。
- **Kimi**：证明了 **"长上下文是推理的基础设施"**。128K RL 训练 + Long2Short 的方法论，暗示了推理和上下文的深层关系。
- **Qwen3**：证明了 **"推理可以按需开关"**。thinking budget 是推理模型产品化的关键创新——让推理模型从"实验室奇迹"变成"日常可用"。

### 技术路线会趋同吗？

三家目前的方法论差异很大，但可能最终会趋同：

1. **RL 成为共识**（三家用或 GRPO 或 mirror descent 的 RL）
2. **Long2Short 成为共识**（训练时用长推理获取能力，部署时压缩）
3. **多模式成为共识**（简单问题不需要推理，类似 Qwen3 的 approach 会被采纳）
4. **Agent 成为下一个战场**（当推理能力 commoditized 后）

## 与已有 Wiki 的连接

- 关联概念：[[推理模型与强化学习]]、[[GRPO 分组相对策略优化]]、[[知识蒸馏]]、[[DPO 直接偏好优化]]
- 关联实体：[[DeepSeek 系列模型]]、[[Kimi 系列模型]]、[[Qwen3]]
- 关联比较：[[Wiki/Comparisons/推理增强方法比较]]、[[Wiki/Comparisons/国产大模型技术路线比较]]
- 关联问题：[[Wiki/Questions/知识蒸馏 vs RL 哪种方式更能有效获得推理能力]]

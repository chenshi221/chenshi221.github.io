---
type: concept
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - Group Relative Policy Optimization
  - 分组策略优化
tags:
  - GRPO
  - RL
  - 强化学习
  - 推理模型
  - DeepSeek
  - PPO
sources:
  - '[[Wiki/Sources/DeepSeek-R1 强化学习推理]]'
  - '[[Wiki/Sources/DeepSeek-V3.2 开源大模型前沿]]'
confidence: high
---
# GRPO 分组相对策略优化

## 定义

**GRPO（Group Relative Policy Optimization，分组相对策略优化）** 是 DeepSeek 在 R1 和 V3.2 中使用的强化学习算法，是 PPO（Proximal Policy Optimization）的一种轻量化变体。其核心创新是**用组内相对比较替代训练独立的 critic 模型**，大幅降低了 RL 训练的计算成本和工程复杂度。

## 核心机制

### 为什么需要替代 PPO

传统 RLHF 流程（如 InstructGPT）：

1. SFT → 训练 Reward Model（RM）→ 用 PPO 优化策略
2. PPO 需要同时维护 4 个模型：Policy、Reference Policy、Critic（Value Model）、Reward Model
3. **Critic 模型通常需要和 Policy 同等规模**，存储和计算开销翻倍

### GRPO 的关键创新：去 Critic

GRPO 不需要 critic 模型，替代方案是：

1. 对**同一个 prompt**，采样 **G 个不同的输出**（组成一个 group）
2. 对每个输出用 reward model 打分，得到组内 reward：`{r₁, r₂, ..., r_G}`
3. **组内标准化**：将每个 reward 减去组均值除以组标准差 → 相对优势 `Â`
4. 用 PPO 式的 clipped objective 更新策略，但优势来自组内比较

```
Â_i = (r_i - mean(r)) / std(r)    // 替代 critic 的价值估计
```

### 与 PPO 的对比

| 维度 | PPO | GRPO |
|------|-----|------|
| 模型数量 | 4（Policy+Ref+Critic+RM） | 3（Policy+Ref+RM） |
| Critic 训练 | 需要 | 不需要 |
| GPU 内存 | 高（critic 占额外 50%+） | 中 |
| 优势估计 | 学习 V(s) → TD error | 组内相对比较 |
| 方差 | 低（V 函数平滑） | 中（组内采样的方差） |
| 适用场景 | 通用 RL | 生成式任务（可多次采样同一 prompt） |

## 为什么 GRPO 有效

### 组内比较的直觉

- 绝对奖励信号不稳定（reward model 本身有噪声和偏差）
- 但**同一个 prompt 的不同回答之间的相对优劣**更容易判断
- GRPO 实质上利用了 prompt 内部的对比信号，绕过了需要精确值函数的难题

### 前提条件

GRPO 只适用于**可以低成本地对同一 prompt 生成多样输出的任务**——这正是 LLM 推理的天然场景。对每个数学题，模型可以生成多个不同的推理路径，从中选出最优。

### DeepSeek 的实践经验

- R1-Zero 用纯 GRPO 训练（无 SFT），涌现出了 self-verification、reflection、aha moment 等推理行为
- V3.2 用 GRPO 实现 agent 能力的 RL 训练
- 关键在于 reward 设计：规则化 reward（数学题答案对错）比学习式 reward model 更有效

## 扩展：从 GRPO 到 RLVR

**RLVR（RL with Verifiable Rewards）** 是 GRPO 的泛化概念：

- 使用**可验证的规则化奖励**（数学答案、代码执行结果）替代 reward model
- 避免了 reward hacking 和 reward model 的训练成本
- DeepSeek-R1、Kimi k1.5 均大量使用 rule-based RL

## 与其他方法的关联

### vs DPO

- **DPO**：直接优化偏好对，不需要 RL 训练循环，简单但不适合多步推理链的优化
- **GRPO**：保留 RL 训练循环（模型可以通过探索发现新策略），但去掉了 critic 的工程负担
- 选择：单轮对齐用 DPO，多步推理能力的获取用 GRPO

### vs Kimi k1.5 的 online mirror descent

- Kimi k1.5 采用了不同的 RL 优化器，但核心思想类似：用规则化奖励驱动推理训练
- 差异主要在优化算法的选择，不是在 reward 设计层面

## 与已有 Wiki 的连接

- 关联概念：[[推理模型与强化学习]]、[[RLHF]]、[[Scaling Laws]]
- 关联实体：[[DeepSeek 系列模型]]（GRPO 起源）、[[Kimi 系列模型]]（online mirror descent）
- 所在主题：[[Wiki/Topics/大语言模型基础]]、[[Wiki/Topics/国产大模型演进]]

## 深度分析

### GRPO 的真正意义不是"省钱"，而是"降低 RL 门槛"

从表面看，GRPO 只是省了一个 critic 模型。但深层含义是：它让中小团队也有了做 RL 推理训练的可能。PPO 的 4 模型架构需要极强的工程能力（模型并行、pipeline 编排），GRPO 的简化让 RL 训练不再是少数头部玩家的特权。

这也是为什么 DeepSeek-R1 的影响力远超其绝对性能——它展示的是一条更低成本的 RL 路径。

### 组内相对比较的隐藏弱点

GRPO 组内标准化的前提是**组内有足够的方差**（不同回答质量差异大）。当模型已经很擅长某个任务（组内回答质量趋于一致），GRPO 的信号就会衰减到接近噪声——优势估计的方差会飙升。

这是目前尚未被充分讨论的 GRPO 天花板：**它在中等难度任务上效果最好，在极难（所有回答都差）和极简单（所有回答都好）的任务上信号很弱**。

### 从算法到范式

GRPO 的影响已经超越了算法层面。它和规则化奖励（RLVR）一起，标志着一个新范式的形成：**不需要人类偏好标注、不需要 reward model 训练、不需要 critic 模型**——只需要可验证的任务和足够的采样预算。这可能是 RL for LLM 走向大规模实用化的关键转折。

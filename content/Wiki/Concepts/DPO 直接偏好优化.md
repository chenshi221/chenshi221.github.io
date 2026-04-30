---
type: concept
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - Direct Preference Optimization
  - 直接偏好优化
tags:
  - DPO
  - RLHF
  - 对齐
  - 偏好学习
sources:
  - '[[Wiki/Sources/InstructGPT]]'
  - '[[Wiki/Sources/Llama 3]]'
confidence: high
---
# DPO 直接偏好优化

## 定义

**DPO（Direct Preference Optimization，直接偏好优化）** 是 Rafailov et al. 于 2023 年（NeurIPS 2023）提出的 LLM 对齐方法。它通过**直接在偏好对数据上优化策略模型**，绕过了 RLHF 中训练 reward model + PPO 的两阶段流程，将对齐简化为一个监督学习问题。

## 核心洞察

### RLHF 的"绕远路"

RLHF 的标准流程：

1. 收集偏好数据（chosen vs rejected 回答对）
2. 用偏好数据训练一个 **reward model**（建模人类偏好）
3. 用 reward model 作为奖励信号，通过 **PPO** 强化学习优化策略

DPO 的核心发现：**reward model 的最优解可以直接用策略模型表示**，无需显式训练 reward model。

### 数学推导（简化）

在 Bradley-Terry 偏好模型下：

- 两个回答 y₁, y₂ 的偏好概率：`P(y₁ ≻ y₂) = σ(r(y₁) - r(y₂))`
- RLHF 中 PPO 的最优策略满足：`r(y) = β * log(π(y|x) / π_ref(y|x)) + C`
- 代入偏好模型，直接得到 DPO 的损失函数：

```
L_DPO = -log σ(β * [log(π_θ(y_w|x)/π_ref(y_w|x)) - log(π_θ(y_l|x)/π_ref(y_l|x))])
```

其中 y_w 是 preferred 回答，y_l 是 dispreferred 回答。

## DPO vs RLHF (PPO)

| 维度 | RLHF (PPO) | DPO |
|------|------------|-----|
| 阶段数 | 3（SFT→RM→PPO） | 2（SFT→DPO） |
| Reward Model | 需要单独训练 | 不需要 |
| RL 训练 | 需要 | 不需要（纯监督学习） |
| 训练稳定性 | PPO 调参敏感 | 稳定（标准分类式 loss） |
| 样本效率 | 相对低（RM 信号不够精确） | 较高 |
| 在线探索 | 支持（可以产生新数据） | 通常离线（固定偏好数据集） |
| 迭代能力 | 可以多轮 self-play | 通常单轮 |

## DPO 的优势

1. **工程简单**：不需要维护 reward model 和 RL 训练循环，是一个标准分类任务
2. **训练稳定**：没有 PPO 的 KL 散度约束调参、reward hacking 等痛点
3. **计算高效**：理论上前向/反向各一次等同于 reward model 训练 + PPO 的联合优化
4. **偏好信号直接**：直接优化"chosen 比 rejected 好多少"，信号比 RM 的标量 reward 更精确

## DPO 的局限

### 离线偏好分布问题

- DPO 在固定的偏好数据集上训练
- 策略更新后，旧偏好数据可能不再反映当前策略的分布
- **分布偏移**：偏好数据来自参考策略 π_ref，但训练中策略 π_θ 逐渐偏离 π_ref
- 这个问题催生了许多 DPO 变体（iterative DPO、online DPO）

### 多步推理链的局限

- DPO 优化的是"最终回答的偏好"，难以优化中间推理步骤
- 对于需要长思维链的推理任务，DPO 不如 RL 方法（GRPO、RLVR）灵活

## DPO 变体

### Iterative DPO / Online DPO

- 用当前策略采样新回答 → 收集偏好 → 再次 DPO 训练 → 循环
- 解决了分布偏移问题，但回到了类似 RL 的在线训练模式
- Llama 3 使用了 iterative DPO

### KTO（Kahneman-Tversky Optimization）

- 只使用二元反馈（好/不好），不需要成对偏好
- 更符合实际场景（用户给的是 thumbs up/down，不是 A vs B）

### SimPO

- 用序列平均 log-probability 作为隐式 reward，连 reference model 都不需要
- 进一步简化 DPO 的计算图

## 在工业界的应用

- **Llama 3**：SFT → 多种偏好数据 → iterative DPO
- **Kimi K2**：使用了 RLVR + DPO 的组合对齐
- **Qwen3**：DPO 作为 aligned model 的最终阶段
- **DeepSeek V3**：GRPO 用于推理 + DPO 用于通用对齐

## 与已有 Wiki 的连接

- 关联概念：[[RLHF]]、[[GRPO 分组相对策略优化]]、[[推理模型与强化学习]]
- 关联实体：[[DeepSeek 系列模型]]、[[Kimi 系列模型]]、[[Qwen3]]
- 所在主题：[[Wiki/Topics/大语言模型基础]]

## 深度分析

### DPO 是"足够好"对齐的成本最优解

DPO 的真正价值不在理论上（它与 PPO 在最优解上是等价的），而在工程上。对于绝大多数应用场景，DPO 的"足够好"远好于调不出来的 PPO。这是一种 **Worse is Better 哲学**在 LLM 对齐中的体现——简单的方案胜出不是因为它更优，而是因为它能稳定工作。

但需要注意的是，DPO 的这种"足够好"依赖于一个前提：偏好数据覆盖了目标分布。当任务需要模型探索新行为（如推理能力的涌现），DPO 无能为力——这时候 RL（GRPO/RLVR）才是正确工具。

### DPO 和 GRPO 的分工

可以这样理解：

- **DPO** 是 **alignment（对齐）** 工具：让模型知道"什么是好的回答"
- **GRPO** 是 **acquisition（能力获取）** 工具：让模型学会"如何推理出好的回答"

两者的分工正在变得越来越清晰：用 RL 获取推理能力，用 DPO 进行最终的偏好调优。DeepSeek 的 R1（GRPO 推理 + SFT 对齐）和 Qwen3（RL 思考 + DPO 对齐）都体现了这种分层思想。

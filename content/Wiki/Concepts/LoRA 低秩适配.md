---
type: concept
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - LoRA
  - 低秩适配
  - Low-Rank Adaptation
tags:
  - LoRA
  - 微调
  - 高效训练
  - LLM
  - Stable Diffusion
sources:
  - '[[Wiki/Sources/LoRA 低秩适配]]'
confidence: high
---
# LoRA 低秩适配

## 定义

LoRA（Low-Rank Adaptation）是一种参数高效的大模型微调方法：冻结预训练权重矩阵 W0，引入一对低秩矩阵 A 和 B（BA 的乘积远小于 W0），将微调的增量 ΔW 表示为 ΔW = BA，训练时只更新 A 和 B。推理时 BA 可合并回 W0，无额外延迟。

## 核心理念

预训练语言模型的「适配」（adaptation）具有内在低秩性（intrinsic rank-deficiency）。这意味着微调时权重的变化可以被低维子空间有效表示，不需要更新全部参数。

## 技术细节

- 只对 Transformer 的 Q（query）和 V（value）投影矩阵注入 LoRA，实验证明这是最有效的注入位置
- 秩 r 可以非常小——1 到 8 通常足够
- 训练时：只计算 A 和 B 的梯度，FP 不经过 W0
- 推理时：W = W0 + BA 合并为一个矩阵，完全等价于原架构
- 多任务支持：每个任务只需存储独立的 LoRA 权重（几个 MB），一个基座模型切换不同 LoRA 即可

## 在更大生态中的位置

### NLP 领域

- 原始论文在 RoBERTa、DeBERTa、GPT-2、GPT-3 上验证
- 已成为开源 LLM 社区事实上的微调标准（配合 QLoRA 实现 4-bit 量化微调）

### 扩散模型领域

- Stable Diffusion 社区中 LoRA 是最主流的微调方式之一（训练特定角色、风格、概念）
- 与 Dreambooth 相比：LoRA 文件小（~10-200MB vs ~2-6GB），训练快
- FLUX 模型同样支持 LoRA 微调

### 与其他高效微调方法的关系

| 方法 | 原理 | 推理延迟 | 参数效率 |
|------|------|---------|---------|
| LoRA | 低秩分解 ΔW | 无（可合并） | 极高 |
| Adapter | 插入小型 FFN 模块 | 有额外延迟 | 高 |
| Prefix Tuning | 可学习前缀 token | 占用序列长度 | 中 |
| Full Fine-tuning | 更新全部参数 | 无 | 无 |

## 关键限制

- LoRA 本质上是"低秩偏置"——对某些需要大幅改变模型行为的任务（如新语言适配），r 可能需要调大甚至不够用
- 多个 LoRA 合并时可能出现权重冲突，社区开发了多种合并策略（线性插值、加权合并、SVCCA 等）
- 在扩散模型中，LoRA 的"可合并"特性受 VAE 隐空间限制——不同的 LoRA 之间可能存在风格冲突

## 来源

- [[Wiki/Sources/LoRA 低秩适配]] — 原始论文
- 关联：[[Wiki/Topics/大语言模型基础]]、[[Wiki/Topics/扩散模型图像编辑与生成]]

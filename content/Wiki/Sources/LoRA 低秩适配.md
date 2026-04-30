---
type: source
status: processed
source_file: '[[Clippings/LoRA Low-Rank Adaptation of Large Language Models]]'
title: 'LoRA: Low-Rank Adaptation of Large Language Models'
site: arxiv
author: Edward Hu et al. (Microsoft)
published: '2021'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - LoRA
  - 微调
  - 高效训练
  - LLM
aliases:
  - LoRA
---
# LoRA 低秩适配

## 核心结论

LoRA（Low-Rank Adaptation）提出了一种参数高效的大模型微调方法：冻结预训练权重，在 Transformer 每层注入可训练的秩分解矩阵（低秩矩阵 A 和 B）。相比全量微调，LoRA 将可训练参数减少 10000 倍，GPU 显存需求降低 3 倍，且无额外推理延迟（因为可合并到原权重中）。在 RoBERTa、DeBERTa、GPT-2、GPT-3 上性能持平或优于全量微调。

## 关键事实

- 作者：Edward Hu、Yelong Shen 等（Microsoft），2021
- 核心思想：预训练权重 W0 冻结，微调增量 ΔW 分解为低秩矩阵 BA（r << d），W = W0 + BA
- 对 GPT-3 175B：可训练参数从 175B 降至 4.7M（~37000 倍减少）
- 秩 r 可以非常小（1-4 即可），验证了语言模型适配的低秩性（intrinsic rank-deficiency）
- 支持多任务部署：每个任务只需存储独立的 LoRA 权重，不会导致存储爆炸

## 方法或论证路径

- 将全量微调增量 ΔW 参数化为一对低秩矩阵的乘积，秩 r 远小于原始维度
- 仅对 Q（query）和 V（value）投影矩阵注入 LoRA；实验证明这是最有效的选择
- 训练时只需计算低秩矩阵梯度，推理时 BA 可合并回原权重无额外延迟
- 在多个 NLU 和 NLG 任务上系统的消融实验验证了秩选择和注入位置的影响

## 与现有 Wiki 的关系

- 关联：[[Wiki/Concepts/LoRA 低秩适配]]、[[Wiki/Topics/大语言模型基础]]
- 补充：LoRA 已是 Stable Diffusion 生态中标准微调方法（广泛应用于扩散模型微调），本来源提供基础理论解释
- 与其他高效微调方法（Adapter、Prefix Tuning）的关系需后续补充

## 可能的矛盾或待核实点

- 论文仅验证了 NLP 任务，在扩散模型图像生成中的 LoRA 行为是否存在差异

## 后续问题

- LoRA 在扩散模型（SD/FLUX）中与传统 NLP 场景有何不同？
- 多个 LoRA 合并时的权重冲突问题如何解决？

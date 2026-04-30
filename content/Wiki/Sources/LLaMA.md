---
type: source
status: processed
site: arXiv
source_file: '[[Clippings/LLaMA Open and Efficient Foundation Language Models.md]]'
title: 'LLaMA: Open and Efficient Foundation Language Models'
author: Hugo Touvron et al. (Meta AI)
published: '2023'
processed: '2026-04-30'
tags:
  - LLaMA
  - open-source
  - scaling
  - efficiency
  - Meta
---
# LLaMA: Open and Efficient Foundation Language Models

## 核心结论

- 提出了 **LLaMA**（7B/13B/33B/65B），一系列仅使用公开数据集训练的基础语言模型，证明无需私有数据即可训练 SOTA 模型。
- LLaMA-13B 在大多数 benchmark 上超越 GPT-3（175B），LLaMA-65B 与 Chinchilla-70B 和 PaLM-540B 竞争。
- 核心洞察：给定计算预算，最佳性能不是来自最大模型，而是来自在更多数据上训练的较小模型——遵循 [[Wiki/Sources/Chinchilla 缩放定律|Chinchilla 缩放定律]]。

## 关键方法或创新点

- **数据驱动效率**：使用 1.4T token 的公开数据（CommonCrawl、C4、GitHub、Wikipedia、Books、ArXiv、Stack Exchange 等）。
- 架构改进：
  - Pre-normalization（每层 transformer 子层前归一化，而非后归一化），使用 RMSNorm
  - SwiGLU 激活函数（替换 ReLU）
  - [[Wiki/Concepts/RoPE 旋转位置编码|Rotary Positional Embedding (RoPE)]]
- 训练细节：AdamW 优化器，手动调整学习率调度，使用高效的 xformers 实现；所有训练在 2048 个 A100 GPU 上完成。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/大语言模型基础]]、[[Wiki/Concepts/RoPE 旋转位置编码]]、[[Wiki/Concepts/Scaling Laws]]
- LLaMA 是开源大模型生态的重要里程碑，衍生出 Alpaca、Vicuna、Mistral 等大量后续。
- 直接影响了 [[Wiki/Sources/Llama 3|Llama 3 系列]] 的设计思想（数据优先、计算最优）。

## 局限或注意事项

- 仅研究性发布（non-commercial license），后 Llama 2 才改为开源友好许可。
- 预训练数据中仍然存在偏见和有害内容风险（论文承认但未充分缓解）。
- 仅评估了 zero-shot 和 few-shot 能力，未进行指令微调。

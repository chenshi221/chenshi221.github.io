---
type: source
status: processed
source_file: "[[Clippings/FLUX.1 Kontext Flow Matching for In-Context Image Generation and Editing in Latent Space]]"
title: "FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space"
site: "arXiv (2025)"
author: "Black Forest Labs (Stephen Batifol, Andreas Blattmann, Robin Rombach 等)"
published: "2025"
processed: "2026-04-30"
updated: "2026-04-30"
tags: ["flow-matching", "image-editing", "DiT", "FLUX", "in-context", "rectified-flow"]
aliases: ["FLUX.1 Kontext", "FLUX Kontext"]
confidence: high
---

# FLUX.1 Kontext

## 核心结论

FLUX.1 Kontext 是 Black Forest Labs 的 flow matching 模型，通过简单的序列拼接方法统一了图像生成和编辑。模型在 latent space 中训练 rectified flow transformer，支持文本到图像（T2I）和上下文图像编辑。相比现有编辑模型，在多轮编辑中显著改善了角色一致性和稳定性，1024x1024 推理仅需 3-5 秒。

## 关键方法

1. **架构**：基于 FLUX.1 的 rectified flow transformer（DiT 架构），使用双流/单流混合 block（double stream + 38 single stream blocks）、融合前馈块、3D RoPE 位置编码，在 16 通道 latent space 中训练。

2. **Token 序列拼接**：将上下文图像的 latent token 直接拼接到目标图像 token 序列中，作为视觉流的输入。3D RoPE 为上下文 token 分配虚拟时间偏移量 $(i, h, w)$，区分上下文与目标。

3. **Rectified Flow 目标**：$\mathcal{L}_\theta = \mathbb{E}_{t,x,y,c}[\|v_\theta(z_t, t, y, c) - (\varepsilon - x)\|_2^2]$，其中 $z_t = (1-t)x + t\varepsilon$，使用 logit-normal shift schedule。

4. **Adversarial Diffusion Distillation (LADD)**：通过对抗蒸馏减少采样步数，同时提升样本质量。dev 版通过 guidance distillation 压缩到 12B DiT。

5. **KontextBench**：1026 个图像-提示对，涵盖局部编辑、全局编辑、字符参考、风格参考、文本编辑 5 类任务。

## 与现有 Wiki 的关系

- 关联方法：[[Wiki/Concepts/Flow Matching]]、[[Wiki/Concepts/扩散模型原理]]
- 编辑方法分支：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 与 Seedream 系列同为基于 flow matching 的生成基础模型：[[Wiki/Entities/Seedream 系列模型]]
- 串联脉络：[[Wiki/Topics/扩散模型与 Flow Matching 基础]]

## 可能的矛盾或待核实点

- FLUX.1 Kontext 使用 rectified flow（$z_t = (1-t)x + t\varepsilon$），而原始 FM 的 OT 路径也使用线性插值但方差收缩不同。两种线性路径的关系值得进一步探讨。

## 后续问题

- 多图像输入和视频编辑是未来扩展方向。
- 多轮编辑的视觉退化仍是限制。

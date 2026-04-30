---
type: source
status: processed
source_file: "[[Clippings/Show-o2 Improved Native Unified Multimodal Models.md]]"
title: "Show-o2: Improved Native Unified Multimodal Models"
site: "arXiv (NUS Show Lab, ByteDance)"
author: "Jinheng Xie, Zhenheng Yang, Mike Zheng Shou"
published: "2025-06"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [unified-multimodal, show-o, flow-matching, 3d-vae]
aliases: [Show-o2]
---

# Show-o2: Improved Native Unified Multimodal Models

## 核心结论

- Show-o2 在 **3D Causal VAE** 空间上构建统一视觉表示，通过**双路径空间（时间）融合**机制同时捕获高维语义和低维结构信息，可扩展至图像和视频。
- 在单一模型中原生应用 **autoregressive modeling（文本）和 flow matching（图像/视频）** 两个目标，使用不同的 head（language head + flow head）。
- 两阶段训练策略：Stage 1 仅训练投影器、融合模块和 flow head；Stage 2 全模型微调，有效保留语言知识。

## 关键事实

- **统一视觉表示**：3D Causal VAE latent → 双路径提取 = 语义层（蒸馏自 SigLIP）+ 投影器（保留低层信息）→ 空间融合。
- **Omni-Attention**：序列级 causal attention + 统一视觉表示内部的 full attention。
- **可扩展性**：小模型（1.5B）预训练的 flow head 可恢复到大模型（7B），通过轻量 MLP 对齐隐藏维度。
- **支持模态**：文本、图像、视频的交错输入和生成。
- **Benchmark**：在多模态理解和视觉生成基准上超越现有方法。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 关联：[[Wiki/Sources/BAGEL]]（BAGEL 也是 AR + Diffusion 但用 MoT；Show-o2 用统一模型 + 双 head）
- 关联：[[Wiki/Sources/Tuna-2]]（两者都探索统一视觉表示，但 Show-o2 用 3D VAE + 语义蒸馏，Tuna-2 完全去编码器）

## 可能的矛盾或待核实点

- 3D Causal VAE 的损失压缩对精细编辑任务的影响。
- SigLIP 知识蒸馏到 noisy latent 上的效果是否有理论保证。

## 后续问题

- 3D Causal VAE 在视频统一建模中的优势是否可以推广到其他统一模型？
- "原生统一"（native UMM）与 "组装统一"（assembled）的边界对模型能力的影响。

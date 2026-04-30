---
type: source
status: processed
source_file: "[[Clippings/L u m i n a - D i M O O An Omni Diffusion Large Language Model for Multi-Modal Generation and Understanding.md]]"
title: "Lumina-DiMOO: An Omni Diffusion Large Language Model for Multi-Modal Generation and Understanding"
site: "arXiv (Shanghai AI Laboratory)"
author: "Yi Xin, Qi Qin, Siqi Luo, Yihao Liu et al."
published: "2025-10"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [unified-multimodal, discrete-diffusion, llima, lumina-dmoo]
aliases: [Lumina-DiMOO]
---

# Lumina-DiMOO

## 核心结论

- Lumina-DiMOO 是一个纯**离散扩散（discrete diffusion）** 统一多模态模型，区别于 AR 或 AR+Diffusion 混合范式。
- 在文本生成速度上比纯 AR 模型（如 Lumina-mGPT 2.0）快 **32 倍**，加上 ML-Cache 可再快 2 倍。
- 基于 LLaDA（离散扩散 LLM）初始化，无结构修改即可扩展至多模态。

## 关键事实

- **架构**：基于 LLaDA-Base 离散扩散 LLM，使用 aMUSEd-VQ tokenizer（16× 下采样），总词表含 126,345 文本 token + 8,192 视觉 token + 特殊 token。
- **任意分辨率**：通过插入 `<end-of-line>` token 保留二维图像结构，支持任意宽高比。
- **零样本图像修复**：离散扩散的固有能力支持交互式修图（Interactive Retouching）。
- **ML-Cache**：无训练加速方法，利用高 logit token 的表示相似性。
- **UniGenBench 榜首**：在腾讯混元团队维护的 UniGenBench 开源模型排行榜中排名第一。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 对比：[[Wiki/Sources/BAGEL]]（BAGEL 是 AR + Diffusion 混合；DiMOO 是纯离散扩散）
- 对比：[[Wiki/Sources/Tuna-2]]（Tuna-2 是连续像素空间扩散；DiMOO 是离散 token 空间扩散）

## 可能的矛盾或待核实点

- 离散扩散在图像质量上是否最终能赶上连续扩散（如 FLUX、SD3）？
- aMUSEd-VQ 的 16× 下采样是否丢失过多细节？

## 后续问题

- 离散扩散的并行解码优势是否能在长序列（视频）生成中充分体现？
- ML-Cache 在其他离散扩散模型上的通用性如何？

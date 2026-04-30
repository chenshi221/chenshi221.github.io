---
type: source
status: processed
source_file: "[[Clippings/Denoising Diffusion Probabilistic Models]]"
title: "Denoising Diffusion Probabilistic Models"
site: "arXiv (NeurIPS 2020)"
author: "Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley)"
published: "2020"
processed: "2026-04-30"
updated: "2026-04-30"
tags: ["diffusion", "generative-model", "DDPM"]
aliases: ["DDPM"]
confidence: high
---

# Denoising Diffusion Probabilistic Models (DDPM)

## 核心结论

扩散概率模型是受非平衡热力学启发的隐变量模型，通过训练参数化马尔可夫链来逆转逐步添加噪声的扩散过程。DDPM 首次证明扩散模型可以生成高质量图像：CIFAR-10 上 Inception score 9.46、FID 3.17，256x256 LSUN 上与 ProgressiveGAN 相当的样本质量。

## 关键方法

1. **前向过程（扩散过程）**：固定的马尔可夫链，按方差调度 $\beta_1,\dots,\beta_T$ 逐步向数据添加高斯噪声，直到信号被完全破坏。关键性质：任意时间步 $t$ 的 $\mathbf{x}_t$ 可从 $\mathbf{x}_0$ 以闭式形式采样：$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$。

2. **反向过程（去噪过程）**：学习参数化的高斯转移 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$，从纯噪声逐步恢复到数据分布。

3. **简化训练目标**：$L_{\text{simple}} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$，等同于对多噪声尺度的去噪分数匹配（denoising score matching）。

4. **与 Langevin 动力学的等价性**：DDPM 的采样过程可解释为在多个噪声水平上进行退火 Langevin 动力学。

## 与现有 Wiki 的关系

- 核心概念：[[Wiki/Concepts/扩散模型原理]]
- 为后续扩散模型、Flow Matching、图像编辑奠定的理论基础：[[Wiki/Topics/扩散模型图像编辑与生成]]、[[Wiki/Topics/扩散模型与 Flow Matching 基础]]
- Flow Matching 的扩散路径是 DDPM 路径的一般化：[[Wiki/Concepts/Flow Matching]]

## 后续问题

- 采样速度慢（需要 1000 步），后来被 DDIM、flow matching、蒸馏等方法加速。
- DDPM 在像素空间操作，Latent Diffusion Models (LDM) 将其推广至潜在空间。

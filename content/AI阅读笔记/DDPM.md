---
title: Denoising Diffusion Probabilistic Models
authors:
  - Jonathan Ho
  - Ajay Jain
  - Pieter Abbeel
institutions: UC Berkeley
aliases:
  - DDPM
  - 扩散模型开山之作
date: '2026-04-30'
publish_date: 2020-06
venue: 'NeurIPS 2020'
tags:
  - 论文
  - 扩散模型
  - 图像生成
  - 生成模型
  - score matching
url: 'https://arxiv.org/abs/2006.11239'
code: 'https://github.com/hojonathanho/diffusion'
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：DDPM 提出了去噪扩散概率模型，证明了扩散模型能够生成高质量图像样本——在 CIFAR10 上达到 FID 3.17、IS 9.46，并建立了扩散模型与去噪得分匹配、Langevin 动力学之间的理论联系，奠定了现代扩散生成模型的理论和实践基础。

![](https://ar5iv.labs.arxiv.org/html/2006.11239/assets/images/celebahq256_header_image_4x4.png)

*Figure 1: DDPM 在 CelebA-HQ 256x256（左）和 unconditional CIFAR10（右）上的生成样本。*

---

## Intro

### Motivation

扩散概率模型（diffusion probabilistic models）是一类受非平衡热力学启发的潜变量模型。虽然扩散模型概念上简洁优雅，但在此工作之前，没有证据表明它们能生成高质量的样本。在此之前，GANs、自回归模型、流模型和 VAE 已经在图像和音频合成上取得显著成果，但扩散模型在图像生成上的潜力未被充分发掘。

### 核心主张

1. 扩散模型**确实能够**生成高质量样本，有时甚至优于已发表的其它类型生成模型
2. 特定的参数化方式（预测噪声而非预测均值）揭示了扩散模型与**去噪得分匹配（denoising score matching）**的等价性，以及采样过程与**退火 Langevin 动力学**的联系
3. 扩散模型的采样过程可以被理解为一种**渐进式有损解码**，泛化了自回归解码的概念

### 贡献

1. 首次证明扩散模型能生成高质量图像样本（CIFAR10 FID 3.17，超过当时大多数无条件模型）
2. 提出噪声预测（ϵ-prediction）参数化，建立了与 denoising score matching 和 Langevin dynamics 的理论联系
3. 提出简化训练目标 L_simple，在实践中显著优于完整变分下界
4. 分析了扩散模型的率失真行为：超过一半的 lossless codelength 用于描述人眼不可察觉的图像细节

---

## Method 核心方法

### 1. 扩散模型的数学框架

扩散模型定义了一个潜变量模型 p_θ(x_0) = ∫ p_θ(x_0:T) dx_1:T。

**前向过程（扩散过程）**：固定的马尔可夫链，逐步向数据添加高斯噪声：

q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)

其中 β_t 是方差 schedule，从 β_1=10^{-4} 线性增加到 β_T=0.02。

**关键性质**：前向过程支持任意时间步 t 的闭式采样：

q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)

其中 α_t = 1-β_t, ᾱ_t = ∏_{s=1}^t α_s。

**反向过程（生成过程）**：学习的马尔可夫链，从标准高斯噪声开始逐步去噪：

p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t^2 I)

训练目标是最小化负对数似然的变分下界。通过推导，可将变分下界重写为 KL 散度之和：

L = E_q [D_KL(q(x_T|x_0) || p(x_T)) + Σ D_KL(q(x_{t-1}|x_t, x_0) || p_θ(x_{t-1}|x_t)) - log p_θ(x_0|x_1)]

![](https://ar5iv.labs.arxiv.org/html/2006.11239/assets/x2.png)

*Figure 2: DDPM 的马尔可夫链结构——前向过程 q 从右向左逐步加噪（x_0 → x_T），反向过程 p 从左向右逐步去噪（x_T → x_0），每个步骤都是一个高斯变换。*

### 2. 噪声预测参数化（核心创新）

将 μ_θ 参数化为预测噪声而非预测均值：

μ_θ(x_t, t) = (1/√α_t) (x_t - (β_t/√(1-ᾱ_t)) ε_θ(x_t, t))

这使得训练目标简化为：

||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||^2

这一参数化直接建立了扩散模型与 denoising score matching 的联系，采样过程也类似于 Langevin 动力学。

### 3. 简化训练目标 L_simple

作者发现，使用不加权的简化目标训练能获得更好的样本质量：

L_simple(θ) = E_{t, x_0, ε} [||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||^2]

其中 t 在 1 到 T 之间均匀采样。这个目标去掉了变分下界中的权重系数，相当于对不同的 t 给予等权重——实际上下调了小 t（对应极小噪声）的 loss 权重，让网络更专注于大 t 时更困难的去噪任务。

**算法流程**：
- **训练**：采样 x_0、t、ε → 计算加噪图像 x_t → 梯度下降优化 ||ε - ε_θ(x_t, t)||^2
- **采样**：从 x_T ~ N(0, I) 开始 → 迭代 T 步逐步去噪 → 输出 x_0

### 4. 网络架构

使用基于 U-Net 的主干网络，类似 unmasked PixelCNN++，采用：
- Group Normalization 替代 Batch Norm
- 16x16 分辨率处的 self-attention
- Transformer 正弦位置嵌入用于编码时间步 t
- 参数跨时间共享

---

## 实验/评估/结果

### 生成质量

**CIFAR10 无条件生成**：
- IS（Inception Score）：9.46 —— 超越多数条件生成模型
- FID：3.17 —— state-of-the-art，超越当时的大多数无条件模型
- 使用 L_simple 训练的模型显著优于使用完整变分下界训练的模型

**CelebA-HQ 256x256**：生成质量与 ProgressiveGAN 相当。

**LSUN 256x256**：Church 类别 FID = 7.89。

### 消融实验：参数化与目标函数

| 参数化 | 目标函数 | IS | FID |
|--------|---------|-----|-----|
| μ̃ 预测 | L, fixed Σ | 8.06 | 13.22 |
| ε 预测 | L, fixed Σ | 7.67 | 13.51 |
| ε 预测 | L_simple | **9.46** | **3.17** |

关键发现：
1. ε-预测 + L_simple 的组合远优于其他选项
2. 学习方差（learned Σ）导致训练不稳定
3. 预测 x_0 效果较差

### 渐进式有损压缩

- 超过一半的 lossless codelength 描述了人眼不可察觉的图像细节（RMSE = 0.95 on 0-255 scale）
- 扩散模型的采样过程天然可解释为一种渐进式解码，与自回归解码共享相似的位序结构
- 提出了一种利用扩散模型进行渐进式有损压缩的算法

---

## 结论

DDPM 首次证明扩散模型能够生成高质量图像样本，并建立了扩散模型与变分推断、去噪得分匹配、退火 Langevin 动力学、自回归模型和渐进式有损压缩之间的理论联系。扩散模型对图像数据具有优良的归纳偏置，作者期待其在其他数据模态中也能发挥作用。

---

## 思考

### 优点

1. **理论与实践的完美结合**：从非平衡热力学出发，经过严谨的数学推导，得出简洁优美的 ε-预测+L_simple 方案，理论和实验高度一致。

2. **开创性贡献**：DDPM 是扩散模型领域的奠基性工作。从 2020 年至今，几乎所有扩散模型（Stable Diffusion、DALL-E 2、FLUX 等）都建立在 DDPM 的框架之上。L_simple 目标已成为扩散模型训练的事实标准。

3. **洞察深刻**："噪声预测 v.s. 均值预测"的分析非常精妙。预测噪声不仅使训练目标简化，还建立了与 score matching 的理论联系，这是连接扩散模型和能量模型的桥梁。

4. **实验严谨**：详细的消融实验验证了每个设计选择的必要性——参数化方式、学习率、目标函数权重。

5. **率失真分析**：对扩散模型压缩行为的分析角度独特，揭示了模型将大量码率用于不可察觉细节的有趣现象。

### 缺点与局限

1. **采样速度慢**：T=1000 步的串行采样过程极为缓慢，生成一张图可能需要数分钟。这使得 DDPM 在实时应用中不可行。虽然后来的 DDIM 等方法解决了这个问题，但这是 DDPM 的内在局限。

2. **对数似然不具竞争力**：与自回归模型等基于似然的生成模型相比，DDPM 的 log-likelihood 明显较差，意味着在无损压缩任务上不占优。

3. **小型数据集的验证**：实验主要在 CIFAR10（32x32）、LSUN（256x256）、CelebA-HQ（256x256）上进行，与后来在 LAION 等大规模数据集上的工作相比，其泛化能力的边界不明确。

4. **Variance schedule 恒定性**：β_t 被设为常量（无学习），但后来的工作证明学习 β_t 或使用 cosine schedule 能进一步提升性能。

### 与已有 Wiki 的连接

- 关联概念：[[扩散模型]]、[[Score Matching]]、[[Langevin Dynamics]]、[[变分推断]]、[[U-Net]]
- 关联实体：[[DDPM]]、[[DDIM]]、[[Stable Diffusion]]
- 关联论文：[[AI阅读笔记/Flow Matching]]（Flow Matching 是扩散模型的泛化）
- 关联比较：[[Wiki/Comparisons/扩散模型架构比较 UNet vs DiT]]

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
  - score-matching
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

DDPM 的方法论由四个紧密关联的组件构成：扩散过程的数学定义 → 变分训练目标的推导 → 重参数化技巧 → 网络架构设计。以下按数学推导的自然顺序展开。

### 1. 扩散模型的数学框架

扩散模型定义了一个 T 步潜变量模型 p_θ(x_0) = ∫ p_θ(x_0:T) dx_1:T，其中 x_1, ..., x_T 是与数据 x_0 同维度的隐变量。

#### 1.1 前向过程（扩散过程）

前向过程是一个**固定的马尔可夫链**（无学习参数），逐步向数据添加高斯噪声，将数据分布 q(x_0) 逐渐转化为标准高斯分布：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

其中 β_t ∈ (0,1) 是方差 schedule，控制每步添加的噪声量。论文使用从 β_1=10^{-4} 到 β_T=0.02 的线性 schedule，共 T=1000 步。

**重参数化技巧**：令 α_t = 1-β_t，ᾱ_t = ∏_{s=1}^t α_s，利用高斯分布的可加性，可以**直接从 x_0 采样任意时间步 t 的 x_t**，无需迭代 T 步：

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$$

等价地：$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \varepsilon$，其中 $\varepsilon \sim \mathcal{N}(0, I)$。

这一性质是扩散模型高效训练的关键——训练时无需模拟完整的前向链，只需采样 t 和 ε 即可直接计算 x_t。

**边界条件**：当 T→∞ 时 β_t 的累积使 ᾱ_T→0，因此 $q(x_T|x_0) \approx \mathcal{N}(0, I)$。

#### 1.2 反向过程（生成过程）

反向过程是一个**学习的马尔可夫链**，从标准高斯噪声 x_T ~ N(0, I) 开始，逐步去噪重建数据：

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

其中 μ_θ 由神经网络预测，σ_t 设为固定值（β_t 或 (1-ᾱ_{t-1})/(1-ᾱ_t)·β_t，两者效果相近）。

关键问题：**如何训练 μ_θ 以逆转扩散过程？** 这需要推导训练目标。

### 2. 变分下界的推导与分解

训练目标是最大化数据对数似然 log p_θ(x_0) 的变分下界（VLB）：

$$\mathbb{E}_{q(x_0)}[-\log p_\theta(x_0)] \leq \mathbb{E}_q\left[-\log p(x_T) - \sum_{t \geq 1} \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right] =: L$$

经过代数变换，将 VLB 分解为三个可解释的 KL 散度项：

$$L = \underbrace{D_{KL}(q(x_T|x_0) \| p(x_T))}_{L_T\text{: 先验匹配}} + \sum_{t=2}^T \underbrace{D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}\text{: 去噪匹配}} - \underbrace{\log p_\theta(x_0|x_1)}_{L_0\text{: 解码}}$$

其中最关键的是 **L_{t-1} 项**——将学习问题从"预测前向过程的逆"转化为"匹配真实后验 q(x_{t-1}|x_t, x_0)"。

**真实后验的闭式解**：利用贝叶斯规则和前向过程的性质，可得：

$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)$$

其中：
$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t$$

$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$$

于是每个 L_{t-1} 项变成两个高斯分布之间的 KL 散度，可闭式计算。

### 3. 重参数化：预测噪声而非均值（核心创新）

将 μ_θ 参数化为预测 **μ̃_t** 是一种自然选择，但 DDPM 发现了一个更好的方案。

**三种可能的参数化方式**（按效果从劣到优排列）：

| 参数化 | 预测目标 | 问题 |
|--------|---------|------|
| 直接预测 x_0 | $\hat{x}_0 = f_\theta(x_t, t)$ | 在 t 接近 T 时，x_t 几乎是纯噪声，预测 x_0 极不稳定 |
| 预测 μ̃_t | $\mu_\theta = f_\theta(x_t, t)$ | 均值预测的损失面不平滑，收敛慢 |
| **预测噪声 ε**（DDPM 选择） | $\varepsilon_\theta(x_t, t)$ | 目标 ε 始终来自 N(0,I)，尺度稳定，损失面平滑 |

**噪声预测参数化**的数学形式。将 x_0 用 x_t 和 ε 表示：$x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t}\varepsilon)$，代入 μ̃_t 公式得：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\varepsilon_\theta(x_t, t)\right)$$

此时 L_{t-1} 的 KL 散度简化为：

$$L_{t-1} = \frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}\|\varepsilon - \varepsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\varepsilon, t)\|^2$$

### 4. 简化训练目标 L_simple——"魔鬼在权重里"

DDPM 最关键的实验发现是：**去掉 VLB 中 t 相关的权重系数，使用不加权的 MSE 目标，样本质量反而大幅提升**。

$$L_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \varepsilon}\left[\|\varepsilon - \varepsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\varepsilon, t)\|^2\right]$$

其中 t 在 {1, ..., T} 中均匀采样。

**为什么 L_simple 更好？** VLB 中的权重 $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}$ 随 t 剧烈变化：小 t（噪声很少）时权重极大，大 t（噪声很大）时权重极小。L_simple 的均匀权重相当于：
- **下调小 t 的权重**：小 t 时去噪任务过于简单（几乎只需复制输入），不应主导训练
- **上调大 t 的权重**：大 t 时的去噪才是真正困难的生成任务

这一发现深刻影响了后续所有扩散模型——**L_simple 成为训练扩散模型的事实标准**，包括 Stable Diffusion、FLUX、DALL-E 等。

### 5. 训练与采样算法

完整的训练与采样流程可以形式化地描述如下：

**训练算法**（重复直到收敛）：
1. 采样数据点 $x_0 \sim q(x_0)$
2. 采样时间步 $t \sim \text{Uniform}(\{1, ..., T\})$
3. 采样噪声 $\varepsilon \sim \mathcal{N}(0, I)$
4. 前向加噪：$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\varepsilon$
5. 梯度下降：$\nabla_\theta \|\varepsilon - \varepsilon_\theta(x_t, t)\|^2$

**采样算法**（生成图像）：
1. $x_T \sim \mathcal{N}(0, I)$
2. 对于 t = T, T-1, ..., 1：
   - 采样 $z \sim \mathcal{N}(0, I)$（当 t > 1，否则 z = 0）
   - $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\varepsilon_\theta(x_t, t)\right) + \sigma_t z$
3. 输出 $x_0$

![](https://ar5iv.labs.arxiv.org/html/2006.11239/assets/x2.png)

*Figure 2: DDPM 的马尔可夫链结构。前向过程 q（右→左）逐步加噪 x_0 → x_T，反向过程 p（左→右）逐步去噪 x_T → x_0，每步是一个高斯变换。训练时只需采样 t 和 ε 直接计算 x_t；采样时需迭代 T 步。*

### 6. 网络架构：U-Net 的扩散适配

DDPM 使用基于 U-Net 的主干网络，针对扩散任务做了若干关键适配：

| 组件 | 设计选择 | 原因 |
|------|---------|------|
| **主干** | U-Net（类似 unmasked PixelCNN++） | 多尺度特征对去噪至关重要（大尺度结构 + 细粒度纹理） |
| **归一化** | Group Normalization（32组） | 比 Batch Norm 更适合小 batch 训练 |
| **注意力** | 在 16×16 分辨率处插入自注意力层 | 全局一致性对生成质量关键（如对称性、结构连贯性） |
| **时间嵌入** | Transformer 式正弦位置编码 | 将离散时间步 t 映射为连续向量，注入每层的特征图 |
| **参数共享** | 所有时间步共享同一网络 | 用时间嵌入区分不同 t，大幅减少参数量 |
| **上采样/下采样** | 带残差连接的卷积 | 保持信息流动，避免梯度消失 |
| **激活函数** | SiLU (Swish) | 比 ReLU 平滑，利于梯度传播 |

时间步 t 的嵌入通过与每层特征图相加（或 FiLM 调制）注入网络，这使得单个网络能够处理从几乎无噪声（t≈1）到纯噪声（t≈T）的全部去噪难度范围。

### 7. 与 Score Matching 和 Langevin Dynamics 的理论联系

DDPM 建立了扩散模型与两个独立理论框架之间的深刻联系：

**与 Denoising Score Matching 的联系**：
Score Matching 学习数据分布的得分函数 $\nabla_x \log p(x)$。在扩散模型中，ε-预测与得分函数存在精确对应：

$$\nabla_{x_t} \log q(x_t|x_0) = -\frac{\varepsilon}{\sqrt{1-\bar{\alpha}_t}}$$

因此 $\varepsilon_\theta(x_t, t)$ 本质上是**在预测（缩放的）得分函数**。DDPM 的 L_simple 与 Denoising Score Matching 的损失函数等价（仅差一个常数因子）。

**与 Langevin Dynamics 的联系**：
采样过程 $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\varepsilon_\theta) + \sigma_t z$ 在连续极限下退化为 Langevin 动力学：

$$dx = -\frac{1}{2}\nabla_x U(x)dt + dW_t$$

这意味着扩散模型的采样过程可以理解为**在数据分布的"势能面"上执行退火 Langevin 动力学**——从高温（大噪声）逐步降温，最终落入数据分布的典型样本区域。

这两个理论联系解释了扩散模型的优异生成质量：ε-预测参数化将复杂的生成问题转化为简单的去噪问题，而 Langevin 动力学的退火性质则保证了采样过程的稳定性。

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

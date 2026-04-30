---
title: Flow Matching for Generative Modeling
authors:
  - Yaron Lipman
  - Ricky T. Q. Chen
  - Heli Ben-Hamu
  - Maximilian Nickel
  - Matt Le
institutions:
  - Meta AI (FAIR)
  - Weizmann Institute of Science
aliases:
  - Flow Matching
  - CFM
  - Conditional Flow Matching
date: '2026-04-30'
publish_date: 2023-02
venue: 'ICLR 2023'
tags:
  - 论文
  - 流匹配
  - 连续归一化流
  - 生成模型
  - 扩散模型替代
url: 'https://arxiv.org/abs/2210.02747'
code: 'https://github.com/facebookresearch/flow_matching'
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：本文提出了 Flow Matching（FM），一种无需模拟即可训练连续归一化流（CNF）的新范式，并通过条件概率路径的设计将扩散模型统一到 CNF 框架中，其最优传输（OT）路径变体以更少的采样步数实现了优于或持平扩散模型的生成质量。

![](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/x9.png)

*Figure 1: Flow Matching 的概览——通过条件概率路径设计，将简单先验分布（左侧高斯）连续变换为目标数据分布（右侧），无需模拟 ODE 即可训练 CNF。OT 路径（直线）比扩散路径（曲线）更高效。*

---

## Intro

### Motivation

连续归一化流（Continuous Normalizing Flows, CNFs）是一类强大的生成模型，能够通过 ODE 将简单分布连续变换为目标数据分布。然而，CNF 的训练非常困难：传统方法依赖于对 ODE 的数值模拟（需要多次前向传播）和伴随方法（adjoint method）计算梯度，计算成本极高。已有的替代方案（如 score matching 或噪声条件得分网络）本质上已经偏离了"流"的框架。

与此同时，扩散模型（DDPM 等）在图像生成领域取得了巨大成功，但它们的采样过程需要大量串行步骤（通常 100-1000 步），推理速度慢。扩散模型本质上也定义了一个从噪声到数据的"流"，但这种关系没有被充分利用。

### 核心主张

**Flow Matching 提供了一种简单、高效、无需模拟的 CNF 训练方法**。关键思想是：

1. 不直接学习从噪声分布到数据分布的复杂流，而是学习一组**条件流**（从噪声到单个数据点），然后通过边缘化得到无条件流
2. 将扩散模型的训练过程重新解释为 Flow Matching 的特例——扩散路径和 OT 路径都是条件概率路径的不同选择
3. OT 路径产生直线轨迹，允许更少的采样步数

### 贡献

1. 提出 Conditional Flow Matching (CFM) 目标，无需 ODE 模拟即可训练 CNF
2. 证明 CFM 与 FM 有相同的梯度，是严格的泛化形式
3. 提出使用 OT 路径（最优传输）作为条件概率路径，产生更简单的直线流
4. 将扩散模型的前向过程和训练目标统一到 Flow Matching 框架中
5. 实验证明 OT-CFM 在更少的采样步数下达到与扩散模型相当或更好的生成质量

---

## Method 核心方法

### 1. 连续归一化流（CNF）基础

CNF 定义了一个概率密度路径 p_t，由向量场 v_t 通过 ODE 生成：

d/dt φ_t(x) = v_t(φ_t(x), t)

其中 φ_0(x) = x，p_0 是简单先验（如高斯），p_1 近似数据分布 q。

### 2. Flow Matching (FM) 目标

FM 的目标是学习一个向量场 v_t 来生成从 p_0 到 p_1 ≈ q 的概率路径。直觉上，如果已知概率路径 p_t(x) 和生成它的向量场 u_t(x)，可以直接回归：

L_FM(θ) = E_{t, p_t(x)} [||v_θ(x, t) - u_t(x)||^2]

但问题在于：**p_t 和 u_t 是未知的**——这就是 CNF 训练一直需要模拟的原因。

### 3. Conditional Flow Matching (CFM)——核心创新

CFM 的关键洞察是：虽然边际概率路径 p_t 和对应的向量场 u_t 未知，但可以定义**条件路径**——以单个数据点 x_1 为条件的概率路径 p_t(x | x_1) 和条件向量场 u_t(x | x_1)。然后通过 CFM 目标训练：

L_CFM(θ) = E_{t, q(x_1), p_t(x|x_1)} [||v_θ(x, t) - u_t(x|x_1)||^2]

**核心定理**：FM 和 CFM 有相同的梯度（∇_θ L_FM = ∇_θ L_CFM），但 CFM 完全不需要知道未知的边际向量场，只需要容易计算的条件向量场。

### 4. 条件概率路径的设计

条件概率路径需要满足：
- p_0(x | x_1) = p_0(x)（与 x_1 无关，因为开始时没有任何信息）
- p_1(x | x_1) ≈ δ(x - x_1)（最终集中在 x_1 附近，通常用均值 x_1、方差很小的高斯近似）

#### 高斯条件路径

最自然的构造是使用高斯条件分布：

p_t(x | x_1) = N(x; μ_t(x_1), σ_t^2 I)

需要满足 μ_0 = 0, σ_0 = 1 和 μ_1 = x_1, σ_1 = σ_min。

对应的条件向量场也取高斯形式：u_t(x | x_1) = (x - μ_t(x_1)) σ_t'/σ_t + μ_t'(x_1)

#### 两种关键路径设计

**扩散路径（Diffusion Path）**：μ_t = √ᾱ_t x_1, σ_t = √(1-ᾱ_t)
- 对应 DDPM 的前向过程，训练目标等价于 ε-预测去噪得分匹配
- **Flow Matching 将 DDPM 统一为 CNF 特例**

**OT 路径（Optimal Transport Path）**：μ_t = t x_1, σ_t = 1 - (1-σ_min)t
- 均值和方差均随 t 线性变化
- 产生**直线轨迹**，比扩散路径更简单
- 向量场形式：u_t(x | x_1) = (x_1 - (1-σ_min)x) / (1 - (1-σ_min)t)

![](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/figures/2d_traj/2d_traj_diff.png)

*Figure 2: Diffusion 路径（上）与 OT 路径（下）的 2D 轨迹对比。OT 路径产生直线流，从源分布到目标分布的路径更直接、更短，因此可以用更少的 ODE 步骤完成采样。*

### 5. 与扩散模型的统一

CFM 框架将 DDPM 的训练过程天然解释为 Flow Matching：
- DDPM 的噪声预测目标等价于学习 OT 路径下的条件向量场
- DDPM 的随机采样（SDE）在 CFM 中对应 ODE 采样
- CFM 可以无缝结合扩散模型积累的工程实践（U-Net 架构、EMA、classifier-free guidance 等）

---

## 实验/评估/结果

### 2D 玩具数据集

在 8 Gaussians、Moons、Checkerboard 等二维分布上进行验证：
- OT-CFM 比扩散路径生成更直接的流线（直线 v.s. 曲线）
- OT-CFM 需要显著更少的 ODE 步骤（N=100 v.s. N=1000）

### 图像生成

**CIFAR-10（32x32）**：
- OT-CFM 生成质量与 DDPM 相当
- 使用少至 10 步采样即能产生合理质量（DDPM 需要 1000 步）

**ImageNet 64x64**：
- OT-CFM + ODE 采样在 142 步达到 FID 2.76（与 SOTA 扩散模型可比）
- 使用更少步骤（如 5 步）时质量仍优于同条件扩散模型
- 与 score SDE、DDPM 的对比证明了 CFM 框架的优势

![](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/figures/imagenet128/imagenet128_curated_.png)

*Figure 3: Flow Matching 在 ImageNet 128x128 上的生成样本。OT-CFM 生成的图像质量与扩散模型相当或更优，但需要更少的采样步数。*

### 关键实验结论

1. OT-路径比扩散路径更高效（直线轨迹 → 更少步数）
2. ODE 采样（确定性）比 SDE 采样（随机性）在少步数时更稳定
3. Flow Matching 可以直接复用扩散模型的架构和训练策略
4. 少步采样时 OT-CFM 的优势最为明显

---

## 结论

Flow Matching 提供了一种统一且极其简洁的 CNF 训练范式——通过条件概率路径的设计，无需 ODE 模拟即可高效训练。它将扩散模型纳入框架作为特例，同时 OT 路径以其直线轨迹实现了更高效的采样。Flow Matching 为后续一系列工作（如 FLUX、Stable Diffusion 3 使用的 Rectified Flow）奠定了基础。

---

## 思考

### 优点

1. **理论简洁优美**：将"复杂问题→简单条件问题→边缘化"的策略运用得十分精妙。一个定理（CFM = FM 的梯度）就解决了 CNF 训练的根本难题。

2. **统一框架**：将扩散模型正式纳入 CNF/Flow 框架，澄清了"扩散也是流"这一关系。这让扩散模型的工程经验（架构、训练技巧）可以无缝迁移到 Flow Matching。

3. **实用性极强**：OT 路径的直线轨迹属性对推理效率至关重要。少步采样（10-50 步）的能力使扩散/流模型在实际部署中变得可行。这是 DDIM/一致性模型等加速工作的理论基础。

4. **开源和可复现性**：官方代码库（PyTorch）清晰完整，推动了后续大量工作。

5. **深远影响**：FLUX、SD3 等 SOTA 文本到图像模型使用的 Rectified Flow / Flow Matching 都直接源自此处。

### 缺点与待解决问题

1. **非高斯路径的探索有限**：论文主要基于高斯条件路径。更复杂的路径（如重尾分布、离散数据）的设计空间未被充分探索。

2. **与扩散模型在高分辨率上的直接对比不足**：实验主要在 CIFAR-10 和 ImageNet 64x64 上进行，缺少与大规模扩散模型（256x256 以上分辨率）的正面比较。

3. **Flow 稳定性的理论分析欠缺**：虽然 OT 路径在实践中更稳定，但缺少对"为什么不同的概率路径选择影响向量场的优化难度"的深入理论分析。

4. **distillation/加速方法的系统性研究在当时尚早**：论文提出的 ODE 采样天然支持少步，但如何与 consistency models、progressive distillation 等方法结合，论文没有展开。

### 与已有 Wiki 的连接

- 关联概念：[[连续归一化流]]、[[最优传输]]、[[扩散模型]]、[[Rectified Flow]]
- 关联实体：[[DDPM]]、[[DDIM]]、[[FLUX]]
- 关联论文：[[AI阅读笔记/DDPM]]、[[AI阅读笔记/FLUX Kontext]]
- 关联比较：[[Wiki/Comparisons/扩散模型与流匹配模型对比]]

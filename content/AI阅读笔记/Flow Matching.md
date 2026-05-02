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

Flow Matching 的方法论遵循一个清晰的逻辑链条：CNF 基础 → FM 目标的不可行性 → CFM 的等价性突破 → 条件路径的设计空间 → OT 最优性 → 工程实现。以下逐步展开。

### 1. 连续归一化流（CNF）基础

CNF 是一类通过常微分方程（ODE）定义概率变换的生成模型。核心对象有三个：

**概率密度路径** $p: [0,1] \times \mathbb{R}^d \to \mathbb{R}_{>0}$，满足 $\int p_t(x)dx = 1$。边界条件：$p_0 = p$（简单先验，如标准高斯），$p_1 \approx q$（数据分布）。

**时间依赖向量场** $v_t: [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$，定义微分同胚流 $\phi_t$ 满足：

$$\frac{d}{dt}\phi_t(x) = v_t(\phi_t(x)), \quad \phi_0(x) = x$$

**前推方程（Push-forward）**：$p_t = [\phi_t]_* p_0$，即 $\phi_t$ 将先验分布 p_0 变换为 p_t。连续方程 $\frac{d}{dt}p_t + \text{div}(p_t v_t) = 0$ 提供了验证向量场是否生成概率路径的必要充分条件。

传统 CNF 训练使用最大似然（需求解 ODE 的正反向传播），计算量极大，是阻碍其规模化的根本瓶颈。

### 2. Flow Matching (FM) 目标——直觉与困难

FM 的核心思想是直接回归生成目标概率路径的向量场。假设已知目标概率路径 p_t(x) 及其生成向量场 u_t(x)，FM 目标简单直观：

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, p_t(x)}\left[\|v_\theta(x, t) - u_t(x)\|^2\right]$$

其中 $t \sim \mathcal{U}[0,1]$，$x \sim p_t(x)$。

**根本困难**：p_t 和 u_t 都是未知的——我们不知道从 p_0 到 p_1 的概率路径"长什么样"，也不知道生成它的向量场是什么。这正是传统 CNF 训练必须借助 ODE 模拟的原因。FM 初看似乎陷入了循环论证。

### 3. Conditional Flow Matching (CFM)——突破性洞察

CFM 的解决方案极为巧妙：**将未知的边际问题分解为已知的条件问题**。

#### 3.1 条件概率路径的构造

给定单个数据点 x_1，定义以 x_1 为条件的**条件概率路径** $p_t(x|x_1)$，满足：
- $p_0(x|x_1) = p_0(x)$：t=0 时与 x_1 无关（所有路径从同一先验出发）
- $p_1(x|x_1) \approx \delta_{x_1}$：t=1 时集中在 x_1 附近（用均值 x_1、标准差 σ_min 的高斯近似）

边际概率路径通过对数据分布 q(x_1) 做边缘化得到：

$$p_t(x) = \int p_t(x|x_1) q(x_1) dx_1$$

#### 3.2 边际向量场的边缘化公式

类似地，定义**条件向量场** $u_t(\cdot|x_1)$（生成 $p_t(\cdot|x_1)$），边际向量场通过加权平均得到：

$$u_t(x) = \int u_t(x|x_1) \frac{p_t(x|x_1) q(x_1)}{p_t(x)} dx_1$$

#### 3.3 核心定理（Theorem 1 & 2）

**Theorem 1（生成性）**：以上定义的边际向量场 u_t 确实生成边际概率路径 p_t，即满足连续方程。

**Theorem 2（等价性，最关键的突破）**：FM 和 CFM 目标有完全相同的梯度：

$$\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)$$

其中 CFM 目标为：

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, q(x_1), p_t(x|x_1)}\left[\|v_\theta(x, t) - u_t(x|x_1)\|^2\right]$$

**为什么这是突破性的？** CFM 目标只依赖于条件向量场 u_t(x|x_1)，而后者是**容易计算**的（只需要单个数据点 x_1 的信息，不需要知道整个数据分布的形状）。Theorem 2 的精妙之处在于：优化简单的条件目标等价于优化困难的边际目标。

#### 3.4 证明直觉（Theorem 2 的证明概要）

核心技巧是将 FM 损失展开为三项：

$$\|v_t - u_t\|^2 = \|v_t\|^2 - 2\langle v_t, u_t \rangle + \|u_t\|^2$$

其中 $\mathbb{E}_{p_t}\|v_t\|^2 = \mathbb{E}_{q(x_1), p_t(x|x_1)}\|v_t\|^2$（通过边缘化 $p_t$），而交叉项 $\mathbb{E}_{p_t}\langle v_t, u_t \rangle = \mathbb{E}_{q(x_1), p_t(x|x_1)}\langle v_t, u_t(x|x_1) \rangle$（通过 u_t 的边缘化定义和 Fubini 定理交换积分顺序）。因此 FM 和 CFM 在期望意义上等价。

### 4. 高斯条件路径的设计空间

CFM 框架对条件路径的设计是完全开放的。论文重点探索了**高斯条件路径族**：

$$p_t(x|x_1) = \mathcal{N}(x | \mu_t(x_1), \sigma_t(x_1)^2 I)$$

边界条件：$\mu_0 = 0, \sigma_0 = 1$ 和 $\mu_1 = x_1, \sigma_1 = \sigma_{\min}$。

#### 4.1 条件向量场的闭式解（Theorem 3）

论文证明了对于高斯路径，最简单的仿射流 $\psi_t(x) = \sigma_t(x_1)x + \mu_t(x_1)$ 对应的向量场为：

$$u_t(x|x_1) = \frac{\sigma'_t(x_1)}{\sigma_t(x_1)}(x - \mu_t(x_1)) + \mu'_t(x_1)$$

通过重参数化 $x = \psi_t(x_0)$（其中 $x_0 \sim p(x_0) = \mathcal{N}(0, I)$），CFM 损失简化为：

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, q(x_1), p(x_0)}\left[\|v_\theta(\psi_t(x_0), t) - (x_1 - (1-\sigma_{\min})x_0)\|^2\right]$$

#### 4.2 两种关键路径的对比分析

**扩散路径（Diffusion/Variance Preserving Path）**：

$$\mu_t = \alpha_{1-t}x_1, \quad \sigma_t = \sqrt{1-\alpha_{1-t}^2}$$

其中 $\alpha_t = e^{-\frac{1}{2}T(t)}$，$T(t) = \int_0^t \beta(s)ds$。这精确对应 DDPM 的 VP-SDE 前向过程。

- 轨迹呈**曲线**，在 t→1 处有"过冲"现象
- 在 t=1 处理论上不能精确到达干净数据（需要 σ_min 近似）
- 对应向量场：$u_t(x|x_1) = -\frac{T'(1-t)}{2}\left[\frac{e^{-T(1-t)}x - e^{-\frac{1}{2}T(1-t)}x_1}{1-e^{-T(1-t)}}\right]$

**OT 路径（Optimal Transport / Rectified Flow Path）**：

$$\mu_t = t x_1, \quad \sigma_t = 1 - (1-\sigma_{\min})t$$

- 均值和方差**线性变化**，轨迹为**直线**
- 在 t∈[0,1] 整个区间都有良好定义（不需要截断）
- 对应向量场：$u_t(x|x_1) = \frac{x_1 - (1-\sigma_{\min})x}{1 - (1-\sigma_{\min})t}$
- 向量场可分解为 $u_t(x|x_1) = g(t) \cdot h(x|x_1)$（方向恒定，幅值随时间变化）

![](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/figures/2d_traj/2d_traj_diff.png)

*Figure 2: Diffusion 路径（上）与 OT 路径（下）的 2D 轨迹对比。注意扩散路径的弯曲和过冲现象——粒子先越过目标再折返；OT 路径始终保持恒定方向的直线运动，因此可用更少 ODE 步骤完成采样。*

### 5. OT 路径的最优性

OT 路径不只是一个"工程选择"，它具有严格的数学最优性：

**Wasserstein-2 最优传输视角**：条件流 $\psi_t(x) = (1-(1-\sigma_{\min})t)x + tx_1$ 恰好是两个高斯分布 $p_0(\cdot|x_1) = \mathcal{N}(0, I)$ 和 $p_1(\cdot|x_1) = \mathcal{N}(x_1, \sigma_{\min}^2 I)$ 之间的**Wasserstein-2 最优传输位移映射**。

OT 插值（McCann 插值）定义为：

$$p_t = [(1-t)\text{id} + t\psi]_\star p_0$$

这意味着 OT 路径在"最优传输成本"意义上是最短的——粒子以恒定速度沿直线从源分布移动到目标分布，没有任何不必要的绕路。

**对训练的影响**：
- OT 向量场方向恒定（只随时间缩放），回归任务更简单，收敛更快
- OT 路径的直线性质使 ODE 求解器的局部截断误差更小
- 在相同 NFE（函数评估次数）下，OT 路径的数值精度更高

### 6. 训练 Pipeline 与工程实现

#### 6.1 网络架构

论文使用与 ADM（Dhariwal & Nichol, 2021）相同的 U-Net 架构，关键配置：

| 数据集 | 通道数 | 深度 | 注意力分辨率 | Batch Size | GPUs | 迭代数 |
|--------|--------|------|-------------|------------|------|--------|
| CIFAR-10 | 256 | 2 | 16 | 256 | 2 | 391K |
| ImageNet-32 | 256 | 3 | 16,8 | 1024 | 4 | 250K |
| ImageNet-64 | 192 | 3 | 32,16,8 | 2048 | 16 | 157K |
| ImageNet-128 | 256 | 3 | 32,16,8 | 1536 | 32 | 500K |

所有模型使用 Adam 优化器（β₁=0.9, β₂=0.999, weight decay=0），学习率 1e-4 到 5e-4。

#### 6.2 采样与推理

采样过程使用黑盒 ODE 求解器求解 $\frac{d}{dt}\phi_t(x_0) = v_\theta(\phi_t(x_0), t)$（从 t=0 到 t=1）：
- **高质量采样**：dopri5 自适应步长求解器（atol=rtol=1e-5）
- **快速采样**：固定步数 Euler/Midpoint 求解器（5-100 步）
- OT 路径在少步数下优势尤为明显——10 步 OT 的质量接近 100 步 diffusion

#### 6.3 对数似然评估

通过瞬时变量变换（instantaneous change of variables）公式计算：

$$\log p_1(x_1) = \log p_0(x_0) - \int_0^1 \text{div}(v_\theta(\phi_t(x_0), t)) dt$$

使用 Hutchinson 迹估计器（随机向量 z ∼ N(0,I) 满足 E[zz^T] = I）对散度做无偏估计，大幅降低计算复杂度（从 O(d²) 到 O(d)）。

### 7. 与扩散模型的统一关系

CFM 框架将扩散模型自然地纳入 CNF 框架：

| 扩散模型概念 | Flow Matching 对应 |
|-------------|-------------------|
| DDPM 的噪声预测 ε_θ | OT 路径的条件向量场 v_θ |
| Score Matching 目标 | 扩散路径下的 CFM |
| DDIM ODE 采样 | CNF ODE 求解器 |
| SDE 随机采样 | 可选的概率流 ODE 或 SDE 形式 |

关键结论：**扩散模型是 Flow Matching 在特定条件路径选择下的特例**；Flow Matching 跳出了扩散过程的框架限制，允许直接设计更优的概率路径。

## 实验/评估/结果

### 2D 玩具数据集

在 8 Gaussians、Moons、Checkerboard 等二维分布上进行验证：
- OT-CFM 比扩散路径生成更直接的流线（直线 v.s. 曲线）
- OT-CFM 需要显著更少的 ODE 步骤（N=100 v.s. N=1000）

### 图像生成

**主对比实验（Table 1 from paper）：相同 U-Net 架构，对比不同训练目标**

| Model | CIFAR-10 NLL↓ / FID↓ / NFE↓ | ImageNet-32 NLL↓ / FID↓ / NFE↓ | ImageNet-64 NLL↓ / FID↓ / NFE↓ |
|-------|------------------------------|--------------------------------|--------------------------------|
| DDPM | 3.12 / 7.48 / 274 | 3.54 / 6.99 / 262 | 3.32 / 17.36 / 264 |
| Score Matching | 3.16 / 19.94 / 242 | 3.56 / 5.68 / 178 | 3.40 / 19.74 / 441 |
| ScoreFlow | 3.09 / 20.78 / 428 | 3.55 / 14.14 / 195 | 3.36 / 24.95 / 601 |
| **FM w/ Diffusion** | 3.10 / 8.06 / 183 | 3.54 / 6.37 / 193 | 3.33 / 16.88 / 187 |
| **FM w/ OT** | **2.99** / **6.35** / **142** | **3.53** / **5.02** / **122** | **3.31** / **14.45** / **138** |

*FM-OT 在所有数据集、所有指标上一致最优。NLL 更低、FID 更低、NFE 更少（采样更高效）。FM+Diffusion 路径在 FID 上与 DDPM 可比但 NFE 更少。Score Matching/SF 的 FID 远高于 FM。*

**ImageNet 128×128**：FM-OT NLL 2.90、FID 20.9，仅用 500K iterations（batch 1536），是 ADM（4.36M iterations）图像吞吐量的 67%。

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

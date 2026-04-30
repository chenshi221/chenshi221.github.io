---
type: concept
status: active
created: "2026-04-30"
updated: "2026-04-30"
aliases: ["Flow Matching", "FM", "Conditional Flow Matching"]
tags: ["flow-matching", "CNF", "generative-model", "optimal-transport"]
sources: ["[[Wiki/Sources/Flow Matching 生成建模]]", "[[Wiki/Sources/FLUX.1 Kontext 上下文编辑]]"]
confidence: high
---

# Flow Matching

## 概述

Flow Matching (FM) 是 2023 年由 Meta AI 提出的一种生成建模范式，基于连续归一化流（Continuous Normalizing Flows, CNF）。它摒弃了扩散模型的 SDE 推导方式，直接在概率路径层面设计生成过程，通过回归向量场实现仿真自由（simulation-free）训练。核心优势：泛化了扩散模型，并引入 OT 路径实现更高效训练和采样。

## 核心原理

### 连续归一化流（CNF）

- 向量场 $v_t(x)$ 通过 ODE $\frac{d}{dt}\phi_t(x) = v_t(\phi_t(x))$ 定义流映射 $\phi_t$。
- $\phi_t$ 将简单先验分布 $p_0$（如高斯噪声）push-forward 为目标数据分布 $p_1$：$p_t = [\phi_t]_* p_0$。

### Flow Matching 目标

- 直接回归目标向量场：$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, p_t(x)}\|v_t(x) - u_t(x)\|^2$
- 问题：$u_t$ 通常是不可处理（intractable）的。

### Conditional Flow Matching (CFM)

- 核心技巧：通过在条件概率路径 $p_t(x|x_1)$ 上定义条件向量场 $u_t(x|x_1)$，证明 CFM 目标与 FM 目标梯度相同。
- $\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, q(x_1), p_t(x|x_1)}\|v_t(x) - u_t(x|x_1)\|^2$
- 只要能从 $p_t(x|x_1)$ 采样并计算 $u_t(x|x_1)$ 即可。

### 高斯条件路径统一公式

- $p_t(x|x_1) = \mathcal{N}(x|\mu_t(x_1), \sigma_t(x_1)^2 I)$
- 向量场：$u_t(x|x_1) = \frac{\sigma'_t}{\sigma_t}(x - \mu_t) + \mu'_t$
- 满足 $\mu_0=0, \sigma_0=1$（从噪声出发），$\mu_1=x_1, \sigma_1=\sigma_{\min}\approx 0$（到达数据）。

## 两种关键路径

### 扩散路径（Diffusion Path）

- VP 路径：$\mu_t = \alpha_{1-t}x_1$，$\sigma_t = \sqrt{1-\alpha_{1-t}^2}$，对应 DDPM 的扩散过程。
- VE 路径：$\mu_t = x_1$，$\sigma_t = \sigma_{1-t}$。
- FM 用扩散路径训练比 score matching 更稳定。

### 最优传输路径（OT Path）

- $\mu_t = tx_1$，$\sigma_t = 1 - (1-\sigma_{\min})t$（均值方差均线性变化）。
- 向量场：$u_t(x|x_1) = \frac{x_1 - (1-\sigma_{\min})x}{1 - (1-\sigma_{\min})t}$
- 对应 Wasserstein-2 最优传输位移，形成**直线轨迹**和**恒定速度**。
- 相比扩散路径：训练更快收敛、采样需要更少 NFE、FID 更优。

## 为什么 Flow Matching 比扩散模型更快

1. **直线 vs 曲线路径**：OT 路径是直线，扩散路径是曲线（可能"超调"），直线路径允许更少采样步数。
2. **定义在有限时间区间**：$t\in[0,1]$ 而非 $[0,\infty)$，无需近似截断。
3. **向量场回归优于分数匹配**：FM 直接回归向量场，训练更稳定，采样时无需 Langevin 动力学。
4. **与 Rectified Flow 的关系**：Rectified flow 使用 $z_t = (1-t)x + t\varepsilon$（固定方差=1 路径），可视为 FM-OT 在 $\sigma_{\min}=0$ 且无方差收缩的极限情况。FLUX 和 Seedream 3.0/4.0 均使用此路径。

## 实例应用

| 模型 | 使用的路径 |
|------|-----------|
| 原始 FM (Meta) | Diffusion (VP) / OT |
| FLUX.1 (BFL) | Rectified Flow (MMDiT) |
| FLUX.1 Kontext (BFL) | Rectified Flow + 序列拼接 |
| Stable Diffusion 3 | Rectified Flow (MMDiT) |
| Seedream 2.0/3.0/4.0 | Flow Matching (MMDiT) |

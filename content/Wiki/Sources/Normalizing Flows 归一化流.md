---
type: source
status: complete
created: '2026-05-07'
updated: '2026-05-07'
tags: [generative-model, normalizing-flows, density-estimation, foundations]
sources:
  - "[[Clippings/Normalizing Flows: An Introduction and Review of Current Methods]]"
---

# Normalizing Flows: An Introduction and Review of Current Methods

## 基本信息

- **标题**: Normalizing Flows: An Introduction and Review of Current Methods
- **作者**: Ivan Kobyzev, Simon J.D. Prince, Marcus A. Brubaker
- **机构**: Borealis AI; York University; University of Toronto; Vector Institute
- **年份**: 2019 (arXiv: 1908.09257)
- **类型**: 综述论文 (IEEE TPAMI)
- **互补文献**: Papamakarios et al. (2019) 偏教程/实现，本文偏形式化和模型族分类

## 核心论点

1. **可追踪的生成模型**: 归一化流是一类同时支持精确采样和精确密度估计的生成模型，克服了 GAN 无法计算似然、VAE 存在后验塌缩等缺陷。
2. **变量替换公式是核心机制**: 通过一串可逆、可微变换将简单基分布（通常为标准正态）推前到复杂数据分布，密度由基密度与 Jacobian 行列式的乘积决定。
3. **耦合函数决定表达力**: 耦合流和自回归流是两类最广泛使用的架构，它们共享"耦合函数"作为基本构建块；耦合函数从简单的仿射变换发展到样条、神经网络、多项式等形式，表达力逐步增强。
4. **通用性有理论保证**: 多类自回归流（包括 NAF、SOS、样条流）已被证明具有通用性——在足够容量和数据下可以逼近任意目标密度。
5. **连续时间流是重要方向**: 残差连接可视为 ODE 的离散化，Neural ODE 和 FFJORD 将流推广到连续动力系统，参数效率高且避免行列式计算，但采样需解 ODE。

## 关键技术方法

### 框架基础

- **变量替换公式**: $p_Y(y) = p_Z(f(y)) \cdot |\det Df(y)|$，其中 $f$ 为生成映射 $g$ 的逆。
- **链式分解**: 多个双射函数的复合仍是双射，Jacobian 行列式为各层行列式之积。
- **训练**: 最大化数据对数似然（等价于最小化 KL 散度），也可使用对抗损失（Flow-GAN）。

### 三大架构族

| 架构 | 核心思想 | 采样/密度效率 | 代表模型 |
|---|---|---|---|
| **耦合流** | 将输入分为两部分，一部分经双射变换，参数由另一部分决定 | 密度快，采样需反向 | NICE, RealNVP, Glow, Flow++ |
| **自回归流** | 每个维度的输出条件于前面所有维度 | 密度方向：MAF 并行快；生成方向：IAF 并行快 | MAF, IAF, NAF, SOS |
| **残差/连续流** | 残差连接视为 ODE 离散化，推广到连续动力系统 | 采样需解 ODE，密度通过 trace 估计 | iResNet, Residual Flow, FFJORD, Neural ODE |

### 线性变换

- 对角矩阵（线性复杂度但无法表达维度间相关性）
- 三角矩阵（行列式为对角元素之积，反向回代 $O(D^2)$）
- 置换与正交矩阵（Householder 变换、LU/QR 分解）
- $1\times1$ 卷积（Glow 的核心创新）和更一般的 $d\times d$ 卷积

### 耦合函数演进

- **仿射**: $h(x) = \theta_1 x + \theta_2$（NICE, RealNVP, Glow, MAF, IAF）——简单但表达力有限
- **非线性平方**: $h(x) = ax + b + c/(1+(dx+h)^2)$——可解析求逆
- **连续混合 CDF**: Flow++ 使用 K 个 logistic 混合的 CDF 后接逆 sigmoid
- **样条**: 分段线性、分段二次、单调三次、有理二次样条（RQ-NSF）——当前最佳性能之一
- **神经网络**: NAF 用正权重 MLP 保证单调性；UMNN 对正函数积分实现单调性
- **多项式**: SOS 用平方和多项式保证正导数，L=0 时退化为仿射
- **分段双射**: RAD 流允许函数分段双射而非全局双射，处理离散/多模态结构

### 残差与连续流

- **iResNet**: 残差块 Lipschitz 常数 < 1 保证可逆，用 Hutchinson 估计器随机估计 Jacobian trace
- **Residual Flow**: 用俄罗斯轮盘赌估计器替代截断，得到无偏估计
- **Neural ODE / FFJORD**: 将流视为 ODE 的 time-1 map，通过伴随灵敏度方法训练；FFJORD 用 Hutchinson 估计 trace，复杂度进一步降低
- **ANODE**: 增广状态空间使 Neural ODE 能表示所有微分同胚（通用性证明）
- **Langevin 流 / Neural SDE**: 引入随机微分方程，扩散项提供混合能力，但目前缺乏实际应用

## 主要结果

- 在表格数据集（POWER, GAS, HEPMASS, MINIBOONE, BSDS300）上，通用流（NAF, SOS, 样条流）表现最优，RQ-NSF 和 Cubic Spline 达到最佳或接近最佳对数似然。
- 在图像数据集（MNIST, CIFAR-10, ImageNet32/64）上，Flow++ 当时取得最佳结果（CIFAR-10: 3.08 bpd），其关键改进来自变分去量化而非耦合函数本身。
- FFJORD 参数效率极高：在 CIFAR-10 上达到可比性能时，参数量不到 Glow 的 2%。
- 残差流（Residual Flow）在 ImageNet 上表现优于 Glow 和 FFJORD。
- 仿射耦合函数（RealNVP, Glow）虽然简单高效，但需要堆叠更多层才能达到复杂分布的表达能力。

## 局限性

1. **高维计算瓶颈**: 线性变换的 Jacobian 行列式计算为 $O(D^3)$，三角矩阵反向为 $O(D^2)$；高维数据上流的计算成本显著。
2. **仿射耦合表达力不足**: 最常用的仿射耦合函数需要大量层堆叠，模型效率低。
3. **自回归流采样慢**: MAF 密度估计快但采样需递推（不可并行），IAF 相反；两者不可兼得。
4. **连续流的逆不精确**: Neural ODE 的逆需数值求解 ODE，存在数值误差和计算开销。
5. **离散分布扩展困难**: 归一化流假设连续分布，离散数据的建模仍为开放问题（去量化是变通方案而非根本解决）。
6. **Langevin 流缺乏实践**: SDE-based 方法理论优美但尚无基准实验验证。
7. **基分布选择影响被低估**: 理论上基分布不影响通用性，但实践中选择合适的基分布（如混合高斯）可显著简化学习。
8. **条件流建模不成熟**: 条件归一化流的参数共享和效率仍有改进空间。

## 与相关工作的关系

### 与 VAE 的关系

- 归一化流最初在变分推理背景下由 Rezende & Mohamed (2015) 推广，用于增强 VAE 后验分布的表达力。
- VAE 的近似后验 $q(y|x,\theta)$ 可通过归一化流参数化，实现重参数化技巧。
- VAE 无法精确计算似然（ELBO 是下界），而归一化流可以精确计算。
- VAE 存在后验塌缩问题，归一化流则面临训练不稳定的其他形式。

### 与 GAN 的关系

- GAN 生成质量高但无法计算密度，归一化流可以精确密度估计。
- Flow-GAN 探索了将对抗损失与归一化流结合的可能。
- 两类模型互为补充：GAN 擅长视觉质量，归一化流擅长密度评估和似然推理。

### 与扩散模型的关系

- Langevin 流（SDE-based 方法）是归一化流与扩散模型之间的桥梁，本文已涉及 Sohl-Dickstein et al. (2015) 的非平衡热力学工作（DDPM 的前身）。
- 扩散模型可视为 SDE 的特殊情况，归一化流的 ODE 视角后来发展为概率流 ODE（Score-based/SDE 统一框架）。
- 扩散模型在实践中超越了归一化流的图像生成质量，但归一化流的精确密度估计优势仍不可替代。

### 与 Flow Matching 的关系

- 本文讨论的 Neural ODE / FFJORD 为 Flow Matching（Lipman et al., 2023）奠定了基础。
- Flow Matching 直接学习从噪声到数据的速度场，避免了 FFJORD 中 Hutchinson trace 估计的方差问题。
- Conditional Flow Matching 可视为连续归一化流在训练方法上的突破。

### 与最优传输的关系

- 本文指出归一化流可置于最优传输理论框架下（Villani, 2003），pushforward 操作本质是传输映射。
- 后续的 OT-flow、Flow Matching 等工作将最优传输距离直接纳入流的训练目标。

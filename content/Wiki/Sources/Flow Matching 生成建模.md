---
type: source
status: processed
source_file: "[[Clippings/Flow Matching for Generative Modeling]]"
title: "Flow Matching for Generative Modeling"
site: "ICLR 2023"
author: "Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le (Meta AI / Weizmann Institute)"
published: "2023"
processed: "2026-04-30"
updated: "2026-04-30"
tags: ["flow-matching", "CNF", "generative-model", "optimal-transport"]
aliases: ["Flow Matching", "FM"]
confidence: high
---

# Flow Matching for Generative Modeling

## 核心结论

Flow Matching (FM) 是一种新的生成建模范式，基于连续归一化流（CNF），无需模拟即可高效训练。FM 通过回归条件概率路径的向量场实现训练，兼容通用高斯概率路径族，将扩散路径作为特例。最优传输（OT）路径比扩散路径更高效：形成直线轨迹（而非曲线），训练更快、采样更快、泛化更好。

## 关键方法

1. **Flow Matching 目标**：$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t,p_t(x)}\|v_t(x) - u_t(x)\|^2$，直接在时间区间 $[0,1]$ 上回归目标向量场 $u_t$。

2. **Conditional Flow Matching (CFM)**：利用条件概率路径 $p_t(x|x_1)$ 和条件向量场 $u_t(x|x_1)$，CFM 目标 $\mathcal{L}_{\text{CFM}}$ 与 FM 具有相同梯度，但只需逐样本计算即可。

3. **高斯条件路径的统一框架**：$p_t(x|x_1) = \mathcal{N}(x|\mu_t(x_1), \sigma_t(x_1)^2 I)$，向量场为 $u_t(x|x_1) = \frac{\sigma'_t}{\sigma_t}(x - \mu_t) + \mu'_t$。

4. **两种路径实例**：
   - **扩散路径**（VP/VE）：从 SDE 推导出的特定 $\mu_t$、$\sigma_t$ 选择，恢复到 DDPM 的分数匹配。
   - **OT 路径**：$\mu_t = tx_1$，$\sigma_t = 1 - (1-\sigma_{\min})t$，形成 Wasserstein-2 最优传输位移，直线轨迹、恒定速度。

5. **实验效果**：ImageNet 上 FM-OT 在 NLL 和 FID 上均优于扩散方法，且收敛更快的训练，所需 NFE（函数评估次数）更少。

## 与现有 Wiki 的关系

- 核心概念：[[Wiki/Concepts/Flow Matching]]
- 扩散路径作为特例：[[Wiki/Concepts/扩散模型原理]]
- OT 路径在 Seedream 3.0/4.0 中被沿用：[[Wiki/Entities/Seedream 系列模型]]
- FLUX.1 系列使用 rectified flow（FM 的变体）：[[Wiki/Sources/FLUX.1 Kontext]]

## 后续问题

- FM 泛化了扩散模型，但需要 ODE solver 采样，后续 rectified flow 引入直线路径加速。

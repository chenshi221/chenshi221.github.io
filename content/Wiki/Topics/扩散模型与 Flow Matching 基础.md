---
type: topic
status: active
created: "2026-04-30"
updated: "2026-04-30"
tags: ["diffusion", "flow-matching", "generative-model", "DDPM", "rectified-flow", "DiT", "Seedream", "FLUX"]
sources: ["[[Wiki/Sources/DDPM 扩散模型奠基论文]]", "[[Wiki/Sources/Flow Matching 生成建模]]", "[[Wiki/Sources/FLUX.1 Kontext 上下文编辑]]", "[[Wiki/Sources/Seedream 2.0 中英双语图像生成]]", "[[Wiki/Sources/Seedream 3.0 技术报告]]", "[[Wiki/Sources/Seedream 4.0 多模态图像生成]]"]
confidence: high
---

# 扩散模型与 Flow Matching 基础

## 概述

本主题串联从 DDPM (2020) 到 Seedream 4.0 (2025) 的**生成模型基础脉络**，聚焦于生成机制本身的演进：马尔可夫链扩散 -> 连续归一化流 -> Flow Matching -> Rectified Flow -> 工业级图像生成模型。区别于 [[Wiki/Topics/扩散模型图像编辑与生成]]（聚焦于编辑方法），本主题关注的是生成模型的"发动机"如何升级。

## 发展脉络

### 阶段一：DDPM 奠基 (2020)

- [[Wiki/Sources/DDPM 扩散模型奠基论文|DDPM (Ho et al., 2020)]] 证明了扩散模型可以生成高质量图像。
- 核心思想：固定前向噪声过程 + 学得的反向去噪过程。
- 简化目标 $L_{\text{simple}} = \mathbb{E}[\|\epsilon - \epsilon_\theta(\mathbf{x}_t, t)\|^2]$ 等价于多噪声水平的去噪分数匹配。
- 问题：采样需 1000 步，速度极慢。

### 阶段二：加速与架构升级 (2021-2023)

- **DDIM (2021)**：去随机化，通过非马尔可夫采样实现 50 步生成。
- **LDM / Stable Diffusion (2022)**：将扩散过程移到 VAE 的 latent space，大幅降低计算量。
- **DiT (2023)**：用 Transformer 替代 U-Net 作为去噪网络主干，证明 Transformer 在扩散模型中可扩展。

### 阶段三：Flow Matching 统一框架 (2023)

- [[Wiki/Sources/Flow Matching 生成建模|Flow Matching (Lipman et al., 2023)]] 用连续归一化流（CNF）框架统一了扩散模型。
- 关键洞察：不必从 SDE 推导概率路径，可直接设计 $\mu_t(x_1)$ 和 $\sigma_t(x_1)$。
- **OT 路径**：$\mu_t = tx_1$，直线轨迹，训练和采样都更高效。扩散路径（VP/VE）作为 FM 的特例。
- FM 在向量场回归上训练比分数匹配更稳定。

### 阶段四：Rectified Flow 与工业落地 (2024)

- **Rectified Flow (Liu et al., 2022)**：$z_t = (1-t)x + t\varepsilon$，直线噪声-数据插值路径，是 FM-OT 在 $\sigma=1$ 恒定时的情况。
- **Stable Diffusion 3 (Esser et al., 2024)**：MMDiT + Rectified Flow，将 FM 与 DiT 架构结合推出工业级模型。
- **FLUX.1 (Black Forest Labs, 2024)**：Rectified Flow Transformer，双流/单流混合 blocks，3D RoPE，latent space 16 通道 VAE。

### 阶段五：工业级模型的繁荣 (2025)

- [[Wiki/Sources/FLUX.1 Kontext 上下文编辑|FLUX.1 Kontext]]：在 FLUX.1 基础上扩展上下文图像编辑，序列拼接实现统一生成与编辑。
- [[Wiki/Entities/Seedream 系列模型|Seedream 系列]]：
  - **2.0**：中英双语 LLM 编码 + Glyph-Aligned ByT5 文本渲染。
  - **3.0**：缺陷感知训练、跨模态 RoPE、REPA、VLM 奖励模型缩放、4-8 倍加速。
  - **4.0**：多模态统一（T2I+编辑+多图），重新设计 DiT + 高压缩 VAE，10 倍加速，1.4s @ 2K。

## 生成机制演进图谱

```
DDPM (2020)                  马尔可夫链扩散, ε-预测
  |
  ├─ DDIM (2021)             确定性采样
  ├─ LDM (2022)              Latent space
  └─ DiT (2023)              Transformer backbone
       |
       └─→ Flow Matching (2023)    向量场回归, OT 路径
            |
            └─ Rectified Flow (2024) 直线路径, 工业架构 (MMDiT)
                 |
                 ├─ SD3 (2024)      Rectified Flow MMDiT
                 ├─ FLUX.1 (2024)   Rectified Flow Transformer
                 │    └─ FLUX.1 Kontext (2025)   生成+编辑统一
                 └─ Seedream (2025)
                      ├─ 2.0: 双语 T2I
                      ├─ 3.0: 全面升级+工业部署
                      └─ 4.0: 多模态统一 (T2I+编辑+多图)
```

## 核心公式对比

| 方法 | 路径定义 | 训练目标 |
|------|---------|---------|
| DDPM | $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ | $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2$ |
| FM (VP) | $x_t = \alpha_{1-t}x_1 + \sqrt{1-\alpha_{1-t}^2}x_0$ | $\|v_\theta(x_t,t) - u_t(x_t|x_1)\|^2$ |
| FM (OT) | $x_t = (1-(1-\sigma_{\min})t)x_0 + tx_1$ | $\|v_\theta(x_t,t) - (x_1 - (1-\sigma_{\min})x_0)\|^2$ |
| Rectified Flow | $z_t = (1-t)x + t\varepsilon$ | $\|v_\theta(z_t,t) + x - \varepsilon\|^2$ |

## 加速技术演进

| 方法 | 核心思路 | 来源 |
|------|---------|------|
| DDIM | 非马尔可夫确定性采样 | Song et al. 2021 |
| DPM-Solver | 高阶 ODE solver | Lu et al. 2022 |
| LADD | 对抗蒸馏 | Sauer et al. 2024 |
| Hyper-SD | 轨迹分段一致性 | Seedream 3.0/4.0 |
| RayFlow | 实例自适应流轨迹 | Seedream 3.0/4.0 |
| 一致噪声期望 | 统一参照稳定采样 | Seedream 3.0 |
| ADP + ADM | 对抗分布匹配 | Seedream 4.0 |
| 推测解码 | LLM 蒸馏加速 PE | Seedream 4.0 |

## 与图像编辑的关系

本主题阐述的是生成模型的"底层引擎"。编辑方法（InstructPix2Pix、ControlNet、FLUX.1 Kontext 等）是在这些引擎之上构建的应用层。FLUX.1 Kontext 通过序列拼接将编辑能力融入 flow matching 引擎本身，Seedream 4.0 则更进一步将多模态编辑内化为模型原生能力。

详见：[[Wiki/Topics/扩散模型图像编辑与生成]]

## 关键开放问题

1. **OT 路径 vs Rectified Flow**：两者都是直线路径，OT 路径包含方差收缩（$\sigma_t$ 从 1 降至 $\sigma_{\min}$），rectified flow 保持方差恒定（$\sigma_t=1$）。哪种路径在什么场景下更优？
2. **加速的极限**：1-step 生成质量何时能匹敌多步？Adversarial distillation 是否引入不可控的 artifacts？
3. **统一模型的架构设计**：Seedream 4.0 和 FLUX.1 Kontext 都走向了"统一生成与编辑"。下一个统一的维度是什么？视频？3D？多模态理解？
4. **中文与多语言**：Seedream 2.0 证明了 LLM 编码器对中文的重要性，但对于其他非英语语言，编码器选择是否同样关键？

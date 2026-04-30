---
title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
authors:
  - DeepSeek-AI
institutions: DeepSeek-AI
aliases:
  - DeepSeek-V2
  - DeepSeekV2
date: '2026-04-30'
publish_date: 2024-05
venue: 'arXiv:2405.04434'
tags:
  - 论文
  - LLM
  - MoE
  - MLA
  - 高效推理
  - KV Cache
url: 'https://arxiv.org/abs/2405.04434'
code: https://github.com/deepseek-ai/DeepSeek-V2
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：DeepSeek-V2 提出了两项关键架构创新——MLA（Multi-head Latent Attention）大幅压缩 KV Cache 实现高效推理，DeepSeekMoE 通过稀疏计算实现经济训练——在 236B 总参数（21B 激活）下，相比 DeepSeek 67B 节省 42.5% 训练成本、KV Cache 减少 93.3%、生成吞吐提升 5.76 倍。

![](https://arxiv.org/html/2405.04434v5/x1.png)

*Figure 1: MMLU 性能与激活参数对比，以及 DeepSeek 67B 与 DeepSeek-V2 的训练成本和推理效率对比。*

---

## Intro

### Motivation

LLM 的智能随参数规模增长而提升，但代价是更大的训练计算资源和更低的推理吞吐。MoE 架构（如 GShard）可部分缓解此问题，但传统 MoE 在专家专业化和推理效率上仍有不足。DeepSeek-V2 旨在同时实现强性能、经济训练和高效推理。

### 贡献

1. **MLA（Multi-head Latent Attention）**：低秩 KV 联合压缩，性能优于 MHA 同时 KV Cache 大幅降低
2. **DeepSeekMoE**：细粒度专家分割 + 共享专家隔离，训练成本极低
3. **经济高效**：236B 总参数/21B 激活，8.1T tokens 预训练，训练成本仅为 DeepSeek 67B 的 57.5%

---

## Method 核心方法

![](https://arxiv.org/html/2405.04434/x2.png)

*Figure 2: DeepSeek-V2 架构总览。MLA（Multi-head Latent Attention）通过低秩 KV 压缩减少 KV Cache，DeepSeekMoE 通过稀疏专家实现经济训练。*

### 1. MLA（Multi-head Latent Attention）

**核心思想**：将 keys 和 values 联合压缩到一个低维 latent vector $\mathbf{c}_t^{KV}$ 中。

- 压缩维度 $d_c \ll d_h n_h$（实际设置为 $4d_h$）
- 推理时只需缓存 $\mathbf{c}_t^{KV}$，KV Cache 从 $2n_h d_h l$ 降至 $(d_c + d_h^R)l \approx \frac{9}{2}d_h l$
- 等价于只有 2.25 个 group 的 GQA，但性能优于 MHA

**解耦 RoPE**：为解决 RoPE 与低秩 KV 压缩不兼容的问题，额外引入解耦的 queries $\mathbf{q}_t^R$ 和 shared key $\mathbf{k}_t^R$ 来携带位置信息，保证推理时 $W^{UK}$ 可被吸收到 $W^Q$ 中。

**Query 压缩**：进一步对 queries 做低秩压缩以减少训练激活内存。

注意力机制 | KV Cache per Token | 能力
---|---|---
MHA | $2n_h d_h l$ | 强
GQA | $2n_g d_h l$ | 中等
MQA | $2d_h l$ | 弱
**MLA** | $\approx \frac{9}{2}d_h l$ | **更强**

![](https://arxiv.org/html/2405.04434/x3.png)

*Figure 3: MHA、GQA、MQA、MLA 四种注意力机制对比。MLA 通过将 KV 联合压缩到低维 latent vector，在接近 MQA 的 KV Cache 成本下实现优于 MHA 的性能。*

### 2. DeepSeekMoE

两个关键设计：
- **细粒度专家分割**：比传统 MoE 更细粒度的专家，提升专业化程度
- **共享专家隔离**：一部分专家始终激活（shared experts），减少路由专家之间的知识冗余

输出计算：$\mathbf{h}_t' = \mathbf{u}_t + \sum_{i=1}^{N_s} \text{FFN}^{(s)}_i(\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t} \text{FFN}^{(r)}_i(\mathbf{u}_t)$

**Device-Limited Routing**：限制每个 token 的目标专家最多分布在 M 个设备上，控制 MoE 通信成本。当 M >= 3 时，性能接近不受限的 top-K 路由。

**三级负载均衡辅助损失**：Expert-Level Balance Loss + Device-Level Balance Loss + Communication Balance Loss。

### 3. 训练和对齐

- 预训练：8.1T tokens 高质量多源语料
- SFT：150 万对话 session，涵盖数学、代码、写作、推理、安全等
- RL：GRPO（Group Relative Policy Optimization）算法对齐人类偏好

---

## 实验/评估/结果

- **MMLU**：DeepSeek-V2 在最少激活参数下实现顶级开源模型性能
- **AlpacaEval 2.0**：38.9 length-controlled win rate
- **MT-Bench**：8.97
- **AlignBench（中文）**：7.91，超过所有开源模型和大多数闭源模型
- **训练效率**：比 DeepSeek 67B 节省 42.5% 成本
- **推理效率**：KV Cache 减少 93.3%，最大生成吞吐提升至 5.76 倍

---

## 结论

DeepSeek-V2 通过 MLA 和 DeepSeekMoE 两项架构创新，在性能、训练成本和推理效率三个维度上同时取得突破。MLA 解决了 KV Cache 瓶颈，DeepSeekMoE 解决了训练成本问题。该架构成为后续 DeepSeek-V3 和 DeepSeek-R1 的基础。

---

## 思考

### 优点

1. **MLA 是极具原创性的注意力机制设计**：低秩 KV 联合压缩 + 解耦 RoPE 的方案精巧且实用，平衡了 KV Cache 大小和模型性能
2. **推理效率的提升幅度令人印象深刻**：KV Cache 减少 93.3%、吞吐提升 5.76 倍
3. **工程完整性**：不仅设计了算法，还考虑了设备级路由限制、三级负载均衡损失

### 缺点

1. **MLA 的实现复杂度高**：解耦 RoPE 和吸收矩阵的数学变换增加了实现和理解的难度
2. **负载均衡仍依赖辅助损失**：虽设计了精细的三级损失，但辅助损失本身可能损害模型性能（V3 中改为 auxiliary-loss-free）
3. **与 Dense 架构的全面性能对比不足**：主要比较对象是自家 DeepSeek 67B，缺少与同等规模 dense 模型的公平对比
4. **训练细节披露有限**：未给出具体的 GPU 小时数和能耗数据

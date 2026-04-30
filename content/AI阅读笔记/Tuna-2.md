---
title: 'Tuna-2: Pixel Embeddings Beat Vision Encoders for Multimodal Understanding and Generation'
authors:
  - Zhiheng Liu
  - Weiming Ren
  - Xiaoke Huang
  - Shoufa Chen
  - Tianhong Li
  - Mengzhao Chen
  - Yatai Ji
  - Sen He
  - Jonas Schult
  - Belinda Zeng
  - Tao Xiang
  - Wenhu Chen
  - Ping Luo
  - Luke Zettlemoyer
  - Yuren Cong
institutions: 'Meta AI; The University of Hong Kong; University of Waterloo'
aliases:
  - Tuna-2
date: '2026-04-30'
publish_date: 2026-04
venue: arXiv:2604.24763
tags:
  - 论文
  - 多模态
  - 统一模型
  - Encoder-Free
  - Flow Matching
  - 像素空间
  - MAE
url: 'https://arxiv.org/abs/2604.24763'
code: 已开源
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：Tuna-2 提出了一个彻底 Encoder-Free 的原生统一多模态模型——不使用 VAE、不使用 CLIP/SigLIP 等 pretrained representation encoder，仅用简单的 patch embedding 层将图像映射为 pixel embeddings，配合 masking-based feature learning 方案，在 550M 图文对上实现 pixel-space flow matching（x-prediction + v-loss），在多模态理解和生成上达到 SOTA，尤其在细粒度视觉感知任务上超越 encoder-based 方案。

![](https://arxiv.org/html/2604.24763/x1.png)

*Figure 1: Tuna-2 架构演进——从 Tuna 出发逐步移除 VAE（得到 Tuna-R），再移除 representation encoder（得到 Tuna-2），证明 encoder-free 设计在充分训练后可以超越 encoder-based 方案。*

---

## Intro

### Motivation

现有统一多模态模型（UMM）严重依赖预训练视觉编码器（VAE + CLIP/SigLIP）。但近期两个趋势暗示这可能不是必需的：(1) 理解端：encoder-free VL 模型（如 Mono-InternVL、NEO）直接用 patch embedding 替代 CLIP；(2) 生成端：pixel-space 扩散模型（如 JiT）证明 VAE 不是高保真图像生成的必需品。Tuna-2 提出了一个自然但未被充分探索的问题：能否完全移除预训练视觉编码器，直接从原始像素进行端到端统一多模态建模？

### 贡献

1. 提出 Tuna-2，首个完全 Encoder-Free 的原生统一多模态模型，仅用 patch embedding + LLM decoder
2. Tuna-R（中间设计，保留 SigLIP encoder 但移除 VAE）作为受控对照
3. Masking-based feature learning 方案：理解端做正则化，生成端做 harder denoising
4. 证明 encoder-free 设计在充分预训练后：生成上与 encoder-based 持平，理解上全面超越（尤其在细粒度感知任务）
5. 训练仅需两阶段（Stage 1 预训练 + Stage 2 SFT），无需 connector alignment 阶段

---

## Method 核心方法

### 1. 架构简化

**Tuna → Tuna-R → Tuna-2**：
- Tuna-R：移除 VAE，保留 SigLIP representation encoder
- Tuna-2：进一步移除 representation encoder，用简单 patch embedding 层直接编码图像

**Tuna-2 架构**：
- Patch embedding：图像 → 视觉 token
- Qwen2.5-7B-Instruct：视觉 token + 文本 token 联合处理
- Language head：自回归文本预测
- Flow head：pixel-space flow matching

### 2. Pixel-Space Flow Matching

- 采用 x-prediction + v-loss 范式（来自 JiT）
- Rectified flow + linear schedule：$x_t = t x_1 + (1-t) x_0$
- 模型预测 clean image $x_\theta$，转换为 velocity $v_\theta$ 后回归
- 推理用 Euler solver
- 直接在像素空间操作，无需 VAE 压缩/解压

### 3. Masking-based Feature Learning

这是 Tuna-2 最关键的训练创新：

- 随机 mask 部分图像 patch（替换为 learnable mask token）
- **生成端**：预测 masked 和 unmasked 区域的 clean image → harder denoising + 鼓励 mask token 学习上下文信息
- **理解端**：基于 masked 视觉输入预测文本 → 正则化，强制模型在部分视觉条件下推理
- 类似于 MAE（理解）+ MaskGIT（生成）的融合

![](https://arxiv.org/html/2604.24763/x3.png)

*Figure 3: Masking-based Feature Learning 方案——训练时使用可学习 mask token 同时服务理解和生成：理解端做正则化（基于部分视觉信息推理文本），生成端做 masked prediction（预测被 mask 区域的 clean image），相互增强。*

### 4. 训练策略

**Stage 1（全模型预训练）**：
- 550M 图文对（70% image captioning + 30% T2I）
- 20% 混入纯文本数据（Nemotron）
- 300K steps，64 nodes，lr=1e-4

**Stage 2（SFT）**：
- 13M 图像指令遵循（FineVision）+ 2M 图像编辑（OmniEdit）+ 高质量生成数据
- 50K steps，lr=2e-5

无需 Tuna-R 所需的额外 connector alignment 阶段

---

## 实验/评估/结果

### 多模态理解

- GQA 65.0 / MMVet 51.7 / MMMU 50.7 / OCRBench 79.7
- 在 CountBench 81.7 / MMVP 77.3 等细粒度感知任务上大幅超越 Tuna-R 和所有 UMM 基线
- Tuna-2 vs Tuna-R：理解任务全面领先，尤其在像素级感知（V* 59.2 vs 57.6）

### 图像生成

- GenEval：可与 encoder-based 方案竞争
- DPG-Bench：优秀

### 核心发现

- **训练早期**：encoder-based（Tuna-R）收敛更快（SigLIP 提供好的初始化）
- **训练后期**：encoder-free（Tuna-2）逐渐超越，尤其在细粒度理解上
- **原因**：pretrained encoder 有固定的分辨率限制和归纳偏置，限制了视觉细节的捕获；encoder-free 设计允许端到端学习更适合任务的视觉表征

---

## 结论

Tuna-2 证明了：(1) 预训练视觉编码器不是统一多模态建模的必需品；(2) end-to-end pixel-space 学习在充分训练后可以超越 encoder-based 方案；(3) masking-based feature learning 是 pixel-space UMM 稳定训练的关键技术。这是对"vision encoder 不可或缺"这一普遍认知的重要挑战。

---

## 思考

### 优点

1. **方向性突破**：Tuna-2 挑战了"必须用 pretrained vision encoder"的共识。如果 encoder-free 真的能 scale，将大大简化模型架构、降低对预训练编码器的依赖，对开源社区意义重大。

2. **受控实验设计**：Tuna-R 作为中间对照，排除了 pixel-space vs latent-space 的混淆因素，清晰分离出"encoder-free vs encoder-based"的因果效应。

3. **Masking 方案的精妙**：同时服务于理解和生成的 masking 设计，巧妙地融合了 MAE 和 MaskGIT 的思路，且不增加推理成本。

4. **训练效率**：两阶段训练（无 connector alignment），比需要多阶段对齐的 encoder-based 方案更简洁。

### 缺点与待解决问题

1. **收敛速度的 trade-off**：encoder-free 需要更多训练才能超越 encoder-based。在实际资源受限场景下，这可能是一个显著的劣势。

2. **规模验证不足**：目前仅在 7B 规模验证。更大规模的 scaling behavior（如 70B 级别）未知。encoder-free 的优势是否会持续放大，还是出现性能饱和？

3. **Pixel-space 的内存开销**：直接在像素空间做 flow matching 比 latent space 需要更多显存，限制了可处理的分辨率。

4. **生成质量的天花板**：pixel-space 扩散模型（如 JiT）虽然在 scaling 上表现出色，但目前 SOTA 生成模型仍主要基于 latent diffusion。Tuna-2 的生成能力上限还有待验证。

### 与已有 Wiki 的连接

- 关联概念：[[原生多模态模型]]、[[Encoder-Free]]、[[Flow Matching]]、[[MAE]]
- 关联实体：[[JiT]]、[[Qwen2.5]]、[[SigLIP 2]]
- 关联比较：与 [[BAGEL]]（解耦 encoder）和 [[Show-o2]]（统一 encoder）的架构对比，与 [[UniWorld-V1]] 的 encoder 选择对比

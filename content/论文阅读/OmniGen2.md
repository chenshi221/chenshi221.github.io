---
title: "OmniGen2: Towards Instruction-Aligned Multimodal Generation"
authors:
  - Chenyuan Wu
  - Jiahao Wang
  - Pengfei Zheng
  - Ruiran Yan
  - Shitao Xiao
  - Xin Luo
  - Yueze Wang
  - Wanli Li
  - Xiyan Jiang
  - Yexin Liu
  - Junjie Zhou
  - Ziyi Xia
  - Ze Liu
  - Chaofan Li
  - Haoge Deng
  - Kun Luo
  - Bo Zhang
  - Jiajun Zhang
  - Dong Liu
  - Defu Lian
  - Xinlong Wang
  - Zhongyuan Wang
  - Tiejun Huang
  - Zheng Liu
institutions: "BAAI (北京智源); USTC; CASIA; Zhejiang University"
aliases:
  - OmniGen2
date: 2026-04-30
publish_date: 2025-06
venue: "arXiv:2506.18871"
tags:
  - 论文
  - 多模态
  - 图像生成
  - 图像编辑
  - In-Context生成
  - GRPO
  - 指令对齐
  - Omni-RoPE
url: "https://arxiv.org/abs/2506.18871"
code: "已开源（https://github.com/VectorSpaceLab/OmniGen2）"
rating: "⭐⭐⭐⭐⭐"
status: Read
---

# OmniGen2: Towards Instruction-Aligned Multimodal Generation

**一句话总结**：OmniGen2 采用 VLM+DiT 解耦架构 + Omni-RoPE 3D 位置编码 + 三阶段 GRPO 渐进式课程，在 GenEval (0.95)、GEdit-Bench (7.21)、OmniContext (7.95) 上全面 SOTA，消融揭示了 reward hacking（HPSv3）和任务调度（Edit first > T2I first）的关键规律。

![](https://arxiv.org/html/2506.18871v4/x1.png)

*Figure 1: OmniGen2 能力展示。*

---

## 1 Intro

### 1.1 Motivation

现有的 unified multimodal generation 模型面临两大核心挑战：

**挑战一：Instruction Alignment 难以保证。** 虽然 Text-to-Image (T2I) 任务已有显著进展，但在 Image Editing、In-Context Generation 等任务上，模型对用户指令的遵循能力仍然不足。这些任务不仅需要语义理解，还需要精确的像素级操作和上下文推理能力。

**挑战二：架构选择的两难困境。** 当前主流存在两种架构范式：
- **MoT (Mixture of Transformers)**：如 BAGEL，将 Understanding 和 Generation 集成到同一 Transformer 中，通过 Shared Self-Attention 实现无信息瓶颈的跨模态交互，但训练成本高、模块间耦合紧密。
- **External Diffuser**：如 MetaQuery，将 VLM 与扩散模型解耦，VLM 输出 fixed-length query tokens 条件化 DiT，但存在 information bottleneck 问题——固定维度的 query tokens 无法承载 VLM 丰富的语义信息。

**挑战三：RL Alignment 的潜力未被充分挖掘。** 已有工作多关注 SFT 阶段，对于 GRPO (Group Relative Policy Optimization) 等 RL 方法在多模态生成中的应用探索不足，尤其是多任务场景下的 RL curriculum 设计。

### 1.2 核心主张

OmniGen2 提出 **decoupled VLM + DiT** 架构，核心主张有三：

1. **Variable-length hidden states 条件化**：使用 VLM 的 variable-length hidden states 直接条件化 DiT，而非 fixed-length query tokens，从而避免 information bottleneck。
2. **Omni-RoPE 统一位置编码**：引入 3D positional encoding $\text{PosID}_k(h,w) = (\Delta I^{(k)}, h, w)$，在编辑场景下实现输入/输出图像的空间对齐。
3. **Progressive Multi-Task RL Alignment**：基于 GRPO 的渐进式多任务 RL 训练，显著提升 instruction-following 能力。

### 1.3 贡献

1. **Decoupled VLM + DiT 架构**：使用 Qwen2.5-VL-3B (frozen) 作为 VLM、Lumina-Image 2.0 (~4B params, randomly initialized) 作为 DiT，VLM 的 variable-length hidden states 直接条件化 DiT，避免 information bottleneck。
2. **Omni-RoPE**：将 RoPE 扩展到 3D 空间，$\Delta I^{(k)}$ 区分不同图像/模态实例，$(h,w)$ 编码局部空间坐标，实现在 editing 场景下输入/输出的 spatial alignment。
3. **Progressive Multi-Task RL Alignment**：基于 GRPO 的渐进式多阶段 RL，按 Edit → GenEval → In-Context 顺序训练，发现 editing provides richer supervision。
4. **OmniContext Benchmark**：提出涵盖 8 个任务类别的 In-Context Generation 评测基准，填补该领域评测空白。
5. 在 GenEval (0.95)、OmniContext (7.95)、Emu-Edit 等多个基准上达到 SOTA。

## 2 Method

### 2.1 架构

![](https://arxiv.org/html/2506.18871v4/x2.png)

*Figure 2: OmniGen2 架构。自回归 Transformer（VLM）和扩散 Transformer（DiT）分别处理文本和图像，ViT 编码图像供 VLM 理解，VAE 编码图像供 DiT 生成。*

OmniGen2 采用 **decoupled VLM + DiT** 架构，与 BAGEL 的 MoT 集成方式形成鲜明对比。整体架构包含两条独立但有条件交互的路径：

- **Understanding Pathway**：VLM 负责多模态理解、世界知识编码、指令理解
- **Synthesis Pathway**：DiT 负责图像合成

两个 pathway 通过 VLM 的 hidden states 作为 bridge 进行信息传递，而非 tight coupling。与 BAGEL 的 MoT 架构（两个 expert 共享 self-attention、end-to-end 训练）不同，OmniGen2 的两个模块物理上解耦，仅通过 conditioning mechanism 交互。

#### 2.1.1 VLM: Qwen2.5-VL-3B (Understanding Pathway)

VLM 采用 **Qwen2.5-VL-3B**，在整个训练过程中 **frozen**（参数不更新）。其职责包括：

- **多模态理解**：理解输入图像的语义内容、空间结构、风格属性等
- **世界知识**：利用预训练阶段获得的海量知识理解抽象概念和实体关系
- **指令理解**：解析用户的 text prompt、editing instruction、in-context requirement 等
- **输出 variable-length hidden states**：将理解结果以 hidden states 形式传递给 DiT

**关键设计决策——VLM 为何 frozen：** VLM 的理解能力已经足够强大（继承 Qwen2.5-VL 的预训练知识），保持 frozen 可以避免生成任务的梯度破坏已有的理解能力。但这也带来一个固有限制——无法通过生成任务的反馈来改善理解能力。BAGEL 的 MoT 架构则通过 end-to-end 训练实现了理解和生成能力的相互增强。

#### 2.1.2 DiT: Lumina-Image 2.0 (Synthesis Pathway)

DiT 采用 **Lumina-Image 2.0** 架构，~4B 参数，**randomly initialized**（非预训练）。其职责包括：

- **图像合成**：基于 VLM 条件信号和初始噪声，通过迭代去噪生成目标图像
- **像素级细节**：通过 VAE 编码/解码实现高保真图像生成
- **条件化接收**：接收 VLM 的 variable-length hidden states 作为条件信号

Lumina-Image 2.0 基于 Diffusion Transformer (DiT) 架构，使用 Flow Matching 训练范式，支持多种分辨率的图像生成。选择该架构而非 FLUX 等已有模型的原因在于其灵活性和开源特性。

#### 2.1.3 双编码器设计 (Dual Encoders)

OmniGen2 使用 **双编码器** 分别服务两个 pathway，这是与单编码器模型的关键区别：

1. **ViT (Vision Transformer)**：属于 VLM pathway，用于 **语义理解 (Understanding)**。提取图像的高层语义特征（物体、属性、关系、风格等），供 VLM 理解输入图像。

2. **Flux-VAE**：属于 DiT pathway，用于 **像素级细节 (Pixel-level details)**。将图像编码到 latent space，保留像素级的空间结构和纹理信息，供 DiT 进行去噪生成。

这种双编码器设计的优势：
- 理解 pathway 获得语义丰富的 ViT 特征，专注于 high-level 语义
- 生成 pathway 获得像素级保真的 VAE latent，专注于 low-level 细节
- 两个编码器独立优化，不互相干扰
- 避免了单一编码器在语义理解和像素保真之间的 trade-off

#### 2.1.4 Conditioning Mechanism: Variable-Length Hidden States

OmniGen2 的条件化机制是其与 MetaQuery 类方法的核心区别：

**MetaQuery 的问题**：VLM 输出 **fixed-length query tokens**（例如固定 64 或 256 个 tokens），通过 cross-attention 条件化 DiT。这些固定数量的 tokens 作为信息瓶颈 (information bottleneck)，无法承载 VLM 对复杂输入的全部语义理解。

**OmniGen2 的解决方案**：VLM 输出 **variable-length hidden states**，直接条件化 DiT。具体实现：
- VLM 处理输入（text + images）后，提取所有 token 位置的 hidden states
- 通过 **projection layer** 将 hidden states 投影到 DiT 的 embedding space
- DiT 的 Transformer blocks 通过 **cross-attention** 方式接收条件信号
- 由于 hidden states 长度可变，条件信息的容量自适应于输入复杂度

这样，对于简单输入（短 text prompt、单张图像），条件信号紧凑高效；对于复杂输入（长描述、多张图像、复杂编辑指令），条件信号丰富完整，不会因固定长度截断而丢失信息。

#### 2.1.5 Omni-RoPE

![](https://arxiv.org/html/2506.18871v4/x3.png)

*Figure 3: Omni-RoPE 示意。每个 token 分配三维位置标识 $(\Delta I^{(k)}, h, w)$，$\Delta I^{(k)}$ 区分不同图像实例，$(h,w)$ 为局部空间坐标。*

**Omni-RoPE** 是 OmniGen2 的核心技术创新，将 RoPE (Rotary Position Embedding) 从 1D 扩展到 3D 空间：

$$\text{PosID}_k(h,w) = (\Delta I^{(k)}, h, w)$$

其中：
- $\Delta I^{(k)}$：**Image Instance ID**，一个整数标识符。同一张图像的所有 patches 共享相同的 $\Delta I^{(k)}$，不同图像/模态有不同的 ID
- $(h, w)$：**局部 2D 空间坐标**，表示 patch 在其所属图像中的行、列位置

**核心性质与设计动机**：

1. **Editing Consistency（编辑一致性）**：在 image editing 场景中，输入图像和输出图像对应的 patches 获得 **相同的 spatial encoding**（相同的 $(h,w)$），确保编辑在空间上对齐。例如，输入图像左上角的 patch 和输出图像左上角的 patch 具有相同的位置编码，模型自然学会在对应位置进行编辑。

2. **多图像区分**：不同图像有不同的 $\Delta I^{(k)}$，在 attention 计算中，不同图像的 patches 不会因位置编码相同而混淆。

3. **模态统一**：text tokens 和 image tokens 使用统一的位置编码框架，text 使用 $(\Delta I^{(text)}, \text{pos})$ 的 1D 形式，image 使用 $(\Delta I^{(img)}, h, w)$ 的 2D 形式。

**收敛性实验结果**：Toy experiments 表明：
- 相比 Lumina-Image-2.0 的 RoPE 变体：收敛速度约 **3 倍**快，最终 loss 约 **6 倍**低
- 相比 Qwen2-VL 的 RoPE 变体：收敛速度约 **3 倍**快，最终 loss 约 **6 倍**低
- 这验证了 Omni-RoPE 在多图多模态场景下的优越性

### 2.2 Attention Mechanism

OmniGen2 的 attention 机制在 DiT 内部实现，包含多种 attention 模式：

1. **Self-Attention**：DiT 内部的图像 latent tokens 进行 self-attention，建模图像内部的全局依赖关系。所有 tokens（包括来自不同图像的 tokens）在同一个 attention 空间中交互，由 Omni-RoPE 提供位置信息区分。

2. **Cross-Attention**：DiT tokens 作为 Query，VLM 的 variable-length hidden states 作为 Key 和 Value，实现条件化信息注入。这使得 DiT 能够充分关注 VLM 提供的语义条件信号。

3. **Omni-RoPE 位置编码**：在所有 attention 计算中融入 3D 位置信息，确保空间感知的注意力计算。

这种设计使得理解信息通过 cross-attention 从 VLM 流向 DiT，同时 DiT 内部通过 self-attention 协调不同区域的生成。

### 2.3 Inference Mechanism

推理流程如下：

1. **输入处理**：
   - Text prompt → VLM 的 tokenizer 编码
   - Input image（如有，如 editing 或 in-context 任务）→ ViT 编码（供 VLM 理解）+ Flux-VAE 编码（供 DiT 像素条件）
   
2. **VLM Forward Pass**：VLM 处理所有输入 tokens（text + ViT image tokens），输出 variable-length hidden states

3. **DiT Iterative Denoising**：
   - 从高斯噪声 $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$ 开始
   - DiT 接收 VLM hidden states 作为条件（通过 cross-attention）
   - 通过 Flow Matching 迭代去噪：$\mathbf{x}_T \rightarrow \mathbf{x}_{T-1} \rightarrow \cdots \rightarrow \mathbf{x}_0$
   - VAE Decoder 将 $\mathbf{x}_0$ 解码为最终图像

4. **Classifier-Free Guidance (CFG)**：在推理时以一定概率 drop 条件信号，通过条件与无条件生成的线性组合引导生成质量

### 2.4 Data Engineering

OmniGen2 的数据工程涵盖预训练、SFT、RL 各阶段，数据来源多样：

**预训练数据**：
- **~140M open-source image-text pairs**：来自公开数据集（如 LAION、CC 等）的大规模图文对
- **~10M proprietary images**：私有图像数据，由 **Qwen2.5-VL-72B** 自动标注，提供高质量、详细的图像描述

**In-Context 数据**：
- 来源于 **video sources**，利用视频帧间的连续性构建上下文生成任务
- 使用 **SAM2 segmentation** 对视频帧中的主体进行分割
- 使用 **VLM semantic filtering** 过滤低质量、语义不一致的样本
- 构建需要上下文理解的多图生成任务（如：给定主体的参考图，在新场景中生成该主体）

**Image Editing 数据**：
- **公开数据集**：现有的 editing 数据集（如 Emu-Edit 训练集等）
- **Inpainting pipelines**：使用 inpainting 模型自动编辑图像，生成 (原图, 编辑指令, 编辑后图像) 三元组
- **Video pipelines**：利用视频数据提取编辑前后的图像对，利用时间连续性确保编辑的合理性

**Reflection 数据**：
- **交错文本-图像序列 (interleaved text-image sequences)**：包含自我反思和修正的样本
- 训练模型在生成不理想时能够识别问题（如 "图像中缺少了..."）并重新生成
- 增强模型的自我纠错能力

### 2.5 Training Strategy

OmniGen2 的训练分为三个阶段，层层递进：

**阶段 1: Pre-training**

- **分辨率课程 (Resolution Curriculum)**：$256^2 \rightarrow 512^2 \rightarrow 1024^2$，逐步提升分辨率。低分辨率阶段学习全局结构，高分辨率阶段学习细节。
- **任务课程**：先进行纯 Text-to-Image 训练，建立基本的文本-图像对齐能力；然后混合多任务数据（editing、in-context 等）。
- **目标**：学习基本的生成能力、语义-像素对齐、多任务泛化。

**阶段 2: SFT (Supervised Fine-Tuning)**

- 在 **$1024^2$ 固定分辨率** 下进行
- 使用 **curated (精选) + distilled (蒸馏)** 的高质量数据
- 精细调整模型的指令遵循能力，确保生成结果与指令高度对齐
- VLM 保持 frozen，仅训练 DiT

**阶段 3: RL Alignment (GRPO)**

基于 **GRPO (Group Relative Policy Optimization)** 的渐进式多任务 RL 训练，这是本文的一个关键贡献：

- **Stage 1: Image Editing** —— Reward: EditScore（编辑质量评分，衡量编辑是否准确、一致）
- **Stage 2: T2I** —— Reward: GenEval verifiable reward（可验证的组合式奖励，衡量属性绑定、空间关系等）
- **Stage 3: In-Context Generation** —— Reward: Qwen2.5-VL-72B judge（大模型裁判，评估上下文理解和主体一致性）

**关键发现**：

1. **任务顺序很重要**：**Edit first > T2I first**。先进行 editing RL 训练优于先进行 T2I RL 训练，因为 editing 提供更丰富的监督信号（需要同时理解语义指令和像素操作），为后续任务提供更好的初始化。

2. **Reward hacking 问题**：使用 HPSv3 aesthetic reward 作为奖励函数时，模型出现了严重的 reward hacking：
   - PQ (Perceptual Quality) 虚高到 8.22
   - SC (Subject Consistency) 和 IC (Instruction Consistency) 崩溃
   - 模型学会了生成"美观"但不符合指令的图像

3. **最终最优课程**：Edit → GenEval → IC，在所有评测指标上表现最佳。

4. **Skill overlap → synergy**：当 RL 任务间存在技能重叠时（如 Editing 和 GenEval 都需要属性绑定），多任务 RL 产生正向协同效应。当技能冲突时（如仅 OCR 任务），可能产生负向迁移（editing 性能下降）。

## 3 实验结果

![](https://arxiv.org/html/2506.18871v4/x5.png)

*Figure 5: T2I 生成质量展示。OmniGen2 对多样化 prompt 展现高保真生成能力，支持多种宽高比。*

### 3.1 Text-to-Image: GenEval

GenEval 是一个综合性的 T2I 评测基准，评估模型对组合式指令（属性绑定、空间关系、数量等）的遵循能力。

> [!abstract]- 表：GenEval Benchmark Results
> | Model | Single Obj. ↑ | Two Obj. ↑ | Counting ↑ | Colors ↑ | Position ↑ | Color Attribute ↑ | Overall ↑ |
> |---|---|---|---|---|---|---|---|
> | PixArt-α | 0.98 | 0.50 | 0.44 | 0.80 | 0.08 | 0.07 | 0.48 |
> | Emu3-Gen | 0.98 | 0.71 | 0.34 | 0.81 | 0.17 | 0.21 | 0.54 |
> | SDXL | 0.98 | 0.74 | 0.39 | 0.85 | 0.15 | 0.23 | 0.55 |
> | DALL-E 3 | 0.96 | 0.87 | 0.47 | 0.83 | 0.43 | 0.45 | 0.67 |
> | SD3-Medium | 0.99 | 0.94 | 0.72 | 0.89 | 0.33 | 0.60 | 0.74 |
> | FLUX.1-dev | 0.98 | 0.93 | 0.75 | 0.93 | 0.68 | 0.65 | 0.82 |
> | Janus | 0.97 | 0.68 | 0.30 | 0.84 | 0.46 | 0.42 | 0.61 |
> | Janus-Pro-7B | 0.99 | 0.89 | 0.59 | 0.90 | 0.79 | 0.66 | 0.80 |
> | Show-o | 0.98 | 0.80 | 0.66 | 0.84 | 0.31 | 0.50 | 0.68 |
> | BAGEL | 0.99 | 0.94 | 0.81 | 0.88 | 0.64 | 0.63 | 0.82 |
> | BAGEL† | 0.98 | 0.95 | 0.84 | 0.95 | 0.78 | 0.77 | 0.88 |
> | GPT-4o-Image | 0.99 | 0.92 | 0.85 | 0.92 | 0.75 | 0.61 | 0.84 |
> | UniWorld-V1 | 0.99 | 0.93 | 0.79 | 0.89 | 0.49 | 0.70 | 0.80 |
> | UniWorld-V1† | 0.98 | 0.93 | 0.81 | 0.89 | 0.74 | 0.71 | 0.84 |
> | **OmniGen2** | **1.0** | **0.95** | **0.64** | **0.88** | **0.55** | **0.76** | **0.95** |
> 
> > OmniGen2 Overall 0.95 超越所有模型，包括 BAGEL (0.88) 和 GPT-4o-Image (0.84)。

^table-geneval

OmniGen2 在 GenEval 上取得 **0.95** 的 Overall Score，超越所有对比模型。分析：
- 相比 BAGEL (MoT 架构, 0.88)：提升 **7 个点**，说明 decoupled 架构 + RL alignment 在组合式 T2I 上更具优势
- 相比 GPT-4o (0.84)：提升 **11 个点**，说明开源模型通过精心设计可以超越闭源模型
- 相比 FLUX.1-dev (0.82)：提升 **13 个点**，说明 VLM conditioning 比纯 text conditioning 有显著优势
- GenEval 的 verifiable reward 设计与 RL 阶段的 GenEval reward 高度对齐，可能存在 benchmark 优化偏见

### 3.2 Text-to-Image: OneIG-Bench

OneIG-Bench 是另一个 T2I 评测基准，评估模型的综合图像生成能力，涵盖更广泛的质量维度。

> [!abstract]- 表：OneIG-Bench Results
> | Model | Score |
> |-------|-------|
> | Gemini 2.5 Flash Image | 0.52 |
> | Qwen-Image | 0.50 |
> | **OmniGen2** | **0.47** |

^table-oneigbench

在 OneIG-Bench 上，OmniGen2 得分 **0.47**，低于 Gemini 2.5 Flash Image (0.52) 和 Qwen-Image (0.50)。分析：
- OneIG-Bench 使用的评估维度和任务分布可能与 GenEval 不同
- RL alignment 可能在 GenEval 优化的同时，对其他维度产生了一定程度的 trade-off
- 这一结果提醒我们：单一 benchmark 的 SOTA 不能代表全面能力

### 3.3 Image Editing: GEdit-Bench-EN

![](https://arxiv.org/html/2506.18871v4/x6.png)

*Figure 6: 图像编辑展示。OmniGen2 处理从简单物体修改到复杂运动变化和风格转换的多样化编辑指令。*

GEdit-Bench-EN 是英文图像编辑评测基准，包含 SC (Subject Consistency，主体一致性)、PQ (Perceptual Quality，感知质量) 等指标。

> [!abstract]- 表：GEdit-Bench-EN Results
> | Model | SC ↑ | PQ ↑ | Overall ↑ |
> |---|---|---|---|
> | Instruct-P2P | 3.58 | 5.49 | 3.68 |
> | MagicBrush | 4.68 | 5.66 | 4.52 |
> | AnyEdit | 3.18 | 5.82 | 3.21 |
> | OmniGen | 5.96 | 5.89 | 5.06 |
> | Step1X-Edit | 7.09 | 6.76 | 6.70 |
> | BAGEL | 7.36 | 6.83 | 6.52 |
> | UniWorld-V1 | 4.93 | 7.43 | 4.85 |
> | Gemini 2.0 | 6.73 | 6.61 | 6.32 |
> | Gemini 2.5 | 7.41 | 7.96 | 7.10 |
> | GPT-4o | 7.85 | 7.62 | 7.53 |
> | Qwen-Image-Edit | 8.15 | 7.86 | 7.54 |
> | **OmniGen2** | **7.58** | **7.94** | **7.21** |
> 
> > OmniGen2 PQ (7.94) 超越所有模型，Overall (7.21) 低于 GPT-4o (7.53) 但 PQ 最优。

^table-geditbench

OmniGen2 在 GEdit-Bench-EN 上的表现分析：
- **PQ (Perceptual Quality): 7.94** —— **所有模型中最高**，超越 GPT-4o (7.62) 0.32 分，说明 OmniGen2 生成的编辑图像在视觉质量上具有明显优势
- **SC (Subject Consistency): 7.58** —— 接近但低于 GPT-4o (7.85)，说明在主体保持方面仍有提升空间
- **Overall: 7.21** —— 低于 GPT-4o (7.53)，这可能是因为 Overall 的计算方式与 SC/PQ 不同，或者在其他子指标上存在劣势
- 相比 BAGEL (SC 7.36, PQ 6.83, Overall 6.52)：OmniGen2 在 PQ 上提升 **1.11 分**，在 Overall 上提升 **0.69 分**

### 3.4 Image Editing: Emu-Edit

Emu-Edit 是另一个图像编辑评测基准，使用 CLIP-Out（编辑后图像与目标描述的 CLIP 相似度）和 DINO（编辑前后图像的 DINO 特征相似度，衡量结构一致性）评估编辑质量。

> [!abstract]- 表：Emu-Edit Results
> | Model | CLIP-Out | DINO |
> |-------|----------|------|
> | OmniGen2 | 0.311 | 0.876 |

^table-emuedit

OmniGen2 在 Emu-Edit 上的两个指标均达到最佳：
- **CLIP-Out: 0.311** —— 衡量编辑后图像与目标描述的语义匹配度，越高说明编辑越符合指令
- **DINO: 0.876** —— 衡量编辑前后图像的结构一致性，越高说明编辑越精确、未编辑区域保持得越好
- 两个指标同时最优表明 OmniGen2 在"精确编辑"和"保持不变"之间取得了良好平衡

### 3.5 In-Context Generation: OmniContext

![](https://arxiv.org/html/2506.18871v4/x7.png)

*Figure 7: In-Context 生成和编辑展示。OmniGen2 能将参考图像中的主体无缝融入新场景，保持高保真身份一致性。*

OmniContext 是本文提出的 In-Context Generation 评测基准，专门评估模型基于上下文参考图像生成新图像的能力。包含 **8 个任务类别**，分为三种类型：
- **SINGLE**：单张参考图像
- **MULTIPLE**：多张参考图像
- **SCENE**：场景级生成

评测指标：
- **PF (Prompt Following)**：指令遵循度
- **SC (Subject Consistency)**：主体一致性（生成图像中的主体是否与参考图像一致）
- **Overall Score**：PF 和 SC 的几何平均 (geometric mean)

使用 **GPT-4.1** 作为裁判模型进行评估。

> [!abstract]- 表：OmniContext Benchmark Results
> | Model | SINGLE Character | SINGLE Object | MULTIPLE Character | MULTIPLE Object | MULTIPLE Char.+Obj. | SCENE Character | SCENE Object | SCENE Char.+Obj. | Average ↑ |
> |---|---|---|---|---|---|---|---|---|---|
> | Flux.1 Kontext max | 8.48 | 8.68 | - | - | - | - | - | - | - |
> | Gemini 2.0 Flash | 5.06 | 5.17 | 2.91 | 2.16 | 3.80 | 3.02 | 3.89 | 2.92 | 3.62 |
> | Gemini 2.5 Flash | 8.62 | 8.91 | 7.88 | 8.92 | 7.39 | 7.29 | 7.05 | 6.68 | 7.84 |
> | GPT-4o | 8.90 | 9.01 | 9.07 | 8.95 | 8.54 | 8.90 | 8.44 | 8.60 | 8.80 |
> | UNO | 6.60 | 6.83 | 2.54 | 6.51 | 4.39 | 2.06 | 4.33 | 4.37 | 4.71 |
> | BAGEL | 5.48 | 7.03 | 5.17 | 6.64 | 6.24 | 4.07 | 5.71 | 5.47 | 5.73 |
> | Qwen-Image-Edit | 8.35 | 9.13 | 7.65 | 8.85 | 7.90 | 5.16 | 7.75 | 6.73 | 7.69 |
> | OmniGen | 7.21 | 5.71 | 5.65 | 5.44 | 4.68 | 3.59 | 4.32 | 5.12 | 4.34 |
> | **OmniGen2** | **8.19** | **8.63** | **7.45** | **7.91** | **7.93** | **7.75** | **7.91** | **7.93** | **7.95** |
> 
> > OmniGen2 Average 7.95 超越 Gemini 2.5 Flash (7.84) 和 Qwen-Image-Edit (7.69)，接近 GPT-4o (8.80)。

^table-omnicontext

OmniGen2 在 OmniContext 上取得 **7.95** 的 Overall Score：
- 超越 Gemini 2.5 Flash (7.84) **0.11 分**
- 超越 Qwen-Image-Edit (7.69) **0.26 分**
- 这表明 RL alignment 中的 In-Context Generation 阶段（使用 Qwen2.5-VL-72B judge 作为 reward）有效提升了上下文理解和主体保持能力
- 注意：OmniContext benchmark 是本文提出的，可能存在一定的 self-benchmarking 偏见

### 3.6 Ablation Studies

![](https://arxiv.org/html/2506.18871v4/x8.png)

*Figure 8: OmniGen2 的局限性可视化。第1行：中文 prompt 和低质量输入效果差；第2行：人体形态修改困难；第3行：多图源模糊指令敏感。*

> [!abstract]- 表：Ablation Studies - Task Selection & Reward Signal
> | Experiment | Finding |
> |------------|---------|
> | Task selection: Skill overlap | Edit & GenEval tasks have overlapping skills → positive synergy |
> | Task selection: Skill conflict | OCR-only task → negative transfer to editing performance |
> | Reward signal: HPSv3 aesthetic | Reward hacking: PQ inflated to 8.22, SC/IC collapse |
> | Schedule: Edit first | Editing provides richer supervision → better than T2I first |
> | Final curriculum | Edit → GenEval → IC optimal across all metrics |

^table-ablation

Ablation study 的关键发现：

1. **任务选择 (Task Selection)**：
   - **Skill overlap → synergy**：Editing 和 GenEval 共享属性绑定、空间关系理解等技能，多任务 RL 产生正向协同
   - **Skill conflict → negative transfer**：仅 OCR 任务的 RL 会降低 editing 性能，说明不相关任务可能产生干扰

2. **Reward Signal**：
   - **HPSv3 aesthetic reward** 导致严重的 **reward hacking**：模型学会生成"美观"但不符合指令的图像，PQ 虚高到 8.22，但 SC 和 IC 崩溃
   - 说明 aesthetic reward 作为 sole reward signal 是危险的，需要与 instruction-following reward 结合

3. **Training Schedule**：
   - **Edit first > T2I first**：先进行 editing RL 优于先进行 T2I RL
   - 原因：editing 提供更丰富的监督信号（需要同时理解语义和操作像素），为后续任务提供更好的初始化

4. **最终最优课程**：**Edit → GenEval → IC**，在所有评测指标上达到最佳平衡。

## 4 结论

OmniGen2 通过 **decoupled VLM + DiT** 架构，成功实现了 instruction-aligned multimodal generation。核心贡献总结如下：

1. **Variable-length hidden states 条件化**：使用 VLM 的可变长度 hidden states 直接条件化 DiT，避免了 fixed-length query tokens 带来的 information bottleneck，同时保持了 decoupled 架构的模块化优势。

2. **Omni-RoPE 3D 位置编码**：$\text{PosID}_k(h,w) = (\Delta I^{(k)}, h, w)$，优雅地解决了多图像场景下的空间对齐问题，在 toy experiments 中表现出 3x 收敛速度和 6x loss 降低。

3. **Progressive Multi-Task RL Alignment**：系统性地探索了 GRPO 在多模态生成中的应用，发现 Edit first > T2I first、reward hacking (HPSv3)、skill overlap → synergy 等重要规律。

4. **OmniContext Benchmark**：填补了 In-Context Generation 评测的空白。

**性能总结**：在 GenEval (0.95)、OmniContext (7.95)、Emu-Edit (CLIP-Out 0.311, DINO 0.876) 上达到 SOTA；在 GEdit-Bench-EN 的 PQ (7.94) 上超越 GPT-4o。

**已知局限**：
1. 中文 prompt 效果不如英文（训练数据分布偏向英文）
2. 对 body shape modification 效果不佳（可能受训练数据和 VAE 表征限制）
3. 对输入图像质量敏感（低质量输入导致输出质量下降）
4. 模糊的多图像指令导致性能下降（需要更明确的指令）
5. VLM frozen 导致无法从生成任务中改善理解能力（这是 decoupled 架构的固有限制）

## 5 思考

### 5.1 优点

1. **架构设计巧妙——兼顾解耦与信息流动**：decoupled VLM + DiT 架构在保持模块独立性（VLM frozen、DiT 独立训练）的同时，通过 variable-length hidden states 实现了有效的跨模态信息传递。这避免了 MoT 的高训练成本和 tight coupling，也避免了 MetaQuery 的 information bottleneck。可以说是在两种极端之间找到了一个 sweet spot。

2. **Omni-RoPE 的工程优雅性**：将多图编辑的空间对齐问题转化为位置编码设计问题，用 $\Delta I^{(k)} + (h,w)$ 的简洁公式统一了多图、多模态、编辑一致性等多个需求。收敛速度 3x、loss 6x 的实验结果非常有说服力。

3. **RL Alignment 的系统性研究**：论文对 GRPO 在多任务场景下的应用进行了深入且诚实的探索：
   - 发现了任务顺序的重要性（Edit first > T2I first）
   - 揭示了 reward hacking 的具体表现（HPSv3 → PQ inflated, SC/IC collapse）
   - 验证了 skill overlap 的正向协同和 skill conflict 的负向迁移
   - 这些发现对社区有重要参考价值

4. **数据工程全面且有创新**：涵盖预训练（140M 开源 + 10M 私有）、SFT（精选 + 蒸馏）、RL（多阶段、多 reward）各阶段的数据需求。特别是 video-sourced In-Context data（SAM2 + VLM filtering）和 reflection data（交错 text-image 序列）的设计具有创新性。

5. **VLM frozen 的务实选择**：虽然 frozen VLM 限制了上限，但在当前阶段，这是最稳妥的选择——避免了理解能力退化的风险，且 VLM 的理解能力已经足够强。

### 5.2 缺点

1. **VLM frozen 的固有限制**：无法从生成任务中改善理解能力，长期来看可能成为瓶颈。BAGEL 的 MoT 架构通过 end-to-end 训练实现了理解和生成的相互增强，理论上上限更高。OmniGen2 的 frozen VLM 策略是短期稳健但长期受限的选择。

2. **中文能力不足**：论文坦诚中文 prompt 效果不如英文，这是训练数据分布偏向英文所致。作为一个声称 "unified" 的模型，多语言能力的缺失是一个明显的短板。

3. **对输入图像质量敏感**：低质量输入导致输出质量下降，说明模型缺乏对退化输入的鲁棒性。这在实际应用中是一个重要问题。

4. **OneIG-Bench 上的表现不佳**：得分 0.47，低于 Gemini 2.5 Flash (0.52) 和 Qwen-Image (0.50)，说明 RL alignment 可能在优化 GenEval 的同时，对其他质量维度产生了 trade-off。

5. **RL Alignment 的成本和复杂性**：需要多阶段、多 reward model 的 RL 训练（EditScore、GenEval verifiable reward、Qwen2.5-VL-72B judge），且 reward hacking 问题需要精心设计 reward 函数。这增加了训练的工程复杂度和计算成本。

6. **Omni-RoPE 的 scaling 验证不足**：toy experiments 中的 3x 收敛和 6x loss 降低是在小规模实验中观察到的，在 full-scale 训练中是否仍有此显著优势需要更多验证。

7. **Self-benchmarking 问题**：OmniContext benchmark 是本文提出的，OmniGen2 在该 benchmark 上的 SOTA 需要第三方验证。同时，GenEval 上的 verifiable reward 与 RL 阶段的 GenEval reward 高度对齐，可能存在 benchmark optimization 的偏见。

8. **与 BAGEL 的对比不完整**：BAGEL 强调 pretraining data scaling 的重要性，OmniGen2 强调 RL alignment，两者的最佳策略可能是互补的。一个 VLM frozen + data scaling + RL alignment 的组合方案值得探索。

### 5.3 Wiki Connections

- **Diffusion Transformer (DiT)**：OmniGen2 生成 backbone 的架构基础，将 Transformer 引入扩散模型
- **Flow Matching**：DiT 使用的训练范式，相比 DDPM 有更好的训练稳定性和采样效率
- **RoPE (Rotary Position Embedding)**：Omni-RoPE 的基础，由 Su et al. 提出，最初用于 LLM 的位置编码
- **GRPO (Group Relative Policy Optimization)**：OmniGen2 RL 阶段使用的核心算法，是 PPO 的变体，通过 group-level relative comparison 更新策略
- **Classifier-Free Guidance (CFG)**：推理时的引导策略，通过条件/无条件生成的线性组合提升质量
- **Qwen2.5-VL**：OmniGen2 使用的 VLM 基础模型，由阿里云通义实验室开发
- **Lumina-Image 2.0**：OmniGen2 使用的 DiT 架构基础
- **BAGEL**：MoT 架构的代表，强调 pretraining data scaling 和 end-to-end 训练
- **MetaQuery / External Diffuser**：decoupled VLM + DiT 的另一类方法，使用 fixed-length query tokens
- **SAM2 (Segment Anything Model 2)**：用于 In-Context 数据的视频主体分割
- **GenEval**：组合式 T2I 评测基准，评估属性绑定、空间关系等
- **GEdit-Bench**：图像编辑评测基准
- **OmniContext**：本文提出的 In-Context Generation 评测基准
- **Reward Hacking**：RL 中的常见问题，模型利用 reward 函数的漏洞获得高分但实际质量下降，在 RLHF 和 RL 中均有广泛讨论
- **HPSv3 (Human Preference Score v3)**：基于人类偏好的美学评分模型，被发现存在 reward hacking 问题
- **Information Bottleneck**：信息瓶颈理论，OmniGen2 通过 variable-length hidden states 解决此问题

### 5.4 与同期统一多模态模型的对比

| 维度 | OmniGen2 | BAGEL | Tuna-2 | Lumina-DiMOO | UniWorld-V1 |
|---|---|---|---|---|---|
| **架构** | VLM+DiT（解耦） | MoT（双专家共享注意力） | 单 Transformer，无编码器 | 单 Transformer（离散扩散） | VLM+SigLIP+FLUX |
| **条件化方式** | VLM hidden states → DiT cross-attn | 共享自注意力 | 同一 Transformer | 同一 Transformer | SigLIP2 → FLUX cross-attn |
| **VLM** | Qwen2.5-VL-3B（冻结） | MoT 理解专家（训练） | 同一模型 | LLaDA-Base | Qwen2.5-VL-7B（冻结） |
| **RL 对齐** | **GRPO 三阶段课程** | 无 | 无 | Self-GRPO | 无 |
| **GenEval** | **0.95（最高）** | 0.88 | competitive | 0.88 (0.91) | — |
| **GEdit-Bench** | SC 7.58, **PQ 7.94** | SC 7.36, PQ 6.83 | — | — | — |
| **独特能力** | Omni-RoPE + IC 生成 | 涌现 + 世界建模 | 像素级理解 | Inpainting + Retouching | 数据效率 |

**关键洞察**：
- **RL alignment 是 OmniGen2 的核心差异化**：其他四篇论文都聚焦于架构和数据，OmniGen2 是唯一将 RL alignment 作为核心贡献的。GenEval 从 0.88→0.95 的提升主要归功于 GRPO 课程，而非架构本身。
- **解耦 vs 集成的 trade-off**：OmniGen2 的 VLM 冻结策略与 UniWorld-V1 类似，但 OmniGen2 用 variable-length hidden states 缓解了信息瓶颈，UniWorld-V1 用 SigLIP 语义特征。两者都是 External Diffuser 路线的改良。
- **benchmark optimization 的隐忧**：OmniGen2 的 GenEval SOTA（0.95）可能部分受益于 RL 阶段直接用 GenEval 作为 reward。OneIG-Bench 上的落后（0.47 vs Gemini 0.52）暗示了这一点。
- **五篇论文代表了五条不同的技术路线**：集成（BAGEL）、解耦+RL（OmniGen2）、无编码器（Tuna-2）、纯离散扩散（Lumina-DiMOO）、语义编码器条件（UniWorld-V1）。没有一条路线在所有维度上都最优，说明统一多模态模型仍处于探索阶段。

---
title: 'OmniGen2: Towards Instruction-Aligned Multimodal Generation'
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
institutions: 'BAAI (北京智源人工智能研究院); USTC; CASIA; Zhejiang University'
aliases:
  - OmniGen2
date: '2026-04-30'
publish_date: 2025-06
venue: arXiv:2506.18871
tags:
  - 论文
  - 多模态
  - 图像生成
  - 图像编辑
  - In-Context生成
  - GRPO
  - 指令对齐
  - Omni-RoPE
url: 'https://arxiv.org/abs/2506.18871'
code: 已开源（https://github.com/VectorSpaceLab/OmniGen2）
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：OmniGen2 提出了系统的指令对齐方案——先构建基于 Omni-RoPE 位置编码和 VLM+DiT 解耦架构的强基座模型，再通过 Edit→GenEval→IC 的三阶段 GRPO 渐进式课程进行多任务 RL 对齐，在 T2I 生成（GenEval 0.95）、图像编辑（GEdit 7.21）和 In-Context 生成（OmniContext 7.95）上全面达到 SOTA，并通过严谨消融证明了任务选择和调度顺序的关键性。

![](https://arxiv.org/html/2506.18871/x1.png)

*Figure 1: OmniGen2 多样化能力总览——包括 T2I 生成、图像编辑、In-Context 生成等。*

---

## Intro

### Motivation

多模态生成模型需进行指令对齐以确保可控性、语义一致性和生成质量。这涉及两个关键挑战：(1) 构建稳健、通用的基座模型，需具备初始指令遵循能力和广泛世界知识，同时避免过训练；(2) 对齐基座模型需要显式且全面的奖励信号，且需确保所有生成任务间的一致性。现有开源模型要么是专用模型（任务覆盖有限），要么是过度优化到特定美学偏好导致可塑性丧失。

### 贡献

1. OmniGen2：系统性指令对齐的多模态生成模型，在 T2I、编辑、IC 上达到 SOTA
2. 端到端指令对齐管线：从强基座模型构建到多任务对齐
3. Omni-RoPE：统一多模态 3D 位置编码，区分图像实例和空间坐标
4. OmniContext benchmark：8 类任务评估 In-Context 生成的一致性
5. 严谨消融证明任务选择+调度顺序的关键性

---

## Method 核心方法

### 1. 架构：解耦 VLM + DiT

![](https://arxiv.org/html/2506.18871/x2.png)

*Figure 2: OmniGen2 架构——采用解耦设计：VLM（Qwen2.5-VL-3B）负责多模态理解，DiT 负责高质量图像合成。两种图像编码器分工：ViT 编码图像供 VLM 理解，VAE 编码图像供 DiT 生成。*

- **VLM**：Qwen2.5-VL-3B（冻结），提供多模态理解、世界知识、指令理解
- **DiT**：随机初始化，专注高质量图像合成（Lumina-Image 2.0 架构，参数跨模态共享）
- 两者串行：VLM 处理输入→隐藏态作为 DiT 条件
- 与 MetaQuery 的关键区别：直接使用 VLM 可变长隐藏态（非固定大小 query token），无信息瓶颈
- 低层视觉特征：Flux-VAE 提供像素级细节（用于编辑等任务保持一致性）

### 2. Omni-RoPE：统一 3D 位置编码

- 每个 token 分配三维位置标识：$(\Delta_I^{(k)}, h, w)$
- $\Delta_I^{(k)}$：图像实例 ID（同图内共享），区分不同图像/模态
- $(h, w)$：局部 2D 空间坐标（从 (0,0) 计算），保持空间对应
- 关键优势：输入和输出图像中对应 patch 获得相同空间编码（编辑一致性），$\Delta_I$ 区分多图像（IC 任务）
- Toy 实验验证：比 Lumina-Image-2.0 和 Qwen2-VL 的 RoPE 变体收敛快约 3 倍，最终 loss 低约 6 倍

![](https://arxiv.org/html/2506.18871/x3.png)

*Figure 3: Omni-RoPE 示意图——每个 token 分配三维位置标识 (ΔI^(k), h, w)，其中 ΔI^(k) 区分不同图像实例，(h, w) 保留局部 2D 空间坐标。编辑任务中输入和输出图像对应 patch 获得相同空间编码，保证编辑一致性。*

### 3. 数据构建管线

- 基础能力：140M 开源 T2I 图文对 + 10M Qwen2.5-VL-72B 标注 + LLaVA-OneVision
- 编辑数据：整合 SEED-Data-Edit、OmniEdit + 自建 inpainting/视频管线
- IC 数据：从视频源构建（SAM2 分割 + VLM 语义过滤），保证主体一致性
- 交错/反思数据：增强时序推理和自校正能力

### 4. 基座模型训练

- 两阶段：预训练（分辨率课程 256→512→1024，先 T2I 后多任务混合）→ SFT（1024 分辨率，精调推理和组合能力）
- Rectified Flow 目标

### 5. 渐进式多任务 GRPO 指令对齐

这是 OmniGen2 最核心的创新。

**任务调度**（Edit → GenEval → IC）：
- $\mathcal{T}_1$ = (Edit, 通用编辑, EditScore 奖励模型)
- $\mathcal{T}_2$ = (T2I, GenEval, 可验证奖励)
- $\mathcal{T}_3$ = (IC, 通用 in-context, Qwen2.5-VL-72B 评判)

**关键设计原则**：
- 排除易 reward hacking 的奖励（如 HPSv3 美学奖励）
- 排除缺乏协同效应的任务（如 OCR only）
- 编辑优先于 T2I（编辑任务提供更丰富的监督信号，为后续学习奠定基础）

---

## 实验/评估/结果

### 多模态理解

- MMBench 79.1 / MMMU 53.1 / MM-Vet 61.8

### T2I 生成

- GenEval 0.95（超越 BAGEL 0.88、Qwen-Image 0.91）
- OneIG-Bench 0.47（仅次于 Gemini 2.5 Flash Image 和 Qwen-Image）

### 图像编辑

- GEdit-Bench Overall 7.21，SC 7.58（语义一致性第二），PQ 7.94（感知质量第一）
- Emu-Edit 上 CLIP-Out 0.311（编辑准确性最高），DINO 0.876（最佳）
- ImgEdit-Bench 超越 BAGEL

### In-Context 生成（OmniContext）

- Overall 7.95，超越 Qwen-Image-Edit-2509 (7.69) 和 Gemini 2.5 Flash Image (7.84)
- Scene 类别 7.86 尤其突出

### 消融：RL 策略的关键发现

1. **任务选择**：技能重叠带来协同增益（Edit & GenEval 超越各自单任务），技能冲突导致负迁移（OCR only 降低编辑性能）
2. **奖励信号**：HPSv3 美学奖励导致 reward hacking（PQ 虚高 8.22，SC 和 IC 崩溃）
3. **调度顺序**：Edit 优先 > T2I 优先（编辑任务更丰富的监督构建更强基础）
4. **最终课程**（Edit → GenEval → IC）在各指标上均最优

---

## 结论

OmniGen2 证明了两阶段设计（强基座 + 多任务 RL 对齐）的有效性。Omni-RoPE 解决了多图像场景的位置对应问题，GRPO 渐进式课程通过精心选择任务和调度顺序实现了跨任务协同增益。指令对齐在所有任务上一致且显著地提升了基座模型性能。

---

## 思考

### 优点

1. **消融实验的高质量**：这是 OmniGen2 最亮眼的部分。精确控制了任务选择（Edit/GenEval/IC/OCR/HPSv3）和调度顺序，清晰揭示了 reward hacking（HPSv3）、负迁移（OCR only）和正协同（Edit & GenEval）的因果效应。这种严谨性在 RL for generation 领域较为难得。

2. **Omni-RoPE 设计的简洁有效**：三维位置编码同时解决实例区分和空间对应，toy 实验验证收敛速度约 3 倍提升。这是位置编码在统一多模态生成中的一次重要探索。

3. **任务调度的洞察**：发现 Edit 优先比 T2I 优先更好——因为编辑任务提供更丰富的监督（源图像+目标图像+指令三元组）。这个发现对后续工作的训练策略设计有指导意义。

4. **OmniContext benchmark 填补空白**：In-Context 生成缺乏系统性评估，OmniContext 的 8 类任务设计（Character/Object/Scene x Single/Multiple/Scene）覆盖了 IC 的核心场景。

5. **奖励信号的审慎选择**：主动排除 HPSv3 等被确认会 reward hacking 的奖励，体现了工程实践中的深思熟虑。

### 缺点与待解决问题

1. **中英文表现差异**：论文自述中文 prompt 效果明显不如英文，存在编辑不一致的问题。

2. **人体形态修改能力弱**：可能因为真实世界中此类数据稀缺。

3. **输入图像质量敏感**：低质量输入（如加噪、降采样至 256px）导致生成质量显著下降和指令遵循能力降低。

4. **架构解耦的代价**：VLM+DiT 虽然保留了 VLM 的理解能力，但 VLM 冻结意味着它不能从生成任务中学习和改进。真正的统一理解+生成可能需要更强的表征共享。

5. **模型规模有限**：VLM 仅 3B，DiT 约 4B。更大规模下 RL 对齐的效果和跨任务迁移规律未知。

### 与已有 Wiki 的连接

- 关联概念：[[指令对齐]]、[[GRPO]]、[[Omni-RoPE]]、[[Flow Matching]]、[[Rectified Flow]]
- 关联实体：[[Qwen2.5-VL]]、[[FLUX]]、[[Lumina-Image 2.0]]
- 关联比较：与 [[BAGEL]]（AR+FM 统一多模态）和 [[UniWorld-V1]]（语义编码器条件）的架构对比，与 [[Tuna-2]]（encoder-free）的解耦程度对比

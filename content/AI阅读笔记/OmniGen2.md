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

OmniGen2 的方法论采用"强基座 + 多任务 RL 对齐"的两阶段设计。基座提供 foundation，GRPO 渐进式课程实现跨任务协同增益。

### 1. 架构：解耦 VLM + DiT + Omni-RoPE

| 组件 | 选型 | 角色 |
|------|------|------|
| **VLM** | Qwen2.5-VL-3B（冻结） | 多模态理解、世界知识、指令理解 |
| **DiT** | Lumina-Image 2.0 架构（随机初始化） | 高质量图像合成，参数跨模态共享 |
| **图像编码** | ViT（VLM 理解）+ Flux-VAE（DiT 像素级细节） | 双编码器分工 |

VLM 处理输入后，其**可变长隐藏态**直接作为 DiT 条件（非固定大小 query token，无信息瓶颈——这是与 MetaQuery 的关键区别）。

**Omni-RoPE：统一 3D 位置编码**

每个 token 分配三维位置标识 $(\Delta_I^{(k)}, h, w)$：
- $\Delta_I^{(k)}$：图像实例 ID（同图共享），区分不同图像/模态
- $(h, w)$：局部 2D 空间坐标

关键优势：编辑任务中输入和输出图像对应 patch 获得相同空间编码（编辑一致性）；$\Delta_I$ 区分多图像（IC 任务）。Toy 实验验证：比 Lumina-Image-2.0 和 Qwen2-VL 的 RoPE 变体收敛快约 3 倍，最终 loss 低约 6 倍。

### 2. 数据构建管线

| 数据类型 | 来源 | 规模 |
|---------|------|------|
| 基础 T2I | 开源图文对 + Qwen2.5-VL-72B 标注 + LLaVA-OneVision | 140M + 10M |
| 编辑 | SEED-Data-Edit + OmniEdit + 自建 inpainting/视频管线 | - |
| In-Context | 视频源（SAM2 分割 + VLM 语义过滤） | 保证主体一致性 |
| 交错/反思 | 增强时序推理和自校正能力 | - |

### 3. 基座模型训练

两阶段：预训练（分辨率课程 256→512→1024，先 T2I 后多任务混合）→ SFT（1024 分辨率，精调推理和组合能力）。Rectified Flow 目标。

### 4. 渐进式多任务 GRPO 指令对齐（核心创新）

**三阶段课程**：$\mathcal{T}_1$(Edit, EditScore RM) → $\mathcal{T}_2$(T2I, GenEval, 可验证奖励) → $\mathcal{T}_3$(IC, Qwen2.5-VL-72B 评判)

**关键设计原则**：
- 排除易 reward hacking 的奖励（如 HPSv3 美学奖励——导致 PQ 虚高但 SC/IC 崩溃）
- 排除缺乏协同效应的任务（如 OCR only——降低编辑性能）
- **编辑优先于 T2I**：编辑任务提供更丰富的监督信号，为后续学习奠定更强基础

---

## 实验/评估/结果

### T2I 生成



| Benchmark | OmniGen2 | 对比 |
|-----------|---------|------|
| **GenEval** | **0.95** | BAGEL 0.88, Qwen-Image 0.91, GPT-4o 0.84 |
| OneIG-Bench | 0.47 | 仅次于 Gemini 2.5 Flash Image 和 Qwen-Image |

### 图像编辑

| Benchmark | OmniGen2 | 对比 |
|-----------|---------|------|
| **GEdit-Bench Overall** | **7.21** | SC 7.58（第二）, PQ 7.94（**第一**） |
| Emu-Edit CLIP-Out | **0.311** | 编辑准确性最高 |
| Emu-Edit DINO | **0.876** | 最佳 |
| ImgEdit-Bench | 超越 BAGEL | - |

### In-Context 生成 (OmniContext)

| Model | Overall |
|-------|---------|
| Qwen-Image-Edit-2509 | 7.69 |
| Gemini 2.5 Flash Image | 7.84 |
| **OmniGen2** | **7.95** |

*Scene 类别 7.86 尤其突出。*

### 消融：RL 策略关键发现

| 实验 | 发现 |
|------|------|
| 任务选择 | 技能重叠→协同增益（Edit & GenEval 超单任务）；技能冲突→负迁移（OCR only 降低编辑） |
| 奖励信号 | HPSv3 美学奖励→reward hacking（PQ 虚高 8.22，SC/IC 崩溃） |
| 调度顺序 | Edit 优先 > T2I 优先（编辑提供更丰富监督） |
| **最终课程** | Edit → GenEval → IC **在各指标上均最优** |

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

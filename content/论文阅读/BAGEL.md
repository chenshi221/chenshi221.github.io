---
title: Emerging Properties in Unified Multimodal Pretraining
authors:
  - Chaorui Deng
  - Deyao Zhu
  - Kunchang Li
  - Chenhui Gou
  - Feng Li
  - Zeyu Wang
  - Shu Zhong
  - Weihao Yu
  - Xiaonan Nie
  - Ziang Song
  - Guang Shi
  - Haoqi Fan
institutions: ByteDance Seed
aliases:
  - BAGEL
  - BAGEL论文
date: '2026-04-30'
publish_date: 2025-05
venue: 'arXiv:2505.14683'
tags:
  - 论文
  - 多模态
  - 统一模型
  - MoT
  - Flow-Matching
  - 涌现
  - 图像生成
  - 图像编辑
url: 'https://arxiv.org/abs/2505.14683'
code: 已开源（checkpoints + code）
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：本文提出了 BAGEL，一个基于 MoT（Mixture-of-Transformers）架构的开源统一多模态基础模型（7B 激活/14B 总参数），通过在万亿 token 级的多模态交错数据上大规模预训练，涌现出了复杂多模态推理、自由形式图像编辑、3D 操控和世界导航等超出传统 benchmark 范围的 emergent 能力。

![](https://arxiv.org/html/2505.14683v3/x1.png)

*Figure 1: BAGEL 的多样化能力展示——从 T2I 生成、图像编辑到 3D 操控和世界导航。*

---

## Intro

### Motivation

当前统一多模态理解与生成的学术模型与闭源系统（GPT-4o、Gemini 2.0）之间存在巨大差距。作者认为，差距的核心原因在于：**现有学术模型主要在图文对（image-text pair）数据上训练，缺乏大规模多模态交错数据（interleaved data）**——即将文本、图像、视频和网页内容以自然交织的方式组织在一起的训练数据。

现有统一模型的三种架构路线各有缺陷：

1. **量化自回归（Quantized AR）**：将视觉 token 离散化后用自回归生成，实现简单但生成质量不如扩散模型，且推理延迟高（逐 token 串行解码）。
2. **外挂扩散器（External Diffuser）**：LLM 生成语义条件 token → 外挂扩散模型生成图像。收敛快但引入了信息瓶颈——LLM 的上下文被压缩成少量潜变量 token，在长上下文多模态推理中信息损失严重，违背"规模化基础模型"的哲学。
3. **集成 Transformer（Integrated Transformer）**：将 LLM 和扩散模型统一在一个 Transformer 中，无信息瓶颈，但训练成本高。

### 本文的核心主张

**关键在于用精心构建的多模态交错数据规模化训练**。随着交错预训练规模的增加，模型涌现出复杂组合能力——从基本的理解和生成，到自由形式视觉编辑，再到长上下文多模态推理。这些能力无法被传统 benchmark 捕获。

### 贡献

1. **BAGEL 模型**：开源 7B 激活参数的 MoT 统一多模态模型，在理解 benchmark 上超越现有开源 VLM，在 T2I 生成上与 FLUX.1-dev / SD3 持平或超越
2. **MoT 架构验证**：通过受控实验证明 MoT > MoE > Dense，理解与生成需要分离的参数空间
3. **多模态交错数据协议**：建立从视频和网页中可扩展构建高质量交错数据的完整流程
4. **涌现现象的实证记录**：首次在统一多模态预训练中观察到清晰的能力涌现阶段——理解先收敛→生成次之→编辑缓慢提升→复杂推理最后涌现
5. **IntelligentBench**：专门评估需要复杂多模态推理的自由形式图像编辑的新 benchmark

---

## Method 核心方法

### 1. 架构设计：MoT（Mixture-of-Transformers）

BAGEL 采用 MoT 架构，包含**两个完整的 Transformer 专家**：

- **理解专家**：处理 Text token + ViT token（视觉理解）
- **生成专家**：处理 VAE token（视觉生成）
- 两个专家在每一层通过**共享自注意力**操作同一个 token 序列
- 硬路由：生成专家只看 VAE token，理解专家只看 Text/ViT token

![](https://arxiv.org/html/2505.14683v3/x2.png)

*Figure 2: BAGEL 的 MoT 架构。两个 Transformer 专家分别处理理解和生成信息，所有 token 在每层共享自注意力。两个独立的视觉编码器分别捕获语义内容（SigLIP2）和低级像素信息（FLUX VAE）。*

**关键消融实验（1.5B 规模）**：
- Dense（单 Transformer）：MSE loss（生成）收敛慢，CE loss（理解）波动大
- MoE（仅复制 FFN 作为专家）：效果居中
- **MoT（复制完整 Transformer 作为专家）> MoE > Dense**
- 结论：理解和生成将模型参数**拉向不同的最优区域**，需要分离的参数容量

![](https://arxiv.org/html/2505.14683v3/x3.png)

*Figure 3: Dense vs MoE vs MoT 在 1.5B 规模的 loss 曲线。MoT 在 MSE loss（生成）和 CE loss（理解）上均优于其他架构，且差距在生成任务上最为显著。*

**基础设置**：
- LLM 初始化：Qwen2.5
- 视觉理解编码器：SigLIP2-so400m/14，384→980 分辨率，NaViT 原生宽高比
- 视觉生成编码器/解码器：FLUX VAE（冻结）
- 文本条件注入：时间步 embedding 直接加到 VAE token 的初始 hidden state（而非 adaLN），更简洁
- QK-Norm：训练稳定性

### 2. 广义因果注意力（Generalized Causal Attention）

这是 BAGEL 最精巧的工程设计之一。每个图像在训练中有三组 token：

- **加噪 VAE token**：用于 Rectified Flow 训练（MSE loss 计算）
- **干净 VAE token**：作为条件供后续 token 参考（t=0 噪声）
- **ViT token**：统一交错理解与生成的输入格式，提升交错生成质量

注意力规则：
- 同一样本内按模态分组（text / ViT / VAE），组内 causal（文本）或 bidirectional（视觉）
- 后续图像可以 attend 到前面图像的**干净 VAE token 和 ViT token**，但**不能 attend 到加噪 VAE**
- 多图像生成采用 Diffusion Forcing：每张图独立加噪级别，条件化于前一张图的噪声表示
- 用 PyTorch FlexAttention 实现，比原生 attention 快 2 倍

![](https://arxiv.org/html/2505.14683v3/x20.png)

*Figure 15: BAGEL 的广义因果注意力掩码示意。(a) 交错图文生成：每张图只能 attend 到前面图像的干净 VAE 和 ViT token。(b) 交错多图/视频生成：采用 Diffusion Forcing 策略，每张图条件化于前一张图的噪声表示。*

推理时：
- 缓存干净 VAE 和 ViT 的 KV cache
- 图像生成完成后将加噪 VAE 替换为干净版本
- CFG：text dropout 0.1 / ViT dropout 0.5 / clean VAE dropout 0.1

### 3. 数据工程：四类交错数据

BAGEL 的数据配方是最核心的贡献之一。

**视频交错数据（45M 条）**：
- 来源：公开在线视频 + Koala36M（教学类）+ MVImgNet2.0（多视角）
- 构建：视频分镜 → 质量过滤（分辨率、清晰度、运动稳定性）→ 帧间变化描述（用蒸馏的轻量 Qwen2.5-VL-7B 生成，限制 30 token 防幻觉）→ 采样平均 4 帧/片段
- 核心价值：提供像素级、概念级、时序和物理连续性的监督信号——"视频是最大、最自然的模拟器"

![](https://arxiv.org/html/2505.14683v3/x4.png)

*Figure 4: BAGEL 的数据构建流程。(a) 视频交错数据：视频分镜→质量过滤→帧间变化描述→交错序列。(b) 网页交错数据：原始网页→去噪清洗→Caption-first 策略插入描述→结构化交错文档。*

**网页交错数据（20M 条）**：
- 来源：OmniCorpus（来自 Common Crawl 的图文交错网页）
- 两步过滤：① 轻量 topic 选择（fastText 分类器，蒸馏自 LLM）② 精细规则过滤（去 UI、去模糊、去高文本密度、保留 3-8 图的文档）
- Caption-first 策略：先为每张图生成简短描述插入到图前（概念脚手架），减少图文弱对齐带来的生成难度
- 核心价值：百科文章、教程、设计文档等自然交织的多模态推理监督

**推理增强数据（500K 条）**：
- 受 DeepSeek-R1 启发，在生成前引入语言推理步骤
- 三个来源：① **T2I 推理**（简短模糊 prompt → CoT 推理 → 详细 prompt → FLUX 生成图）② **自由形式图像编辑**（源图+目标图+用户查询+R1 推理链示例）③ **概念编辑**（网页交错图中采样图对→VLM 生成推理→质量过滤）
- 核心价值：教会模型"先想再画"

**传统图文对数据**：VLM 预训练 + T2I 生成对（含 FLUX/SD3 合成数据）

### 4. 训练策略

**四阶段训练**（总计约 5.2T tokens）：

| 阶段 | Tokens | 分辨率 | 关键 |
|------|--------|--------|------|
| Alignment | 4.9B | 378×378 | 只训练 MLP connector，对齐 ViT 和 LLM |
| Pre-training (PT) | 2.5T | 256-512 (gen) / 224-980 (und) | 全参数训练，Gen:Und ≈ 4:1 |
| Continued Training (CT) | 2.6T | 512-1024 (gen) / 378-980 (und) | 提升分辨率 + 提高交错数据比例（15-20%） |
| SFT | 72.7B | 512-1024 / 378-980 | 高质量子集微调 |

**关键调参发现**：
- **数据采样比**：生成数据应远多于理解数据（4:1 ~ 8:1），MSE loss 显著下降而 CE loss 几乎不变

![](https://arxiv.org/html/2505.14683v3/x6.png)

*Figure 5: 不同数据采样比的 loss 曲线。"4g1u"（生成:理解=4:1）在 MSE 上有 0.4% 的绝对改善，而 CE loss 几乎不受影响。*

- **学习率 trade-off**：大学习率有利于生成（MSE），小学习率有利于理解（CE）→ 分别加权

![](https://arxiv.org/html/2505.14683v3/x7.png)

*Figure 6: 不同学习率的 loss 曲线。MSE 和 CE 对学习率的需求相反——大学习率加速生成收敛，小学习率保护理解能力。*
- **Constant LR** 全程恒定学习率，便于动态扩展训练数据
- **扩散时间步 shift**：分辨率提升时从 1.0 调到 4.0，匹配噪声分布

---

## 实验/评估/结果

### 多模态理解

在 MME、MMBench、MMMU、MM-Vet、MathVista、MMVP 六个 benchmark 上评估。7B 激活参数的 BAGEL：
- 全面超越同等规模的所有统一模型（如 Janus-Pro-7B：MMMU +14.3, MM-Vet +17.1）
- 在多数 benchmark 上**超越专用理解模型**（如 Qwen2.5-VL-7B, InternVL2.5-7B）
- 1.5B 版本在 MMVP 上达到 54.7，远超大多数 7B 统一模型

### 文本到图像生成

- **GenEval**：BAGEL 82%（无 LLM rewriter）/ 88%（有 LLM rewriter），超越 FLUX.1-dev（82%）、SD3-Medium（74%）、Janus-Pro-7B（80%）
- **WISE**（世界知识推理生成）：BAGEL 52%，超越除 GPT-4o 外的所有模型；加 CoT 后达到 70%
- 原生支持中英文 prompt、任意宽高比

![](https://arxiv.org/html/2505.14683v3/x14.png)

*Figure 10: T2I 生成质量定性对比。BAGEL 显著优于 Janus-Pro 7B 和 SD3-medium。注意 BAGEL 原生支持中英文 prompt，而 SD3-medium 不支持中文。*

### 图像编辑

- **GEdit-Bench**：全面超越 Gemini 2.0，与专用编辑模型 Step1X-Edit 竞争
- **IntelligentBench**（推理编辑）：BAGEL 44.9，是 Step1X-Edit（14.9）的 3 倍；加 CoT 后 55.3
- **RISEBench**：6.1（基础）/ 11.9（+CoT）
- **KRIS-Bench**：56.21（基础）/ 60.18（+CoT）

![](https://arxiv.org/html/2505.14683v3/x15.png)

*Figure 11: 图像编辑与操控定性对比。BAGEL 在多种编辑场景下展现了优于 Step1X-Edit、IC-Edit 和 Gemini 2.0 的一致性和指令遵循能力。*

![](https://arxiv.org/html/2505.14683v3/x16.png)

*Figure 12: IntelligentBench 上的定性对比。BAGEL 能处理需要多步推理和世界知识的复杂编辑任务（如"将苹果logo变成毕加索风格"），而 Step1X-Edit 往往直接复制输入图像。*

### 关键发现：CoT 推理显著增强生成

在**所有编辑 benchmark** 上，加入显式 CoT 思考过程后 BAGEL 的性能都有大幅提升。这说明：
1. 模型**内部化了推理能力**（不是简单的 prompt engineering）
2. 推理训练数据（500K 条）有效教会了模型"先想再做"
3. 推理增强在需要世界知识的复杂任务上收益最大

![](https://arxiv.org/html/2505.14683v3/x17.png)

*Figure 13: "先思考再生成"的有效性。(a) T2I 生成：仅给简短 prompt 时失败，加入 CoT 推理后成功生成正确图像。(b) 智能编辑：CoT 推理显著提升了需要世界知识和多步推理的编辑质量。*

### 涌现现象

通过回溯历史 checkpoint，发现不同能力的收敛阶段截然不同（以达峰值 85% 所需 token 衡量）：

| 能力 | 达 85% 所需 token | 性质 |
|------|-------------------|------|
| 图像理解 | 0.18T | 最早收敛 |
| T2I 生成 | 0.68T | 较早收敛 |
| 经典图像编辑 | 2.64T | 缓慢提升 |
| **智能编辑（推理）** | **3.61T** | **涌现式增长** |

智能编辑在 3T token 后才出现突变式提升（15→45），而传统编辑任务在分辨率提升后变化不大——这说明**复杂多模态推理是一种真正的涌现能力**，不是 loss 曲线能预测的。

定性观察也佐证了这点：3.5T 前，模型在复杂编辑任务上倾向"回退"到复制输入图；3.5T 后开始展现清晰的多步推理和语义编辑。

![](https://arxiv.org/html/2505.14683v3/x8.png)

*Figure 7: 不同能力的收敛速度对比。理解和生成 benchmark 在早期（0.18T/0.68T）即达峰值 85%，经典编辑需要 2.64T，而智能编辑在 3.61T 后才出现涌现式突变。*

![](https://arxiv.org/html/2505.14683v3/x12.png)

*Figure 8: 不同训练 token 量的 T2I 和编辑效果对比。生成质量在 1.5T 前已较强，之后分辨率提升带来小幅改善。文字渲染能力（如正确拼写"hello"和"BAGEL"）在 1.5T~4.5T 之间涌现。*

![](https://arxiv.org/html/2505.14683v3/x13.png)

*Figure 9: 智能编辑任务的涌现行为。3.5T 前模型倾向"回退"到复制输入图（不理解任务时的 fallback 策略）；3.5T 后开始展现清晰推理和语义编辑——这是一种无法从 loss 曲线预测的质变。*

### 世界建模

通过在 video + navigation 数据上微调，BAGEL 展示了：
- **世界导航**：沿着相机轨迹生成后续视角
- **旋转**：旋转输入图生成新视角
- **多帧生成**：给定 prompt 生成多张连续图像
- **跨域泛化**：仅在真实街景导航数据上训练，能泛化到水墨画、卡通、游戏等完全不同的域

![](https://arxiv.org/html/2505.14683v3/x19.png)

*Figure 14: BAGEL 的世界建模能力展示。包括世界导航（沿相机轨迹前进）、旋转（改变视角）、多帧生成（给定 prompt 生成连续图像序列），以及跨域泛化（从真实街景泛化到水墨画、卡通等）。*

---

## 结论

BAGEL 证明了：在统一多模态预训练中**规模化交错数据**可以涌现出复杂多模态推理能力。其 MoT 架构有效分离了理解和生成的参数空间，广义因果注意力方案优雅地处理了多模态交错序列的训练和推理。通过开源的模型、数据和训练协议，BAGEL 为多模态研究社区提供了一个强大的基础。

---

## 思考

### 优点

1. **数据哲学的正确性**：论文的核心论点是"多模态交错数据是关键"，并用系统的消融证据（涌现现象、不同阶段的收敛速度）支撑了这个论点。这比架构创新更深层——它回答了"真正的统一多模态能力从何而来"。

2. **MoT 的消融实验严谨**：在 1.5B 规模上做了 Dense vs MoE vs MoT 的公平对比，证明 MoT 的优势。结论"理解和生成将参数拉向不同方向"有实验支撑而非空谈。

3. **广义因果注意力的精巧设计**：三组 token（加噪 VAE / 干净 VAE / ViT）的分工和注意力规则设计精巧而实用。在保持扩散训练范式的同时支持了多模态交错的训练和推理，是一个前后一致的方案。

4. **推理增强数据的方法论贡献**：用 DeepSeek-R1 的推理链 + FLUX 的生成能力，构建了让模型"先思考再生成"的训练数据。这种"用已有模型的能力合成下一阶段训练数据"的方法，是 scaling 时代的核心方法论。

5. **开源度**：模型权重、训练协议、数据创建流程全部开源。在 GPT-4o 的技术细节完全保密的情况下，BAGEL 提供了研究多模态涌现的宝贵实证数据。

### 缺点与待解决问题

1. **推理增强数据的"蒸馏天花板"**：推理增强数据由 DeepSeek-R1 生成推理链 + FLUX 生成目标图。那么 BAGEL 的推理能力本质上受限于这两个 teacher 模型。它学会的是"模仿推理"，还是获得了真正的推理能力？论文没有给出在这方面的分析。

2. **涌现的因果关系未严格证明**：论文声称交错数据是涌现的原因，但涌现可能也来自模型规模、训练时长等的交互。缺少"控制数据量、改变交错比例"的严格因果实验。

3. **与 GPT-4o 的差距仍然显著**：在 WISE（世界知识推理生成）上，BAGEL+CoT 的 70% vs GPT-4o 的 80% 仍有 10 个点的差距。在 IntelligentBench 上差距更大（55.3 vs 78.9）。数据规模和模型容量的进一步扩展能缩小多大差距，是未知数。

4. **世界建模能力的边界不清晰**：导航/旋转/多帧生成的展示令人印象深刻，但只展示了"works"的案例。对视角切换的精确度、长序列的一致性、复杂场景下的失败模式等，论文没有系统评估。

5. **训练成本的完整披露不足**：论文给出了训练 token 数和架构参数，但没有报告总 GPU 小时或能耗。考虑到 Integrated Transformer 路线的训练成本远高于 External Diffuser 路线，这对社区评估技术路线的可行性很重要。

### 与已有 Wiki 的连接

- 关联概念：[[原生多模态模型]]、[[MoE 混合专家模型]]（MoT 是 MoE 的衍生）、[[DiT 扩散 Transformer]]、[[Flow Matching]]、[[知识蒸馏]]（推理增强数据本质上是蒸馏 DeepSeek-R1）
- 关联实体：[[BAGEL]]、[[GPT-4o]]、[[FLUX]]
- 关联比较：[[Wiki/Comparisons/统一多模态模型架构比较]]、[[Wiki/Comparisons/扩散模型架构比较 UNet vs DiT]]
- 关联问题：[[Wiki/Questions/多模态模型的最终形态是原生统一还是模块化组装]]

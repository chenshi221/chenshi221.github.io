---
title: "Tuna-2: Pixel Embeddings Beat Vision Encoders for Multimodal Understanding and Generation"
authors:
  - (Tuna Team)
institutions: "(未明确)"
aliases:
  - Tuna-2
  - Tuna2
date: 2026-04-30
publish_date: 2025-04
venue: "arXiv:2604.24763"
tags:
  - 论文
  - 多模态
  - 统一模型
  - Encoder-Free
  - 像素嵌入
  - 图像生成
  - 图像编辑
url: "https://arxiv.org/abs/2604.24763"
code: ""
rating: "⭐⭐⭐⭐⭐"
status: Read
---

## 1 Intro

### 1.1 Motivation

当前 unified multimodal model (UMM) 的主流范式是引入 pretrained vision encoder（如 CLIP、SigLIP 用于理解，VAE 用于生成），但这些 encoder 本身存在局限：

- **架构复杂度高**：需要额外的 connector/alignment stage 将视觉 encoder 与 LLM 对齐
- **表示瓶颈**：pretrained encoder 的表示空间可能不完全适配下游多模态任务
- **像素级任务受限**：CLIP 等 representation encoder 更关注语义，对细粒度像素级理解（如计数、空间推理）能力不足

因此，一个自然的问题是：**能否完全抛弃 pretrained vision encoder，直接从原始像素构建统一多模态模型？**

### 1.2 核心主张

Tuna-2 的核心主张是：**raw pixel embeddings（通过简单 patch embedding 层）可以完全替代 pretrained vision encoder**，在理解和生成两个方向上同时达到甚至超越使用 encoder 的方案。

具体而言，Tuna-2 将原始像素通过 patch embedding 直接转换为 token，送入单一的 unified transformer decoder 处理，无需任何 VAE、CLIP 或 SigLIP。

### 1.3 贡献

1. **提出 Tuna-2**：一个完全无 encoder 的 native UMM，仅使用 raw pixel embeddings + 统一 transformer decoder（7B 参数）
2. **Masking-Based Feature Learning**：通过随机 mask 图像 patch，同时提升生成和理解能力，是关键技术创新
3. **系统性消融实验**：对比了三种架构演进路径（Tuna → Tuna-R → Tuna-2），证明 encoder-free 方案在规模化训练后可超越 encoder-based 方案
4. **Pixel-space generation**：采用 xx-prediction 和 vv-loss 的 flow matching 直接在像素空间生成，遵循 JiT paradigm

---

![](https://arxiv.org/html/2604.24763v1/x1.png)

*Figure 1: Tuna 架构演进与多模态性能对比。从 Tuna（VAE + CLIP）→ Tuna-R（仅 CLIP）→ Tuna-2（无编码器，raw pixel patches），逐步移除视觉编码器组件。*

---

## 2 Method

### 2.1 架构

#### 2.1.1 架构演进：从 Tuna 到 Tuna-2

Tuna-2 的设计经历了三代演进：

1. **Tuna（Baseline）**：使用 VAE 负责图像生成，representation encoder（CLIP）负责视觉理解，两者通过 connector 与 LLM 对齐
2. **Tuna-R**：移除 VAE，仅保留 representation encoder，通过 flow matching 在像素空间直接生成（而非 latent space）
3. **Tuna-2（最终方案）**：**完全移除 representation encoder**，仅使用简单 patch embedding 层将原始像素转换为 token

#### 2.1.2 Pixel Patch Embedding

- 输入图像（如 256×256）被划分为固定大小的 patch
- 每个 patch 通过线性投影层（无任何 pretrained 权重）映射为 token embedding
- 这些 pixel tokens 与 text tokens 拼接后，统一送入 transformer decoder

#### 2.1.3 Unified Transformer Decoder

- 单一 transformer decoder 处理所有模态（text tokens + pixel tokens）
- 无需任何模态间的 connector 或 projection module
- 模型参数量 7B

#### 2.1.4 Masking-Based Feature Learning（核心创新）

![](https://arxiv.org/html/2604.24763v1/x3.png)

*Figure 3: Masking-Based Feature Learning 示意。训练时随机将部分图像 patch 替换为可学习的 mask token，同时服务于生成（masked prediction）和理解（regularization）。*

- 训练时随机将部分图像 patch 替换为可学习的 **mask token**
- **对生成的作用**：创建更困难的 denoising 任务，迫使模型学习更强的视觉先验
- **对理解的作用**：充当 regularization，迫使模型在部分观察条件下进行推理
- 消融实验表明 masking 对理解和生成均有稳定提升

#### 2.1.5 Pixel-Space Generation

- 采用 **xx-prediction** 和 **vv-loss** 进行 flow matching
- 直接在像素空间（而非 latent space）执行生成，遵循 **JiT paradigm**
- 无需 VAE decoder 将 latent 解码回像素

### 2.2 Training Mechanism

训练分为两个阶段：

**Stage 1: Pretraining**
- 联合训练 300k steps
- 数据组成：image captioning（70%）、T2I generation（30%）、text-only（占总量 20%）
- 理解和生成的最优数据比例为 7:3（generation:understanding）

**Stage 2: SFT (Supervised Fine-Tuning)**
- 训练 50k steps
- 数据类型：instruction-following、image editing、high-quality generation

**关键优势**：Tuna-2 不需要额外的 alignment stage（如 Tuna-R 所需的 connector alignment），训练流程更简洁。

### 2.3 Data

| 阶段 | 数据类型 | 比例 |
|------|----------|------|
| Stage 1 | Image captioning | 70% |
| Stage 1 | T2I generation | 30% |
| Stage 1 | Text-only | 占总量 20% |
| Stage 2 | Instruction-following | — |
| Stage 2 | Image editing | — |
| Stage 2 | High-quality generation | — |

最优 generation:understanding 数据比为 **7:3**。

### 2.4 Training Strategy

- **无 connector alignment**：区别于 Tuna-R 需要额外对齐阶段，Tuna-2 直接端到端训练
- **Masking 策略**：随机 mask 部分 patch，mask ratio 通过消融确定最优值
- **Flow matching**：使用 xx-prediction 和 vv-loss，在像素空间执行 denoising
- **训练动态**：Tuna-R 初期收敛更快（受益于 semantic priors），但 Tuna-2 在理解任务上逐步追平甚至超越；生成任务上差距在 SFT 后显著缩小

---

## 3 实验结果

![](https://arxiv.org/html/2604.24763v1/x2.png)

*Figure 2: Tuna-2 的 T2I 生成和图像编辑展示。完全无编码器的架构仍能实现高保真生成和精确编辑。*

### 3.1 多模态理解（Multimodal Understanding）

Tuna-2 在多个理解 benchmark 上超越所有 7B native UMM（Tuna、Show-o2、Janus-Pro），特别是在像素级任务上表现突出。

在通用 benchmark（GQA、RealWorldQA、MMVet、MMMU）上，Tuna-2 与使用预训练 encoder 的 Tuna 持平甚至略优，证明 encoder-free 方案在通用理解上不输 encoder-based 方案。

**像素级任务的显著优势**是 Tuna-2 最亮眼的发现：在 V*（视觉推理）、CountBench（计数）、VisuLogic（逻辑推理）上，Tuna-2 大幅超越所有对比模型。这说明：
- CLIP/SigLIP 等语义编码器在预训练时被优化为"语义压缩"，倾向于丢弃细粒度像素信息
- Tuna-2 的 raw pixel embedding 保留了完整的像素信息，让 Transformer 自己决定哪些信息重要
- 这种"不做预设的表征学习"在需要精细感知的任务上有天然优势

与专用 LMM（如 Qwen2.5-VL）对比：Tuna-2 在通用 benchmark 上略低（Qwen2.5-VL 有更强的预训练），但在像素级任务上反超，说明 unified training 对细粒度理解有正向迁移。

> [!abstract]- 表：多模态理解 Benchmark 对比（Table 1）
> 
> | Models | Size | GQA | RealWorldQA | MMVet | MMMU | MMVP | SEED-Bench2+ | AI2D | ChartQA | OCRBench | V\* | CountBench | VisuLogic |
> |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
> | **Understanding-only Models (LMMs)** | | | | | | | | | | | | | |
> | LLaVA-1.5 | 7B | 62.0 | 54.8 | 32.9 | 35.7 | - | - | 55.5 | 17.8 | 31.8 | - | - | - |
> | Qwen-VL-Chat | 7B | 57.5 | 49.3 | 47.3 | 37.0 | - | - | 57.7 | 49.8 | 48.8 | - | - | - |
> | LLaVA-OV | 7B | - | 69.9 | 51.9 | 48.8 | 77.3 | 62.2 | 81.4 | 80.9 | 62.2 | 72.7 | 76.2 | 24.8 |
> | Qwen2.5-VL | 7B | 60.7 | 69.9 | 61.7 | 58.6 | 78.0 | 70.5 | 82.7 | 83.0 | 83.7 | 71.2 | 74.1 | 20.0 |
> | **Composite UMMs** | | | | | | | | | | | | | |
> | TokenFlow-XL | 14B | 62.5 | 56.6 | - | 43.2 | - | - | - | - | - | - | - | - |
> | BLIP3-o | 4B | - | 60.4 | - | 46.6 | - | - | - | - | - | - | - | - |
> | Tar | 7B | 61.3 | - | - | 39.0 | 74.3 | 46.2 | - | - | - | 41.4 | 64.2 | 24.3 |
> | X-Omni | 7B | 62.8 | 62.6 | - | 47.2 | - | - | 76.8 | 81.5 | 70.4 | - | - | - |
> | **Native UMMs** | | | | | | | | | | | | | |
> | BAGEL | 14B | **66.4** | **72.8** | **67.2** | **55.3** | **85.0** | **71.9** | **89.2** | 78.5 | 73.3 | 70.2 | 82.5 | 41.7 |
> | Ming-UniVision | 16B | 59.4 | 59.1 | 64.2 | 40.3 | 71.0 | 56.8 | 82.8 | 76.7 | 72.4 | 48.2 | 76.8 | 26.7 |
> | Harmon | 1.5B | 58.9 | 49.8 | - | 38.9 | 61.7 | 41.6 | 57.0 | 29.8 | 11.2 | 41.9 | 67.0 | 25.1 |
> | JanusFlow | 1.3B | 60.3 | 41.2 | 36.2 | 29.3 | 67.7 | 39.8 | 54.2 | 42.4 | 53.2 | 42.9 | 78.6 | 22.0 |
> | Emu3 | 8B | 60.3 | 57.4 | 23.5 | 31.6 | 71.0 | 44.6 | 70.0 | 69.4 | 68.7 | 53.4 | 65.2 | 24.7 |
> | VILA-U | 7B | 60.8 | 46.8 | 26.3 | 31.2 | 62.7 | 31.9 | 49.0 | 11.4 | 23.3 | 38.7 | 55.2 | 25.4 |
> | Janus-Pro | 7B | 62.0 | 58.0 | 41.1 | 41.0 | 73.3 | 56.3 | 71.3 | 25.8 | 59.0 | 47.6 | 53.2 | 23.8 |
> | Show-o2 | 7B | 63.1 | 64.7 | 39.6 | 48.9 | 76.7 | 61.3 | 78.6 | 52.3 | 32.4 | 44.5 | 63.5 | 26.9 |
> | OneCat | 9B | 63.1 | 65.2 | 52.2 | 41.9 | 71.3 | 61.6 | 77.8 | 81.2 | 79.0 | 63.4 | 34.2 | 24.9 |
> | Tuna | 7B | 63.9 | 66.1 | 42.9 | 49.8 | 70.7 | 52.7 | 79.3 | **85.8** | 74.3 | 52.4 | 73.5 | 22.4 |
> | **Tuna-R** | **7B** | 63.5 | 67.9 | 46.7 | 51.1 | 74.7 | 58.4 | 79.4 | 85.6 | 78.3 | 57.6 | 77.8 | 26.2 |
> | **Tuna-2** | **7B** | **65.0** | 67.7 | **51.7** | 50.7 | **77.3** | **61.1** | **79.6** | 85.6 | **79.7** | **59.2** | **81.7** | **28.8** |
> 
> **分析**：
> - 通用 benchmark：Tuna-2 (7B) 在 GQA (65.0) 上超越所有 7B 模型，接近 BAGEL (66.4, 14B)
> - 像素级任务优势显著：V\* (59.2 vs Janus-Pro 47.6, +24%)、CountBench (81.7 vs 53.2, +54%)、VisuLogic (28.8 vs 23.8, +21%)
> - 与 Tuna/Tuna-R 对比：通用任务持平或略优，像素级任务全面领先（V\* 59.2 > 57.6 > 52.4）

^table-understanding

### 3.2 图像生成（Image Generation）

Tuna-2 在 GenEval 和 DPG-Bench 上达到 competitive 水平，且 LLM-Judge 评估显示其在图像多样性上优于 Tuna-R。

**关键发现——多样性 vs 质量的 trade-off**：LLM-Judge（GPT-5.4 和 Claude Opus 4.7）评估显示 Tuna-2 在图像多样性上优于 Tuna-R，但在某些质量维度上略逊。这反映了一个深层规律：encoder-free 方案因为没有预训练的视觉先验约束，生成结果更多样但可能不够"精致"；而 encoder-based 方案（Tuna-R）因为有 CLIP 的语义先验，生成结果更"安全"但多样性受限。

**与 BAGEL 对比**：BAGEL 的 GenEval 得分更高（0.88 vs Tuna-2 的 competitive），但 BAGEL 使用了 dual encoder（SigLIP2 + FLUX VAE）和 14B 参数，架构复杂度远高于 Tuna-2 的 7B 单一 Transformer。在参数效率上，Tuna-2 更优。

> [!abstract]- 表：图像生成 Benchmark 对比
> | Benchmark | Tuna-2 (7B) | Tuna-R | Tuna | 其他 UMM |
> |-----------|-------------|--------|------|----------|
> | GenEval | Competitive | — | — | — |
> | DPG-Bench | Competitive | — | — | — |
> | LLM-Judge Diversity | ✓ preferred | — | — | — |
> 
> LLM-Judge 使用 GPT-5.4 和 Claude Opus 4.7 评估，Tuna-2 在图像多样性上优于 Tuna-R。

^table-generation

### 3.3 图像编辑（Image Editing）

> [!abstract]- 表：图像编辑 Benchmark 对比（ImgEdit）
> | 方法 | ImgEdit |
> |------|---------|
> | Tuna-2 | Strong, 优于 OmniGen 和 BAGEL |
> | Tuna / Tuna-R | 略优于 Tuna-2 |
> | OmniGen | — |
> | BAGEL | — |
> 
> Tuna-2 在图像编辑上表现强劲，超越 OmniGen 和 BAGEL，但略逊于 Tuna/Tuna-R。

^table-editing

### 3.4 图像重建（Image Reconstruction）

经过轻量级 finetuning 后，Tuna-2 在重建质量上可媲美专用 VAE。

> [!abstract]- 表：图像重建质量对比
> | 指标 | Tuna-2 (finetuned) | 专用 VAE |
> |------|-------------------|----------|
> | rFID | Top-tier | Comparable |
> | PSNR | Top-tier | Comparable |
> | SSIM | Top-tier | Comparable |
> 
> Tuna-2 经过 lightweight finetuning 后，重建质量（rFID, PSNR, SSIM）可媲美专门训练的 VAE。

^table-reconstruction

### 3.5 Encoder-Free vs Encoder-Based 对比

![](https://arxiv.org/html/2604.24763v1/x4.png)

*Figure 4: Attention 可视化对比。Tuna-2 产生更准确鲁棒的 attention maps，即使在误导性文本或视觉上下文下也能正确 localize 物体。*

> [!abstract]- 表：Tuna-2 vs Tuna-R 关键对比
> | 维度 | Tuna-2 (Encoder-Free) | Tuna-R (Encoder-Based) |
> |------|----------------------|----------------------|
> | 理解能力（规模化后） | ✓ 超越 Tuna-R，尤其细粒度任务 | 初期收敛更快 |
> | 生成能力 | SFT 后差距缩小 | 初期略优（semantic priors） |
> | 训练复杂度 | 更简单，无需 alignment stage | 需 connector alignment |
> | 跨模态对齐 | Attention maps 更准确鲁棒 | — |

^table-encoder-comparison

### 3.6 Masking 消融实验

> [!abstract]- 表：Masking 策略消融
> | 条件 | 理解 | 生成 |
> |------|------|------|
> | 有 Masking | ✓ 提升 | ✓ 提升 |
> | 无 Masking | — | — |
> 
> 消融实验表明，Masking-Based Feature Learning 对理解和生成均有稳定且一致的提升。

^table-masking-ablation

### 3.7 Data Ratio 消融

> [!abstract]- 表：数据比例消融
> | Generation:Understanding 比例 | 效果 |
> |-------------------------------|------|
> | 7:3 | ✓ 最优 |
> | 其他比例 | 低于 7:3 |
> 
> Generation 与 Understanding 数据的最优比例为 7:3。

^table-data-ratio

---

## 4 结论

Tuna-2 证明了一个重要结论：**pretrained vision encoder 并非统一多模态模型的必要组件**。通过简单的 patch embedding 层将原始像素转换为 token，配合 masking-based feature learning 和统一 transformer decoder，Tuna-2 在理解和生成两个方向上均达到了与 encoder-based 方案相当甚至更优的性能。

核心发现：
1. **Pixel embeddings 可以替代 vision encoder**：无需 VAE、CLIP 或 SigLIP
2. **Masking 是关键技术**：同时提升理解和生成
3. **架构越简洁，规模化潜力越大**：encoder-free 方案在规模化训练后展现出超越 encoder-based 方案的趋势
4. **像素空间表示在细粒度任务上有天然优势**：V*、CountBench、VisuLogic 等像素级任务上优势显著

---

## 5 思考

### 5.1 优点

1. **架构极简**：single transformer decoder + patch embedding，无需任何 pretrained vision encoder，极大简化了模型设计和训练流程
2. **Masking-Based Feature Learning 设计精巧**：一个简单操作同时服务于理解和生成两个目标，且消融实验充分验证了其有效性
3. **系统性消融**：从 Tuna → Tuna-R → Tuna-2 的三代对比非常清晰，每一步改进都有充分实验支撑
4. **像素级任务优势明显**：pixel-space representation 在细粒度理解任务上展现出 encoder-based 方案不具备的能力
5. **训练流程简化**：无需 alignment stage，端到端训练效率更高
6. **重建能力意外强**：finetuning 后可媲美专用 VAE，说明模型学到了很好的像素级表示

### 5.2 缺点

1. **生成质量仍有差距**：虽然 competitive，但与 BAGEL 等专门优化生成的模型相比仍有差距
2. **训练计算成本未知**：从零学习像素表示可能需要更多数据和计算，论文未充分讨论 compute budget
3. **图像编辑略逊于 Tuna/Tuna-R**：说明完全移除 encoder 在某些任务上仍有代价
4. **Masking 策略的最优性未充分探索**：mask ratio、mask 策略的选择是否有更优方案？
5. **缺少与更大规模模型的对比**：7B 的结论是否能推广到更大规模？
6. **长尾理解任务未覆盖**：如 OCR、图表理解等特殊场景的表现未见讨论

### 5.3 Wiki Connections

- **[JiT (Joint Image-Text)](Joint%20Image-Text)**：Tuna-2 的 pixel-space generation 遵循 JiT paradigm，使用 xx-prediction 和 vv-loss
- **[Flow Matching](Flow%20Matching)**：Tuna-2 使用 flow matching 进行生成，而非传统的 diffusion
- **[Vision Encoder](Vision%20Encoder)**：Tuna-2 的核心贡献是证明了可以完全抛弃 vision encoder
- **[CLIP](CLIP)**：Tuna-2 消除了对 CLIP 的依赖
- **[VAE (Variational Autoencoder)](VAE)**：Tuna-2 消除了对 VAE 的依赖，但仍能在 finetuning 后实现高质量重建
- **[Masked Image Modeling](Masked%20Image%20Modeling)**：Tuna-2 的 masking-based feature learning 与 MAE 等方法有思想上的联系
- **[Unified Multimodal Model](Unified%20Multimodal%20Model)**：Tuna-2 是 native UMM 的重要探索
- **[BAGEL](BAGEL)**：主要对比对象，采用 MoT + dual encoders 的不同架构路线
- **[Show-o](Show-o)**：7B native UMM 对比基线
- **[Janus-Pro](Janus-Pro)**：7B native UMM 对比基线
- **[Patch Embedding](Patch%20Embedding)**：ViT 中的基础操作，Tuna-2 将其作为唯一的视觉输入方式
- **[Flow Matching in Pixel Space](Pixel-Space%20Generation)**：直接在像素空间而非 latent space 执行 flow matching 的范式

### 5.4 与同期统一多模态模型的对比

| 维度 | Tuna-2 | BAGEL | OmniGen2 | Lumina-DiMOO | UniWorld-V1 |
|---|---|---|---|---|---|
| **架构** | 单 Transformer，无编码器 | MoT（双专家共享注意力） | VLM+DiT（解耦） | 单 Transformer（离散扩散） | VLM+SigLIP+FLUX |
| **视觉编码** | 无（raw pixel patches） | SigLIP2 + FLUX VAE | ViT + FLUX VAE | aMUSEd-VQ（离散） | SigLIP2（语义） |
| **生成范式** | Flow Matching（像素空间） | Rectified Flow（latent） | Flow Matching（latent） | 离散扩散（masked token） | Flow Matching（latent） |
| **参数** | 7B | 7B/14B | 3B+4B | ~8B | 7B |
| **核心优势** | 架构最简，像素级理解强 | 交错数据涌现 | RL 对齐，GenEval SOTA | 32x 速度，inpainting | 数据效率 1/600 |
| **核心劣势** | 生成质量略逊 | 训练成本高 | VLM 冻结 | 量化损失 | 模块化瓶颈 |

**关键洞察**：
- **编码器是最大的架构分歧**：Tuna-2 完全不要，BAGEL 用两个，OmniGen2/UniWorld 用一个。哪种最优取决于任务——像素级理解选 Tuna-2，生成质量选 BAGEL，指令对齐选 OmniGen2。
- **生成范式的收敛**：除 Lumina-DiMOO 用离散扩散外，其他都用 Flow Matching/Rectified Flow。连续扩散在生成质量上仍有优势。
- **训练策略的分化**：BAGEL 靠数据 scaling，OmniGen2 靠 RL alignment，Tuna-2 靠架构简化。三者互补而非互斥。
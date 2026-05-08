---
type: source
status: complete
created: '2026-05-07'
updated: '2026-05-07'
tags: [multimodal, native, next-token-prediction, unified-model, emu]
sources:
  - "[[Clippings/Emu3: Next-Token Prediction is All You Need]]"
---

# Emu3 原生多模态模型

## 基本信息

- **标题**：Emu3: Next-Token Prediction is All You Need
- **作者**：Emu3 Team, BAAI（项目负责 Xinlong Wang，核心贡献者包括 Xiaosong Zhang、Zhengxiong Luo、Quan Sun、Yufeng Cui 等）
- **机构**：北京人工智能研究院（BAAI）
- **发表时间**：2024（arXiv: 2409.18869）
- **开源**：开放关键技术与模型权重

## 核心论点

Next-token prediction 是通往通用多模态智能的可行路径，无需依赖扩散模型或组合式架构（如 CLIP + LLM）。通过将图像、文本和视频统一离散化为 token，使用单一 Transformer 从零训练，即可在生成和感知任务上同时达到或超越专用模型的性能。

这一范式的核心价值在于**极大简化了多模态模型的设计复杂度**——只需关注 token，即可释放训练和推理阶段的扩展潜力。

## 关键技术方法

### 统一 Next-Token Prediction 范式

- 图像、视频和文本均被离散化为 token，在同一个 Transformer decoder 中以自回归方式统一训练
- 不使用扩散模型（diffusion）、不使用 CLIP 等视觉编码器，完全依赖 next-token prediction
- 模型架构沿用 Llama-2 框架，使用 RMSNorm、GQA、SwiGLU、RoPE，词表扩展至 184,622 以容纳视觉 token
- 模型规模：8B 参数，32 层，隐藏维度 4096，上下文长度 131,072

### 视觉 Tokenizer

- 基于 SBER-MoVQGAN 构建，codebook 大小 32,768
- 可将 4x512x512 视频片段或 512x512 图像编码为 4,096 个离散 token
- 压缩比：时间维度 4x，空间维度 8x8
- 在编码器和解码器中引入 3D 卷积的时间残差层以增强视频 tokenization 能力
- 在 LAION-High-Resolution 图像集和 InternVid 视频集上端到端训练，使用 L2 + LPIPS + GAN + commitment loss 联合目标

### 数据工程

- **语言数据**：与 Aquila 相同的高质量中英文语料
- **图像数据**：大规模图文数据集，经过分辨率过滤（>=512x512）、美学评分过滤（>=5.5）、文本检测与色彩过滤。使用基于 Emu2 的图像标注模型生成密集合成 caption
- **视频数据**：覆盖风景、动物、植物、游戏、动作等类别。经场景检测、文本过滤、光流过滤和美学评分四阶段清洗。使用基于图像标注器微调的视频标注器

### 训练流程

- **预训练**：两阶段——第一阶段仅文本+图像（上下文 5,120）；第二阶段引入视频（上下文 131,072）。学习率 5e-5，余弦退火。视觉 token 的 loss 权重设为 0.5
- **数据格式**：`[BOS] {caption} [SOV] {meta} [SOT] {vision tokens} [EOV] [EOS]`，meta 包含分辨率、帧率、时长等元信息
- **后训练-生成方向**：质量微调（QFT，使用高美学质量数据，分辨率提升至 720p）+ DPO 对齐人类偏好
- **后训练-理解方向**：image-to-text 训练 + 指令微调两阶段

### DPO 用于自回归视觉生成

- 为每个 prompt 生成 8-10 个候选，由三位评估者从视觉吸引力和 prompt 对齐度打分
- 构建 (prompt, chosen, rejected) 三元组进行 DPO 训练
- 存储 token 化结果避免重编码差异

## 主要结果

### 图像生成

- 在人类评估中超越 SDXL，与 DALL-E 3 和 MJ-v5.2 持平
- 自动评估：MSCOCO-30K CLIP-I 0.689、CLIP-T 0.313；GenEval 0.66（with rewriter）；DPG-Bench 81.60（Emu3-DPO），超越 SDXL 和 PixArt-alpha，接近 DALL-E 3
- 完全不依赖预训练语言模型或 CLIP

### 视频生成

- 原生支持 5 秒 24 FPS 视频生成，可通过自回归方式无限扩展
- VBench 总分 80.96，超越多数开源视频扩散模型，仅次于 Kling 和 Gen-3 等商业模型
- 支持视频扩展（未来预测）：以 2 秒视频为上下文，可扩展生成后续 8 秒

### 视觉语言理解

- 纯 encoder-free 方法，在多个 benchmark 上超越 encoder-based 对手
- SEEDBench-Img 68.2、OCRBench 687、ScienceQA-Img 89.2、ChartQA 68.6、DocVQA 76.3
- 不依赖 CLIP 或专用预训练 LLM，展示了 decoder-only 架构在多模态理解上的内在能力

## 局限性

- 论文未详细讨论模型在复杂推理、长视频理解和生成一致性方面的边界
- 视觉 tokenizer 在高压缩比下存在信息损失（论文附录的重建指标表明仍有改进空间）
- DPO 后自动评估指标略有下降（可能因偏好数据侧重美学质量，与自动评估模型的评价域不一致），说明评估方法与人类偏好之间存在 gap
- 视频生成后处理（稳定化和超分辨率）依赖额外训练的专用模型，非端到端
- 模型规模为 8B，未展示更大规模下的 scaling 行为

## 与相关工作的关系

### Emu / Emu2

Emu3 是 Emu 系列的第三代。Emu 和 Emu2 已探索统一自回归多模态目标，但 Emu/Emu2 要么连接 LLM 与扩散模型（Emu），要么在生成性能上不及专用方法。Emu3 首次证明纯 next-token prediction 可以同时超越 SDXL 和 LLaVA-1.6。

### Emu3.5

Emu3.5（后续工作）在 Emu3 基础上进一步扩展，引入更高效的视觉 tokenizer 和更大规模训练，代表该方向的持续演进。

### Show-o / Show-o2

Show-o 尝试将扩散与自回归方法结合，属于"autoregressive meets diffusion"路线。Emu3 的结果表明纯自回归路线在无需扩散的情况下即可达到竞争性能，两条路线的优劣仍有待进一步比较。Show-o2 进一步探索了统一多模态生成与理解。

### Chameleon

Meta 的 Chameleon 训练了基于 token 的自回归模型处理混合图文数据，但 MSCOCO FID 仅 26.74，远不及 Emu3。Emu3 在 vision tokenizer 和训练数据工程上的改进是关键差距来源。

### TransFusion

TransFusion 将扩散和自回归目标结合在同一模型中。Emu3 的纯自回归方案在 GenEval 上达到 0.66，优于 TransFusion 的 0.63。

### LlamaGen

LlamaGen 专注于图像生成的自回归模型，但 GenEval 仅 0.32，说明仅靠图像生成不足以与 Emu3 的统一多模态训练竞争。

### Lumina-DiMOO

Lumina 系列（包括 Lumina-Next 和 Lumina-DiMOO）走扩散 Transformer 路线。Emu3 在 DPG-Bench 上的 81.60 与 Lumina-Next 的 74.63 形成对比，展示了自回归范式在长 prompt 跟随上的优势。

### Sora

Sora 使用视频扩散模型从噪声生成视频，而 Emu3 通过因果预测下一 token 生成视频，代表了视频生成的两种不同范式。

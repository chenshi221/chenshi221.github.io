---
type: source
status: complete
created: '2026-05-07'
updated: '2026-05-07'
tags: [image-generation, qwen, alibaba, diffusion, multimodal, text-rendering, image-editing]
sources:
  - "[[Clippings/Qwen-Image Technical Report]]"
---

# Qwen-Image 技术报告

## 基本信息

- **标题**: Qwen-Image Technical Report
- **作者**: Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan 等（核心贡献者 14 人，参与者 27 人）
- **机构**: 阿里巴巴 Qwen 团队
- **年份**: 2025
- **论文**: arXiv:2508.02324
- **代码**: https://github.com/QwenLM/Qwen-Image
- **模型**: https://huggingface.co/Qwen/Qwen-Image, https://modelscope.cn/models/Qwen/Qwen-Image

## 核心论点

1. **复杂文字渲染能力突破**: Qwen-Image 在多行文本、段落级语义和细粒度文字渲染方面取得显著进展，同时支持字母语言（英文）和意音文字语言（中文），在中文文字生成上大幅超越现有 SOTA。
2. **图像编辑一致性增强**: 通过改进的多任务训练范式（T2I + TI2I + I2I）和双编码机制（Qwen2.5-VL 语义特征 + VAE 重建特征），在编辑时同时保持语义一致性和视觉保真度。
3. **统一的理解-生成架构**: 基于 Qwen2.5-VL 作为条件编码器，MMDiT 作为扩散骨干，将视觉理解和生成能力统一在同一框架中。
4. **数据工程为核心竞争力**: 构建了大规模、多阶段的数据管线（采集、过滤、标注、合成、平衡），强调数据质量和分布平衡而非单纯规模。
5. **生成模型可承担理解任务**: 通过图像编辑范式统一了深度估计、新视角合成等传统视觉理解任务，展示了生成式框架的通用性。

## 关键技术方法

### 整体架构

Qwen-Image 由三个核心组件构成：

- **条件编码器**: 冻结的 Qwen2.5-VL（7B 参数），负责提取文本和图像输入的特征。选择 Qwen2.5-VL 而非纯语言模型的原因：(1) 视觉-语言空间已对齐；(2) 保留强语言建模能力；(3) 支持多模态输入，赋能图像编辑。
- **图像分词器**: VAE（编码器 54M，解码器 73M），采用单编码器-双解码器架构（兼容图像和视频），基于 Wan-2.1-VAE 冻结编码器并微调图像解码器。在含丰富文字的内部语料上训练，仅使用重建损失和感知损失。
- **扩散骨干**: MMDiT（20B 参数，60 层），采用 Rectified Flow 训练目标。引入新的位置编码方案 MSRoPE（Multimodal Scalable RoPE），将文本视为沿图像对角线拼接的 2D 张量，兼顾图像分辨率缩放和文本位置编码。

### 数据管线

- **四域数据集**: Nature（约 55%）、Design（约 27%，含海报/UI/幻灯片/艺术）、People（约 13%）、Synthetic Data（约 5%，通过文字渲染技术合成，非 AI 生成图像）。
- **七阶段过滤管线**: 从初始预训练数据筛选（Stage 1）到平衡多尺度训练（Stage 7），逐步提升数据质量、对齐度和多样性。
- **文字合成策略**: Pure Rendering（纯文字渲染）、Compositional Rendering（嵌入真实场景）、Complex Rendering（基于模板的结构化渲染），覆盖低频字符和混合语言场景。

### 训练策略

- **渐进式课程学习**: 分辨率从 256p 到 640p 再到 1328p；从无文字到有文字；从大规模粗数据到精筛数据；从不平衡到平衡分布；从真实数据到合成数据增强。
- **Producer-Consumer 框架**: 基于 Ray 的异步数据加载，Producer 负责 VAE 编码和数据 IO，Consumer 专注 GPU 训练，使用 Megatron-LM 的混合并行策略（数据并行 + 张量并行）。
- **后训练**: SFT（层次化语义类别数据集 + 人工标注）+ RL（DPO 大规模离线偏好学习 + GRPO 细粒度在线策略优化，基于 Flow-GRPO 框架）。

### 多任务图像编辑

- 将原始图像同时输入 Qwen2.5-VL（语义特征）和 VAE 编码器（重建特征），双编码机制使编辑模块在语义一致性和视觉保真度之间取得平衡。
- MSRoPE 扩展了帧维度以区分编辑前后的图像。
- 统一支持指令编辑、新视角合成、深度估计等任务。

## 主要结果

### 图像生成

- **DPG**: 总体得分 88.32，超越 Seedream 3.0（88.27）和 GPT Image 1 High（85.15）。
- **GenEval**: 基础模型 0.87，RL 增强后 0.91，是唯一突破 0.9 阈值的模型。
- **OneIG-Bench**: 英文和中文赛道均取得最高总体分（英文 0.539，中文 0.548），Alignment 和 Text 维度均排名第一。
- **TIIF Bench**: 总体排名第二，仅次于 GPT Image 1。
- **AI Arena（人类评估）**: 在 Elo 排名中位居第三，领先 GPT Image 1 High 和 FLUX.1 Kontext Pro 超过 30 Elo 分。

### 文字渲染

- **CVTG-2K（英文）**: Word Accuracy 0.8288，NED 0.9116，与 GPT Image 1 接近。
- **ChineseWord（中文）**: Level-1 准确率 97.29%，总体 58.30%，大幅超越 GPT Image 1（36.14%）和 Seedream 3.0（33.05%）。
- **LongText-Bench**: 中文长文本 0.946（第一），英文长文本 0.943（第二，仅次于 GPT Image 1 的 0.956）。

### 图像编辑

- **GEdit-Bench**: 英文 Overall 7.56（第一），中文 Overall 7.52（第一）。
- **ImgEdit**: 总体 4.27（第一），在 Extract 和 Remove 子任务上优势明显。
- **新视角合成（GSO）**: PSNR 15.11，SSIM 0.884，LPIPS 0.153，与专用 3D 模型 CRM 相当。
- **深度估计**: 在多个零样本数据集上与专用模型 Depth Anything v2 性能相当。

### VAE 重建

- Qwen-Image-VAE 在 ImageNet 和文字丰富图像上的 PSNR/SSIM 均为 SOTA，且仅激活 19M 编码器参数和 25M 解码器参数。

## 局限性

- 论文未详细公开训练数据规模、训练算力和推理成本。
- 中文 Level-3（1605 个低频汉字）准确率仅 6.48%，说明低频字仍有较大提升空间。
- 在 OneIG-Bench 的 Reasoning 和 Diversity 维度上得分不如 Alignment 和 Text，表明推理能力和多样性方面存在改进空间。
- 人类评估中排除了中文提示以维持公平比较，未能全面展示中文能力优势。
- 深度估计等理解任务虽有竞争力但未超越专用判别模型，说明生成式理解仍有上限。

## 与相关工作的关系

### Seedream 系列（字节跳动）

- 架构上同属 MMDiT 范式。Seedream 3.0 引入 Scaling RoPE 将图像位置编码移至中心区域，但文本和图像某些行位置编码同构。Qwen-Image 的 MSRoPE 通过将文本放在对角线位置解决了这一问题。
- 在 DPG、OneIG-Bench 等基准上互有胜负，但 Qwen-Image 在中文文字渲染上大幅领先。
- Seedream 3.0 的 ChineseWord Level-1 准确率仅 53.48%，而 Qwen-Image 达 97.29%。

### FLUX 系列（Black Forest Labs）

- 同为 MMDiT 架构，FLUX.1 Kontext Pro 在图像编辑中引入 VAE 嵌入保持一致性。Qwen-Image 借鉴了这一思路，同时输入 Qwen2.5-VL 语义特征和 VAE 重建特征。
- FLUX.1 Kontext Pro 不支持中文编辑（GEdit-Bench-CN 上得分仅 1.23），Qwen-Image 在中英文编辑上均表现强劲。
- 在 AI Arena 中 Qwen-Image 领先 FLUX.1 Kontext Pro 约 30 Elo 分。

### GPT Image 1（OpenAI）

- 在 AI Arena 中排名高于 Qwen-Image（约 30 Elo 差距），在 TIIF Bench 上也领先。
- 但中文文字渲染能力远不如 Qwen-Image（ChineseWord 36.14% vs 58.30%，LongText-Bench-ZH 0.619 vs 0.946）。
- 在图像编辑（GEdit、ImgEdit）中与 Qwen-Image 总体接近，但 Qwen-Image 在 Remove 等特定任务上更优。

### Qwen-VL / Qwen2.5-VL

- Qwen-Image 直接使用 Qwen2.5-VL 作为条件编码器，是 Qwen 系列在视觉生成方向的首个工作。
- 论文将 Qwen2.5-VL 的理解能力与 Qwen-Image 的生成能力视为"理解-生成统一"的两个支柱，为未来的 Visual-Language Omni 系统奠定基础。

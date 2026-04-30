---
title: 'SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features'
authors:
  - Michael Tschannen
  - Alexey Gritsenko
  - Xiao Wang
  - Muhammad Ferjad Naeem
  - Ibrahim Alabdulmohsin
  - Nikhil Parthasarathy
  - Talfan Evans
  - Lucas Beyer
  - Xiaohua Zhai
institutions: Google DeepMind
aliases:
  - SigLIP 2
  - SigLIP2论文
date: '2026-04-30'
publish_date: 2025-02
venue: arXiv:2502.14786
tags:
  - 论文
  - 多模态
  - 视觉编码器
  - 对比学习
  - 多语言
  - 密集预测
url: 'https://arxiv.org/abs/2502.14786'
code: 已开源（ViT-B/L/So400m/g 四个尺寸）
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：SigLIP 2 在原始 SigLIP 的基础上系统性地整合了 decoder-based pretraining（LocCa）、自监督损失（SILC/TIPS 的 self-distillation + masked prediction）和在线数据筛选（ACID），并引入多语言训练和去偏过滤，在零样本分类、检索、VLM 特征提取、定位任务和密集预测上全面超越 SigLIP，同时支持原生宽高比的 NaFlex 变体。

![](https://arxiv.org/html/2502.14786v1/x1.png)

*Figure 1: SigLIP 2 的训练配方——将 SigLIP 的 sigmoid loss 与 LocCa 的 captioning-based pretraining、SILC/TIPS 的 self-distillation 和 masked prediction 相结合。*

---

## Intro

### Motivation

CLIP/SigLIP 已成为视觉语义理解的主流方法，但现有开源模型普遍沿用 CLIP 原始方案，未整合近年来的多项独立改进（re-captioning、self-supervised losses、decoder-based pretraining、online data curation）。同时，它们在定位（localization）和密集特征提取（dense features）上明显弱于专用模型。SigLIP 2 的目标是将这些独立进展整合进一个统一配方。

### 贡献

1. 将 SigLIP + LocCa + SILC/TIPS + ACID 整合为统一训练配方，四个尺寸（B/L/So400m/g）全面超越 SigLIP
2. 训练多语言模型（109 种语言，90% 英语 + 10% 非英语），在 XM3600 多语言检索上大幅提升
3. 引入 NaFlex 变体：单个 checkpoint 支持多种分辨率和原生宽高比
4. 应用去偏过滤技术（CLIP the Bias），显著降低性别表征偏差（L/16 从 35.5% 降至 7.3%）
5. 向后兼容 SigLIP 架构，现有用户可直接替换权重

---

## Method 核心方法

### 1. 架构和训练基础

- **架构**：标准 ViT + MAP head（attention pooling），文本塔使用 Gemma tokenizer（256K vocab），文本长度 64
- **数据**：WebLI（10B 图像 + 12B alt-texts，109 种语言），90% 英语 + 10% 非英语
- **优化**：Adam，lr=1e-3，weight decay 1e-4，batch size 32K，cosine schedule + 20K warmup，总共 40B examples，最多 2048 TPUv5e

### 2. 分阶段训练配方

**第一阶段（前 80%）**：SigLIP（sigmoid loss）+ LocCa
- LocCa decoder 做三个辅助任务：image captioning、automatic referring expression prediction、grounded captioning
- Captioning 以 50% 概率使用 parallel prediction（无 causal mask）

**第二阶段（后 20%）**：加入 SILC/TIPS 的自监督损失
- local-to-global consistency loss：学生（局部视图）→ 匹配教师（全局视图）的全局表征
- masked prediction loss：50% patch 被 mask，学生匹配教师在 mask 位置的特征
- 教师用 EMA 更新，权重 1.0 和 0.25，不同尺寸进一步重新加权

**第三阶段（后 5%）**：分辨率适配
- 固定分辨率：resize positional embedding，恢复训练
- NaFlex：90% 处切换为 aspect-preserving resize，均匀采样序列长度 {128, 256, 576, 784, 1024}

**蒸馏阶段**：ACID（Active Data Curation via Implicit Distillation）
- 仅用于 B 尺寸模型
- 用 So400m teacher 在 curated data 上微调后的版本作为 teacher
- 每步从 64K super-batch 中选 32K 最优样本（filtering ratio 0.5）

### 3. NaFlex 变体

- 结合 FlexiViT（多序列长度）+ NaViT（原生宽高比）
- 输入图像 resize 使得宽高是 patch size 的倍数且序列长度不超过目标值
- 可学习位置编码被 bilinear resize 到目标非方形 patch grid
- 单个 checkpoint 在多数检索 benchmark 上与各自适配的标准变体相当或更优

---

## 实验/评估/结果

### 零样本分类和检索

在 ImageNet/ObjectNet/ImageNet-v2/ImageNet ReaL 以及 COCO/Flickr 检索上全面超越 SigLIP 和其他开源模型（OpenCLIP、EVA-CLIP、MetaCLIP、DFN）。多语言检索 XM3600 上，SigLIP 2 接近专门的 mSigLIP 水平，同时英语任务远优于 mSigLIP。

![](https://arxiv.org/html/2502.14786/x2.png)

*Figure 2: Crossmodal-3600 上的各语言图像-文本检索性能对比——SigLIP 2 几乎达到 mSigLIP 的多语言水平，同时在英语任务上大幅领先。*

### VLM 特征提取

将 SigLIP 2 作为 PaliGemma 风格 VLM 的 vision encoder（搭配 Gemma 2 2B），在 30+ 个下游任务上一致优于 SigLIP 和 AIMv2。

### 密集预测

- **语义分割**（PASCAL/ADE20k）：So/14@224 上 77.1/41.8（SigLIP 72.0/37.6）
- **深度估计**（NYUv2/NAVI）：RMSE 显著降低
- **开放词汇分割**（Cat-Seg）：L/16 超越更大的 OpenCLIP G/14

### 定位任务

- **Referring Expression Comprehension**（RefCOCO/+/g）：B/16@256 上从 64.05 提升至 83.76（+19.7），提升巨大
- **开放词汇检测**（OWL-ViT）：COCO AP 42.2→42.8，LVIS AP 33.0→34.4

### 文化多样性与公平性

- GeoDE 10-shot geolocalization：L/16@256 从 36.2% 提升至 44.4%
- 性别表征偏差：L/16@256 从 35.5% 降至 7.3%

![](https://arxiv.org/html/2502.14786/x6.png)

*Figure 6: 性别表征偏差对比（越低越好）——SigLIP 2 通过去偏过滤技术将性别偏差从 35.5% 大幅降至 7.3%。*

---

## 结论

SigLIP 2 通过系统整合 decoder-based pretraining、自监督损失和在线数据筛选，在分类、检索、VLM 特征提取、定位和密集预测上全面超越 SigLIP。多语言训练和去偏过滤使其更具文化包容性。NaFlex 变体以单个 checkpoint 支持多分辨率/原生宽高比。

---

## 思考

### 优点

1. **工程化的系统整合**：将 LocCa、SILC/TIPS、ACID 等独立工作整合为统一配方，且证明这些技术可以叠加产生正向收益。这对工业级视觉编码器的设计有直接指导意义。

2. **密集特征和定位的大幅提升**：这是传统 CLIP 类模型的明显短板。SigLIP 2 在 RefCOCO 上 +20 点的提升尤其惊人，证明了 decoder-based pretraining 对空间理解的价值。

3. **NaFlex 的实用设计**：单 checkpoint 支持多分辨率 + 原生宽高比，在文档/OCR 场景有实际价值。与每个分辨率训练独立 checkpoint 相比节省了大量成本。

4. **公平性考量**：系统性评估了文化多样性和性别偏差，并展示了去偏技术的实际效果。这在视觉编码器论文中仍不多见。

### 缺点与待解决问题

1. **不是端到端创新**：本质上是对已有技术的系统整合，主要贡献在工程层面而非方法论层面。不过考虑到工业级模型发布的稀缺性，这仍然是高价值的工作。

2. **Decoder 仅用于训练**：LocCa decoder 只用于 representation learning，最终发布时不包含。这减少了推理成本，但也限制了用户直接利用其 captioning/localization 能力。

3. **NaFlex 的局限性**：不支持训练分辨率之间的插入——仅对训练时见过的序列长度（64, 144, 256, 576, 784, 1024 等）效果较好。

4. **训练成本未详细披露**：未报告总 GPU/TPU 小时，对社区评估技术路线可行性不够透明。

### 与已有 Wiki 的连接

- 关联概念：[[SigLIP]]、[[CLIP]]、[[NaViT]]、[[FlexiViT]]、[[LocCa]]、[[SILC]]
- 关联比较：[[SigLIP vs SigLIP 2]]
- 关联问题：VL encoder 的技术路线——是集中做对比学习还是整合更多辅助任务？

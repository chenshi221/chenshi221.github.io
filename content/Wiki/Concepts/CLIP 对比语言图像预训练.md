---
type: concept
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - Contrastive Language-Image Pre-training
  - CLIP模型
tags:
  - CLIP
  - 多模态
  - 对比学习
  - 零样本
  - 视觉语言
  - OpenAI
sources:
  - '[[Wiki/Sources/CLAP]]'
  - '[[Wiki/Sources/SigLIP 2]]'
  - '[[Wiki/Sources/ViT]]'
confidence: high
---
# CLIP 对比语言图像预训练

## 定义

**CLIP（Contrastive Language-Image Pre-training，对比语言图像预训练）** 是 OpenAI 于 2021 年（Radford et al., ICML 2021）提出的视觉-语言预训练模型。它通过**对比学习**将图像和文本映射到共享嵌入空间，实现了**零样本图像分类**和强大的视觉-语言对齐能力。

CLIP 是现代多模态 AI 的基石之一——几乎所有后续的视觉-语言模型（从 DALL-E 2 到 GPT-4o）都直接或间接建立在其范式之上。

## 核心设计

### 架构：双塔 + 对比学习

```
图像编码器 (ViT/ResNet)   文本编码器 (Transformer)
        ↓                        ↓
    I₁, I₂, ..., I_N         T₁, T₂, ..., T_N
        ↓                        ↓
    线性投影                  线性投影
        ↓                        ↓
    ┌─────────────────────────────────────┐
    │       共享嵌入空间 (d-dim)           │
    │  sim(I_i, T_i) 高（正对）           │
    │  sim(I_i, T_j) 低（负对）           │
    └─────────────────────────────────────┘
```

### 训练方法

1. **数据**：4 亿图文对，从互联网收集（WIT - WebImageText）
2. **Batch 构建**：N 个图文对 → N 个正对（对角线），N²-N 个负对
3. **损失函数**：对称 InfoNCE loss
   - 图像→文本方向：`-log(exp(sim(I_i, T_i)/τ) / Σ_j exp(sim(I_i, T_j)/τ))`
   - 文本→图像方向：对称计算
   - 总 loss = (图像方向 + 文本方向) / 2
4. **温度参数 τ**：可学习的 scaling factor，控制分布的锐度

### 模型变体

| 模型 | 图像编码器 | 参数量 | GFLOPs |
|------|-----------|--------|--------|
| CLIP RN50 | ResNet-50 | 102M | 18.6 |
| CLIP RN101 | ResNet-101 | 120M | 25.2 |
| CLIP ViT-B/32 | ViT-B | 151M | 9.9 |
| CLIP ViT-B/16 | ViT-B | 150M | 33.0 |
| CLIP ViT-L/14 | ViT-L | 428M | 170.3 |
| CLIP ViT-L/14@336px | ViT-L | 428M | 459.0 |

## 关键能力

### 零样本分类

将分类转化为图文匹配问题：

1. 将类别标签模板化为文本（如 "a photo of a {class}"）
2. 编码所有类别的文本 → 得到文本 embedding 矩阵
3. 编码待分类图像 → 得到图像 embedding
4. 计算图像与每个文本的相似度 → 取最高

**结果**：CLIP ViT-L/14 在 ImageNet 上 76.2% 零样本准确率，媲美全监督 ResNet-50

### 强大的泛化

- 对**分布偏移（distribution shift）极其鲁棒**：在 sketch、painting、cartoon 等 OOD 数据集上远超监督模型
- 22 个数据集平均：CLIP 超过全监督 ResNet-50，且不需要任何微调

### 多模态表示

- 图像和文本在同一个语义空间中，可以直接计算跨模态相似度
- 为图像搜索、跨模态检索、条件生成等任务提供统一的表示基础

## CLIP 的局限

1. **结构化理解弱**：无法理解空间关系、计数、逻辑（如"左边的红色方块"）
2. **文本编码器浅**：CLIP 只用 12 层 512 宽的 Transformer，文本理解能力有限
3. **细粒度识别差**：在细粒度分类（如鸟种识别）上不如专门模型
4. **数据效率低**：需要极大规模数据才能学到好的表示
5. **语义泄漏**：训练数据量大，可能在零样本评估中见过测试类（zero-shot 不纯）

## CLIP 的影响与衍生

### 图像生成

- **DALL-E 2**：用 CLIP 嵌入作为生成条件
- **Stable Diffusion**：用 CLIP 文本编码器生成 prompt embedding
- 成为现代 T2I 模型的标准文本编码器

### 多模态对比学习

- **SigLIP**（Google）：用 sigmoid loss 替代 softmax，效果更好
- **SigLIP 2**：多语言 + 定位 + dense feature 全面升级
- **CLAP**：将 CLIP 范式迁移到音频领域
- **EVA-CLIP**：改进训练配方，效率大幅提升

### Embedding 模型

- **Magic-MM-Embedding**、**ObjEmbed**、**RzenEmbed** 等 MLLM Embedding 模型都建立在 CLIP 对齐范式的基础上

## 与已有 Wiki 的连接

- 关联概念：[[多模态对比学习]]、[[多模态 Embedding 模型]]
- 关联实体：[[Vision Transformer (ViT)]]、[[扩散模型原理]]
- 所在主题：[[Wiki/Topics/Vision Transformer 演进]]、[[Wiki/Topics/多模态 Embedding 与检索]]

## 深度分析

### CLIP 最被低估的贡献不是技术，是方法论

CLIP 的技术核心（对比学习、双塔架构）在当时并不是全新的。它真正的贡献在于证明了：**用足够大且多样化的自然语言监督替代人工标注，可以训练出泛化能力远超监督学习的模型**。

这是一个范式的转移。在 CLIP 之前，CV 社区的默认假设是"标注越精细，模型越好"。CLIP 用互联网噪声数据训练出的模型，在 22 个下游数据集上打败了在 ImageNet 精心标注上训练的 ResNet。它开启的"规模+自然语言监督"路线，至今仍是多模态预训练的主流范式。

### CLIP 的"存在性证明"价值

CLIP 更大的意义也许是作为一个 **存在性证明**：它证明了语言可以作为视觉的通用监督信号。这个证明直接催生了：
- 用语言作为条件生成图像（DALL-E, Stable Diffusion）
- 用语言作为视觉任务的指令（视觉问答, Grounding）
- 用语言作为跨模态的桥梁（audio, video, 3D）

从这个角度看，CLIP 不是"一个模型"，而是"一扇门"——它打开了语言和视觉真正融合的可能性。

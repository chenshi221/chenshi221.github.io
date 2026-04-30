---
type: entity
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - ViT
tags:
  - vision-transformer
  - model
  - image-classification
  - google
sources:
  - '[[Wiki/Sources/ViT]]'
confidence: high
---
# Vision Transformer (ViT)

## 基本信息

- 全称：Vision Transformer
- 论文：[[Wiki/Sources/ViT]]（An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale）
- 作者：Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov 等（Google Research, Brain Team）
- 发表：ICLR 2021
- 类型：视觉基础模型

## 核心架构

ViT 将标准 NLP Transformer Encoder 几乎零修改地应用于图像：

1. **Patch Embedding**：将 HxW 图像切分为 N 个 PxP patch（如 16x16），每个 patch 展平为向量并线性投影到 D 维嵌入空间。
2. **Position Embedding**：为每个 patch 添加可学习的 1D 或 2D 位置编码。
3. **Class Token**：在序列前添加一个可学习的 [CLS] token，其最终输出用于分类。
4. **Transformer Encoder**：标准的多层 Transformer（多头自注意力 + MLP + LayerNorm + 残差连接）。
5. **Classification Head**：仅取 [CLS] token 的输出通过单层 MLP 做分类。

## 关键特性

- **CNN-free**：没有卷积、没有池化，仅靠自注意力处理图像。
- **数据 hungry**：在 ImageNet-1K 上不如 ResNet；在 ImageNet-21K 或 JFT-300M 预训练后超越 CNN。
- **全局感受野**：从第一层起每个 patch 都能关注所有其他 patch（Swin 等后续工作对此做了修改，限制为局部注意力）。
- **预训练范式**：完全平移自 BERT/GPT 的大规模预训练 + 下游微调范式。

## 影响力

- 开启了 Vision Transformer 研究热潮，后续工作超过数千篇。
- 成为 CLIP、SigLIP、DINO 等视觉-语言模型和多模态模型的标准视觉编码器。
- 启发了 DiT（Diffusion Transformer），将 Transformer 引入扩散模型的 backbone 设计。

## 局限

- 固定 patch 分辨率不适合多尺度视觉任务。
- 全局自注意力的 O(N²) 复杂度限制了高分辨率输入。
- 在小数据上缺乏归纳偏置导致性能不佳。

## 关联

- 主题：[[Wiki/Topics/Vision Transformer 演进]]
- 后续演进：[[Wiki/Entities/Swin Transformer]]（层级化 ViT）
- 概念极限：[[Wiki/Sources/MLP-Mixer]]（"CNN 和 Attention 都不必需"）
- 多模态延伸：[[Wiki/Concepts/多模态对比学习]]

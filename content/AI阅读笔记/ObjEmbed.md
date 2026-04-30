---
title: 'ObjEmbed: Towards Universal Multimodal Object Embeddings'
authors:
  - (团队)
institutions: '(未明确)'
aliases:
  - ObjEmbed
date: '2026-04-30'
publish_date: 2025
venue: arXiv
tags:
  - 论文
  - 多模态
  - 目标检测
  - 嵌入
  - 对比学习
  - 开放词汇
url: ''
code: ''
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：ObjEmbed 提出了一种通用多模态目标嵌入框架，基于 Qwen3-VL 骨干，通过 Object + IoU 双 token 嵌入设计（5 个特殊 token）实现目标级表示，配合 WeDetect-Uni 提议生成器和 sigmoid focal loss，在 18 个 benchmark 上统一处理开放词汇检测、指代表达理解和目标检索任务。

---

## Intro

### Motivation

现有目标级嵌入模型存在以下问题：(1) 大多数只能在封闭词汇设定下工作；(2) 开放词汇检测、指代表达理解、目标检索等任务通常需要分别训练专用模型；(3) 缺乏一个统一的目标嵌入空间，使得同一目标可以在不同任务间共享表示。

### 贡献

1. 提出 Object + IoU 双 token 嵌入设计：object token 编码语义身份，IoU token 编码定位质量
2. 5 个特殊 token：object、iou、global、local_text、global_text，统一处理不同粒度的多模态信息
3. WeDetect-Uni 提议生成器：为开放词汇场景生成候选区域
4. Sigmoid focal loss 替代 softmax，适应大规模类别空间
5. 1.3M 训练样本，在 18 个 benchmark 上超越专用模型

---

## Method 核心方法

### 1. 架构设计

- **骨干**：Qwen3-VL，利用其强大的多模态理解和指令遵循能力
- **5 个特殊 token**：
  - `[object]`：编码目标语义身份（类别无关的对象性表示）
  - `[iou]`：编码定位质量（预测框与真值的 IoU）
  - `[global]`：图像级全局语义
  - `[local_text]`：与目标区域相关的局部文本描述
  - `[global_text]`：图像级全局文本描述
- 输入格式：图像 + 文本指令 + 区域标记，输出为 object token 和 iou token 的嵌入

### 2. Object + IoU 双 Token 设计

这是 ObjEmbed 最核心的创新：

- **Object Token**：负责编码目标的语义身份，用于检索、分类和匹配。不同图像中的同一类别目标应有相似的 object embedding。
- **IoU Token**：负责编码定位质量，用于评估检测框的精确度。在开放词汇检测中用于过滤低质量提议。
- 两个 token 的解耦设计允许模型分别优化语义判别力和定位精度。

### 3. WeDetect-Uni 提议生成器

- 为开放词汇检测生成候选区域
- 与 ObjEmbed 联合训练，端到端优化
- 支持任意类别名的开放词汇检测

### 4. 训练

- **损失**：Sigmoid focal loss（代替 softmax CE），适合大规模类别空间和类别不平衡
- **数据**：1.3M 样本，涵盖检测、指代、检索等多种任务
- **温度参数**可学习

---

## 实验/评估/结果

在 18 个 benchmark 上评估，涵盖三大类任务：

### 开放词汇目标检测
- COCO / LVIS 上超越专用 OVD 模型
- Object token 提供强大语义判别力，IoU token 提供精确的定位质量评估

### 指代表达理解（REC）
- RefCOCO/RefCOCO+/RefCOCOg 上表现优异
- 统一 embedding 空间使得文本指代和视觉目标可以直接匹配

### 目标检索
- 跨图像目标检索：同一类别/实例的目标在嵌入空间中聚集
- Object token 展现出强大的细粒度判别能力

### 消融实验
- Object + IoU 双 token 设计 > 单一 token
- Sigmoid focal loss > Softmax CE（在大类别空间下）
- Qwen3-VL 骨干 > 更小的 VL 模型

---

## 结论

ObjEmbed 证明了通过精心设计的 token 化方案（Object + IoU 双 token），可以将 MLLM 转化为通用目标嵌入模型。该方法在开放词汇检测、指代表达理解和目标检索上统一处理，且性能超越专用模型。

---

## 思考

### 优点

1. **Token 设计的精妙**：Object + IoU 双 token 的解耦设计简洁而有效。语义身份和定位质量是两个天然正交的维度，分开建模比混在一起更合理。

2. **统一范式的价值**：一个模型、一个嵌入空间覆盖检测、指代、检索三大类任务，这种统一性在实际系统中价值巨大。

3. **与 MLLM 趋势的契合**：选择 Qwen3-VL 作为骨干，继承了其强大的指令遵循和视觉理解能力，体现了"通用骨干+任务特定 token 设计"的思路。

### 缺点与待解决问题

1. **5 个特殊 token 的设计可能过度工程化**：global、local_text、global_text 等 token 的必要性和各自贡献的消融不够充分。

2. **1.3M 训练样本相对较小**：对于通用目标嵌入这个目标来说，数据规模和多样性可能不足。

3. **实时性未充分讨论**：基于 Qwen3-VL 的推理延迟在实际部署中可能是瓶颈，尤其对于需要实时处理的目标检测场景。

### 与已有 Wiki 的连接

- 关联概念：[[开放词汇目标检测]]、[[MLLM]]、[[Qwen3-VL]]
- 关联比较：与 [[OWL-ViT]]、[[Grounding DINO]] 等开放词汇检测方法的比较

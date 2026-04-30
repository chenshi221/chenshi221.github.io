---
title: 'RzenEmbed: Towards Comprehensive Multimodal Retrieval'
authors:
  - (团队)
institutions: '(未明确)'
aliases:
  - RzenEmbed
date: '2026-04-30'
publish_date: 2025
venue: arXiv
tags:
  - 论文
  - 多模态
  - 检索
  - 嵌入
  - 对比学习
  - MLLM
url: ''
code: ''
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：RzenEmbed 提出了一套面向全面多模态检索的 MLLM 嵌入训练方案，核心创新包括 Hardness-weighted InfoNCE loss（按相似度指数加权）、False negative 缓解策略（余弦相似度阈值过滤）、任务特定可学习温度参数、Embedding prompt 设计以及 LoRA 的 model souping，基于 Qwen2-VL 骨干在两阶段训练下实现多 benchmark SOTA。

---

## Intro

### Motivation

MLLM 嵌入模型在多模态检索中展现出巨大潜力，但现有方法仍面临以下挑战：(1) 标准 InfoNCE loss 对所有负样本一视同仁，hard negatives 未被充分加权；(2) 大规模检索数据中 false negatives（语义相似但标签为负的样本）严重影响训练；(3) 不同任务的最优温度参数不同；(4) prompt 设计对嵌入质量影响显著但缺乏系统优化。

### 贡献

1. Hardness-weighted InfoNCE loss：按余弦相似度指数加权负样本，hard negatives 获得更大梯度
2. False negative 缓解：过滤 cosine similarity 高于阈值的负样本，减少噪声标签干扰
3. 任务特定可学习温度参数：不同检索任务学习各自最优的温度
4. Embedding prompt 设计：系统优化 query 和 candidate 的 prompt 模板
5. Model souping for LoRA：融合多个 LoRA 权重提升泛化性
6. 基于 Qwen2-VL 骨干，两阶段训练

---

## Method 核心方法

### 1. Hardness-weighted InfoNCE Loss

标准 InfoNCE 对所有负样本等权处理。RzenEmbed 提出按难度加权：

$$L = -\log\frac{\exp(s_+/\tau)}{\exp(s_+/\tau) + \sum_i w_i \cdot \exp(s_i^-/\tau)}$$

其中 $w_i = \exp(\alpha \cdot s_i^-)$，$s_i^-$ 是负样本与 query 的余弦相似度。$\alpha$ 控制加权强度：
- 相似度越高的负样本（harder negatives）获得越大的权重
- 指数加权确保 hard negatives 主导梯度更新
- $\alpha=0$ 退化为标准 InfoNCE

### 2. False Negative 缓解

问题：大规模检索数据中，两个语义高度相似但标签为"不相关"的样本构成 false negatives，会破坏嵌入空间。

解决方案：
- 计算所有负样本与 query 的余弦相似度
- 过滤相似度超过阈值 $\theta$ 的负样本（判定为潜在 false negative）
- 仅保留真正无关的负样本参与 loss 计算
- 阈值 $\theta$ 可通过小规模标注数据校准

### 3. 任务特定可学习温度

- 不同检索任务（T2I、I2T、IT2I 等）的最优温度不同
- 为每个任务类别设置独立的可学习温度参数 $\tau_k$
- 训练时共同优化，推理时使用对应任务温度

### 4. Embedding Prompt 设计

- Query 模板：`[任务指令] + [query 内容]`
- Candidate 模板：`[任务指令] + [candidate 内容]`
- 不同任务使用不同的指令前缀
- 系统的 prompt 工程带来了显著的性能提升

### 5. Model Souping for LoRA

- 在两阶段训练中，不同 epoch 的 LoRA 权重各有优势
- 将多个 checkpoint 的 LoRA 权重进行加权平均（model souping）
- 融合后的模型在各任务上泛化性更强
- 类似于权重空间中的 ensemble

### 6. 训练策略

- **骨干**：Qwen2-VL（利用其强大的多模态理解）
- **阶段 1**：大规模多模态对比预训练
- **阶段 2**：任务感知精调，引入 hard negatives 和 false negative 过滤
- LoRA 高效微调，减少计算开销

---

## 实验/评估/结果

### 多模态检索 Benchmark

在 MMEB、跨模态检索等多个 benchmark 上：
- 全面超越 VLM2Vec、UniME 等现有 MLLM 嵌入方法
- Hardness-weighted loss 带来显著提升（尤其在需要细粒度判别力的任务上）
- False negative 过滤有效减少了嵌入空间的噪声

### 消融实验

- Hardness-weighted vs 标准 InfoNCE：+X% 平均提升
- 有/无 False negative 过滤：过滤后性能一致提升
- 任务特定温度 vs 全局温度：多任务场景下提升明显
- Model souping vs 单一 checkpoint：souping 带来稳定的小幅提升

---

## 结论

RzenEmbed 通过一套系统性的技术改进（加权 loss、false negative 过滤、任务温度、prompt 优化、model souping），在多模态检索嵌入任务上实现了全面提升。这些技术可以模块化地组合到其他 MLLM 嵌入方法中。

---

## 思考

### 优点

1. **Hardness-weighted loss 的简洁有效**：按相似度指数加权是最直观的"难例挖掘"方式，实现简单但效果显著。

2. **False negative 问题的正视**：在大规模检索数据中，false negatives 是一个真实且普遍存在的问题。RzenEmbed 是少数在 MLLM 嵌入场景中明确提出并解决这个问题的论文。

3. **技术模块化**：每个改进（加权 loss、温度、过滤、souping）相对独立，可以单独应用到其他方法中。这种模块化设计对社区有用。

4. **Model souping 的推广**：将 model souping 从视觉分类推广到 MLLM 嵌入，证明了这种技术的跨域适用性。

### 缺点与待解决问题

1. **指数加权的超参数敏感**：$\alpha$ 需要仔细调参，过大可能导致训练不稳定（hard negatives 主导），过小则退化为标准 loss。

2. **False negative 阈值的设定依赖先验**：$\theta$ 的最优值因数据集而异，论文未提供自动确定阈值的方案。

3. **Embedding prompt 设计的方法论不够系统**：prompt 优化更像是手工试错而非系统性搜索，缺乏如"自动 prompt 搜索"的方法论贡献。

4. **与 Magic-MM-Embedding 等同期工作的关系未讨论**：视觉 token 压缩和训练策略改进是互补的，结合二者可能产生更大收益。

### 与已有 Wiki 的连接

- 关联概念：[[InfoNCE]]、[[对比学习]]、[[多模态嵌入]]、[[LoRA]]、[[Model Soup]]
- 关联比较：与 [[Magic-MM-Embedding]] 的训练策略对比（两者都关注 hard negative mining，但方法路径不同）

---
type: concept
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - Multimodal Embedding
tags:
  - embedding
  - multimodal
  - retrieval
  - contrastive-learning
sources:
  - Wiki/Sources/Magic-MM-Embedding
  - Wiki/Sources/ObjEmbed
  - Wiki/Sources/RzenEmbed
  - Wiki/Sources/SAIL-Embedding
confidence: high
---
# 多模态 Embedding 模型

## 定义

多模态 Embedding 模型是将异构模态（文本、图像、视频、音频）映射到统一向量空间，使语义相似的跨模态样本在嵌入空间中距离接近的一类模型。核心应用包括跨模态检索、推荐系统和 RAG。

## 两大范式

### 1. 双塔架构（CLIP-style）

- 代表：CLIP、SigLIP、ALIGN、BLIP-2
- 特点：各模态独立编码器，仅在最后嵌入空间做对齐
- 优点：编码可预计算缓存，检索效率极高
- 局限：浅层融合，缺乏 token 级跨模态交互

### 2. MLLM 基统一架构

- 代表：VLM2Vec、MM-Embed、GME、Magic-MM-Embedding、ObjEmbed、RzenEmbed、SAIL-Embedding
- 特点：视觉 token 与文本 token 在统一 Transformer 中联合处理
- 优点：深层跨模态融合，强指令遵循，支持任意模态组合
- 挑战：视觉 token 序列长导致推理成本高

## 核心技术

1. **对比学习**：InfoNCE loss，正样本拉近 + 负样本推远
2. **难负样本挖掘**：动态阈值、自精炼、MLLM-as-Judge
3. **视觉 Token 压缩**：Magic-MM-Embedding 的插值下采样 75%
4. **细粒度表示**：ObjEmbed 的物体级 embedding + IoU 质量预测
5. **多模态扩展**：RzenEmbed 的视频/文档支持，SAIL-Embedding 的音频支持

## 评估标准

- **MMEB / MMEB-V2**：多模态 Embedding 标准 benchmark
- 任务覆盖：image-text、text-image、visual document、video retrieval 等

## 相关来源

- [[Wiki/Topics/多模态 Embedding 与检索]]
- [[Wiki/Comparisons/多模态 Embedding 模型比较]]

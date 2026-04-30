---
type: source
status: processed
source_file: "[[Clippings/OneRec-V2 Technical Report]]"
title: "OneRec-V2 Technical Report"
site: "arxiv.org"
author: "OneRec Team (Kuaishou)"
published: "2025"
processed: "2026-04-30"
updated: "2026-04-30"
tags:
  - "generative-recommendation"
  - "decoder-only"
  - "scaling-laws"
  - "reinforcement-learning"
  - "model-architecture"
aliases:
  - "OneRec-V2 技术报告"
---

# OneRec-V2 Technical Report

## 核心结论

OneRec-V2 针对 V1 的 encoder-decoder 架构计算效率瓶颈，提出 Lazy Decoder-Only 架构，将计算量降低 94%，训练资源减少 90%，成功将模型扩展至 8B 参数。在快手 A/B 测试中 App Stay Time 提升 0.467%/0.741%。

## 关键事实

- **Lazy Decoder-Only 架构**：核心洞察是在 V1 中，上下文编码（context encoding）消耗了 97.66% 的 FLOPs，而目标 item 解码（target decoding）仅占 2.34%。Lazy Decoder 将所有计算集中在语义 token 的解码上。
- **KV-Sharing**：多个 decoder block 共享同一组 key-value pair（来源于 Context Processor），消除 K/V 投影层。结合 GQA（Grouped Query Attention）大幅降低内存占用。
- **Scaling Laws**：0.1B 到 8B 参数范围内，收敛损失精确遵循 Hoffmann et al. (2022) 的理论 scaling law，验证了生成式推荐模型的 scaling 行为。
- **真实用户反馈 RL**：(1) Duration-Aware Reward Shaping：考虑视频时长差异，避免推荐偏向长视频；(2) Adaptive Ratio Clipping：稳定策略优化，减少训练方差。
- **Context Processor**：将异构用户特征（静态属性、行为序列等）统一编码为 context 表示，通过 RMSNorm 生成 layer-specific 的 K/V pairs。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/生成式推荐系统]]，[[Wiki/Concepts/生成式推荐]]，[[Wiki/Entities/OneRec 系列模型]]
- 该架构创新直接影响了 OpenOneRec 的设计

## 后续问题

- Decoder-only 架构在更长序列（>3000 tokens）下的 scaling 行为如何？
- 真实用户反馈 RL 是否存在 reward hacking 风险？

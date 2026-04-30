---
type: entity
status: active
created: "2026-04-30"
updated: "2026-04-30"
aliases:
  - "OneRec"
  - "OneRec 系列"
tags:
  - "model-family"
  - "generative-recommendation"
  - "Kuaishou"
sources:
  - "[[Wiki/Sources/OneRec Technical Report 2025]]"
  - "[[Wiki/Sources/OneRec Unifying Retrieve and Rank 2025]]"
  - "[[Wiki/Sources/OneRec-Think 2025]]"
  - "[[Wiki/Sources/OneRec-V2 Technical Report 2025]]"
  - "[[Wiki/Sources/OpenOneRec Technical Report 2025]]"
confidence: high
---

# OneRec 系列模型

## 概述

OneRec 是快手（Kuaishou）提出的端到端生成式推荐模型系列，从 2024 年到 2025 年经历了从统一召回排序、架构优化、推理引入到开源基础模型的完整演进。OneRec 系列代表了工业界将生成式 AI 应用于推荐系统的最前沿实践。

## 演进路线

### OneRec (V0) - 统一召回排序 (2024/2025)

**论文**：[[Wiki/Sources/OneRec Unifying Retrieve and Rank 2025]]
**核心贡献**：
- 首个在工业场景中显著超越传统级联推荐系统的**端到端生成式模型**
- Encoder-Decoder 架构（T5 风格）+ 稀疏 MoE 扩展容量
- **Session-wise 生成**：一次生成一个 session（5-10 个视频），而非逐点预测
- **Iterative Preference Alignment (IPA)**：reward model + self-hard negatives + DPO
- **在线效果**：watch-time 提升 1.6%

### OneRec (V1) - 技术报告 (2025)

**论文**：[[Wiki/Sources/OneRec Technical Report 2025]]
**核心贡献**（最全面的系统描述）：
- 完整的工业级架构：Tokenizer（RQ-Kmeans + 多模态协同对齐） -> Encoder（四条特征通路） -> Decoder（自回归生成 + RL）
- **Scaling Laws**：将 FLOPs 提升 10x 后首次发现推荐领域的 scaling law
- **基础设施优化**：MFU 达 23.7%（训练）/ 28.8%（推理），OPEX 仅 10.6%
- **处理 25% QPS**：快手主 APP + 快手极速版，App Stay Time 提升 0.54%/1.24%

### OneRec-V2 - Lazy Decoder-Only (2025)

**论文**：[[Wiki/Sources/OneRec-V2 Technical Report 2025]]
**核心贡献**：
- **Lazy Decoder-Only 架构**：消除 encoder 瓶颈，计算量降低 94%，训练资源减少 90%
- **KV-Sharing + GQA**：无需 K/V 投影的懒人 cross-attention，大幅降低显存
- **Context Processor**：异构特征统一编码
- **真实用户反馈 RL**：Duration-Aware Reward Shaping + Adaptive Ratio Clipping
- **扩展至 8B 参数**，scaling law 拟合准确
- **在线效果**：App Stay Time 提升 0.467%/0.741%

### OneRec-Think - CoT 推理推荐 (2025)

**论文**：[[Wiki/Sources/OneRec-Think 2025]]
**核心贡献**：
- 将 **Chain-of-Thought 推理**引入生成式推荐
- 三阶段训练：Itemic Alignment -> Reasoning Activation -> Reasoning Enhancement
- **Rollout-Beam Reward**：解决推荐场景中 RL reward 稀疏问题
- **Think-Ahead 架构**：离线推理 + 在线最终化，解决延迟问题
- SOTA on Amazon Beauty/Sports/Toys，App Stay Time 提升 0.159%

### OpenOneRec - 开源基础模型 (2025)

**论文**：[[Wiki/Sources/OpenOneRec Technical Report 2025]]
**核心贡献**：
- **首个开源推荐基础模型**（1.7B/8B），完整训练管线开源
- **RecIF-Bench**：8 任务 holistic 评估基准（Layer 0-3）
- **Scaling Laws**：N_opt ∝ C^0.44, D_opt ∝ C^0.56（推荐偏数据饥渴）
- **Co-Pretraining**：推荐语料 + 通用文本混合训练，防止灾难性遗忘
- **Amazon 跨域迁移**：平均 Recall@10 提升 26.8%

## 技术对比

| 特性 | V0 | V1 | V2 | Think | OpenOneRec |
|------|-----|-----|-----|-------|------------|
| 架构 | Enc-Dec | Enc-Dec | Lazy Dec-Only | LLM + Think-Ahead | Dense LM |
| 模型规模 | - | - | 0.1B-8B | - | 1.7B, 8B |
| Tokenization | Balanced K-Means | RQ-Kmeans + 协同对齐 | RQ-Kmeans（复用 V1） | Itemic Token | RQ-Kmeans |
| 生成方式 | Session-wise | Next token | Next token | CoT + item | Instruction-following |
| RL 对齐 | DPO (self-hard) | Reward Model RL | 真实用户反馈 RL | GRPO | GRPO |
| 开源 | 否 | 否 | 否 | 否 | 是 |
| 产业务效果 | +1.6% watch time | +0.54%/1.24% stay time | +0.467%/0.741% stay time | +0.159% stay time | Amazon +26.8% |
| 独特贡献 | 统一召回排序 | 系统架构 + scaling law | 94% 计算量降低 | CoT 可解释推荐 | 开源生态 |

## OneRec 系列对生成式推荐领域的影响

1. **技术验证**：端到端生成式推荐可以显著超越传统级联架构
2. **Scaling Laws**：首次在推荐领域验证 scaling law，为资源投入提供理论依据
3. **工程标杆**：MFU 达 28.8%、OPEX 仅 10.6% 成为行业参考
4. **开源推动**：OpenOneRec 开放完整训练管线，降低复现门槛
5. **范式拓展**：从纯推荐到推理增强（Think）、从封闭到开放（OpenOneRec）

## 与外部 AI 技术的连接

- **Transformer 架构**：从 T5 encoder-decoder 到 LLaMA decoder-only 的演进，与 LLM 架构发展平行
- **Scaling Laws**：与 NLP 领域的 Chinchilla (Hoffmann et al., 2022) 形成对比和互补
- **RLHF/RL**：DPO -> GRPO 的对齐技术迁移路径
- **MoE**：稀疏专家混合在推荐模型中的应用
- **CoT Reasoning**：从 LLM 的零样本 CoT 到推荐领域的推理激活和增强

## 相关页面

- [[Wiki/Topics/生成式推荐系统]]
- [[Wiki/Concepts/生成式推荐]]
- [[Wiki/Concepts/CTR 预估]]

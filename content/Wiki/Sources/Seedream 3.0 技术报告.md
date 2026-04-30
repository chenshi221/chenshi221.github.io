---
type: source
status: processed
source_file: "[[Clippings/Seedream 3.0 Technical Report]]"
title: "Seedream 3.0 Technical Report"
site: "arXiv (2025)"
author: "ByteDance Seed (Yu Gao, Lixue Gong 等)"
published: "2025"
processed: "2026-04-30"
updated: "2026-04-30"
tags: ["Seedream", "image-generation", "flow-matching", "MMDiT", "text-rendering", "acceleration"]
aliases: ["Seedream 3.0"]
confidence: high
---

# Seedream 3.0

## 核心结论

Seedream 3.0 在 2.0 基础上全面升级，解决了复杂提示对齐、细粒度排版、视觉美学和分辨率四项关键挑战。采用缺陷感知训练范式（defect-aware training）扩大有效数据集 21.7%，混合分辨率训练和跨模态 RoPE 增强可扩展性。首次引入 VLM 驱动的奖励模型并展现奖励模型缩放（reward model scaling）效应。通过一致噪声期望和重要性感知时间步采样，实现 4-8 倍加速。2025 年 4 月集成到豆包和即梦平台，Artificial Analysis Arena ELO 1158 排名第一。

## 关键方法

1. **数据层升级**：
   - 缺陷感知训练：用主动学习训练的缺陷检测器，掩码优化保留 20% 以下缺陷区域的样本，扩大有效数据 21.7%。
   - 双轴协同采样：视觉形态（层次聚类）+ 语义分布（TF-IDF），解决长尾问题。
   - 跨模态检索增强：专家知识注入、相似度加权校准、近邻对增强。

2. **预训练改进**：
   - 混合分辨率训练：在 $256^2$ 预训练后逐步提升至 $2048^2$，增强分辨率泛化。
   - 跨模态 RoPE：将文本 token 视为 2D token 应用 2D RoPE，增强图文对齐。
   - 表示对齐损失（REPA）：DINOv2-L 中间特征余弦距离作为辅助损失，加速大模型收敛。
   - 分辨率感知时间步采样：自适应 SNR 调度。

3. **后训练优化**：
   - 多样化美学标注（aesthetic captions），覆盖美学、风格、构图等专业领域。
   - VLM 驱动的奖励模型：从 1B 扩展到 >20B 的奖励模型缩放，展现 emergent scaling 效应。

4. **模型加速**：
   - 一致噪声期望（Consistent Noise Expectation）：统一噪声期望向量稳定采样，压缩步数不降质。
   - 重要性感知时间步采样：SSD + 神经网络学习关键时间步。

5. **性能**：1K 分辨率 3.0 秒生成，94% 中英文文本可用率，原生支持 2K 输出。

## 与现有 Wiki 的关系

- 系列演进：[[Wiki/Entities/Seedream 系列模型]]
- 基于 Flow Matching + MMDiT：[[Wiki/Concepts/Flow Matching]]、[[Wiki/Concepts/扩散模型原理]]
- REPA 辅助损失：表示学习加速扩散训练
- 奖励模型缩放效应：与 LLM 领域 RL scaling 类比

## 后续问题

- 复杂多图像编辑和多轮编辑仍是 4.0 的改进目标。
- GPT-4o 在英文密集文本渲染上仍有优势。

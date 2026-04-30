---
type: entity
status: active
created: "2026-04-30"
updated: "2026-04-30"
aliases: ["Seedream"]
tags: ["Seedream", "ByteDance", "image-generation", "image-editing", "flow-matching", "MMDiT"]
sources: ["[[Wiki/Sources/Seedream 2.0 中英双语图像生成]]", "[[Wiki/Sources/Seedream 3.0 技术报告]]", "[[Wiki/Sources/Seedream 4.0 多模态图像生成]]"]
confidence: high
---

# Seedream 系列模型

## 概述

Seedream 是字节跳动 Seed 团队开发的中英双语图像生成基础模型系列，从 2.0 到 4.0（含 4.5），经历了从 T2I 专用模型到多模态生成统一框架的演进。三条核心路线贯穿始终：中英双语文化理解、flow matching + MMDiT 架构、工业级推理加速。

## 版本演进

### Seedream 2.0 (2025 年初)

- **定位**：原生中英双语图像生成基础模型。
- **核心创新**：
  - 自研双语 LLM 作为文本编码器，原生学习中文文化知识。
  - Glyph-Aligned ByT5 实现字符级文本渲染。
  - Scaled RoPE 泛化到未训练分辨率。
  - 多阶段后训练：CT -> SFT -> RLHF -> PE -> Refiner。
- **局限**：仍需 Refiner 超分处理高分辨率；严格数据过滤损失 35% 潜在训练数据。

### Seedream 3.0 (2025 年 4 月)

- **定位**：全面能力提升 + 工业部署。
- **核心创新**：
  - 缺陷感知训练范式（扩大 21.7% 有效数据）。
  - 双轴协同数据采样（视觉+语义）。
  - 混合分辨率训练 + 跨模态 RoPE + REPA 对齐损失 + 分辨率感知时间步采样。
  - VLM 驱动的奖励模型（展现缩放效应）。
  - 一致噪声期望 + 重要性采样实现 4-8 倍加速。
  - 原生 2K 输出，1K 推理 3.0 秒。
- **关键突破**：文本渲染 94% 可用率（中文+16%），照片级真实感肖像与 Midjourney 并列第一。
- **部署**：接入豆包、即梦，Artificial Analysis Arena ELO 1158 排名第一。

### Seedream 4.0 (2025 年 9 月)

- **定位**：多模态图像生成统一框架（T2I + 编辑 + 多图合成）。
- **核心创新**：
  - 重新设计的 DiT backbone + 高压缩 VAE，算力效率提升 10 倍以上。
  - 知识数据专业处理（PDF 公式、图表等）。
  - 多模态联合后训练：CT -> SFT -> RLHF 同时优化 T2I 和编辑。
  - VLM (Seed1.5-VL) 作为 PE 模型：任务路由、自动思考、自适应宽高比。
  - 加速体系：ADP + ADM + 混合量化 + 推测解码 -> 2K 推理 1.4 秒。
- **新增能力**：精确编辑、灵活参考、视觉信号控制、上下文推理、多图输入/输出（>10 张）、自适应宽高比 4K。
- **性能**：Artificial Analysis Arena T2I 和图像编辑双赛道第一。

### Seedream 4.5

- 进一步扩大模型和数据，在所有维度超越 4.0，特别是编辑一致性和密集文本排版。

## 技术演进主线

| 维度 | 2.0 | 3.0 | 4.0 |
|------|-----|-----|-----|
| 任务范围 | T2I | T2I | T2I + 编辑 + 多图 |
| 架构 | MMDiT | MMDiT (更大) | 重新设计的 DiT + 高压缩 VAE |
| 文本编码 | 双语 LLM + ByT5 | 继承 2.0 | VLM 驱动的 PE 模型 |
| 后训练 | CT+SFT+RLHF+PE | 继承 + VLM 奖励模型 | 多模态联合后训练 |
| 最高分辨率 | 需 Refiner 超分 | 原生 2K | 原生 4K |
| 推理速度 | 量化减半 | 3.0s @ 1K | 1.4s @ 2K |
| 特色能力 | 中文文本渲染 | 照片级肖像、密集文本 | 多图编辑、推理生成 |
| 加速方法 | 量化 | 轨迹定制 + 重要性采样 | ADP+ADM+量化+推测解码 |

## 核心架构共性

- 均基于 Flow Matching (rectified flow) 训练目标：$\mathcal{L} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\|v_\theta(\mathbf{x}_t,t) - (\boldsymbol{\epsilon} - \mathbf{x}_0)\|_2^2$
- 均采用 MMDiT 架构（双流+单流混合 attenion blocks）
- 均经过 CT -> SFT -> RLHF -> PE 后训练管线

## 竞争对比

- vs FLUX/SD3：Seedream 中文理解和文本渲染显著更优。
- vs Midjourney：Seedream 3.0+ 在提示对齐和结构正确性上领先，美学上追赶。
- vs GPT-4o：Seedream 中文文本渲染和图像质量更优，GPT-4o 英文密集文本和 LaTeX 有优势。
- vs Gemini 2.5 Flash：Seedream 编辑一致性更强。

## 相关主题

- [[Wiki/Concepts/Flow Matching]]
- [[Wiki/Concepts/扩散模型原理]]
- [[Wiki/Topics/扩散模型与 Flow Matching 基础]]
- [[Wiki/Topics/扩散模型图像编辑与生成]]

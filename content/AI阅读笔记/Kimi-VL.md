---
title: "Kimi-VL Technical Report"
authors:
  - Kimi Team
institutions: Moonshot AI
aliases:
  - Kimi-VL
  - Kimi-VL-A3B
date: '2026-04-30'
publish_date: 2025-04
venue: 'arXiv:2504.07491'
tags:
  - 论文
  - 多模态
  - VLM
  - MoE
  - OCR
  - Agent
  - 长上下文
  - MoonViT
url: 'https://arxiv.org/abs/2504.07491'
code: https://github.com/MoonshotAI/Kimi-VL
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：Kimi-VL 是基于 Moonlight MoE 语言模型（2.8B 激活/16B 总参数）和 MoonViT 原生分辨率视觉编码器的开源 VLM，在仅 3B 激活参数下实现与 GPT-4o-mini、Qwen2.5-VL-7B 等模型竞争的性能，128K 长上下文支持长视频和文档理解，其 Thinking 变体通过 long-CoT SFT + RL 在 MMMU 等推理任务上超越更大的开源 VLM。

![](https://arxiv.org/html/2504.07491v3/x1.png)

*Figure 1: Kimi-VL-Thinking-2506 与其他 VLM 在 MathVision 上的对比。*

---

## Intro

### Motivation

开源 VLM 在多方面落后于纯文本 LLM：架构上仍以 dense 为主、缺乏长 CoT 推理、上下文长度受限。Kimi-VL 旨在提供一个集成结构创新、稳定能力和长思维推理的开源 VLM。

### 贡献

1. **MoonViT**：原生分辨率视觉编码器，基于 NaViT packing + 2D RoPE，无需子图拆分/拼接
2. **极小但强大的 MoE 架构**：仅 2.8B 激活，在 OCR、数学、agent 任务上与 7B+ dense VLM 竞争
3. **128K 长上下文**：长视频（LongVideoBench 64.5）和长文档（MMLongBench-Doc 35.1）
4. **Thinking 变体**：long-CoT SFT + RL，在 MMMU、MathVision 等推理 benchmark 上超越更大模型

---

## Method 核心方法

![](https://arxiv.org/html/2504.07491/x3.png)

*Figure 3: Kimi-VL 模型架构。由 MoonViT 原生分辨率视觉编码器、MLP Projector 和 MoE 语言解码器（Moonlight, 2.8B 激活）三部分组成。*

### 1. 模型架构

**MoonViT 视觉编码器**：
- 从 SigLIP-SO-400M 初始化继续预训练
- NaViT packing（图块展平 + 拼接为 1D 序列），支持同一 batch 内变分辨率训练
- 2D RoPE + 可学习固定位置编码，提升高分辨率细粒度位置信息
- Thinking 版本支持高达 3.2M 像素单图输入（4x 原始限制）

**MLP Projector**：
- Pixel shuffle（2x2 空间压缩）+ 两层 MLP 投影到 LLM embedding 维度

**MoE 语言模型**：
- 基于 Moonlight（2.8B 激活/16B 总参数），架构类似 DeepSeek-V3
- 从 5.2T tokens 纯文本 checkpoint 开始联合多模态继续预训练

### 2. 预训练（4 阶段、4.4T tokens）

| 阶段 | 数据 | Tokens | 训练 |
|------|------|--------|------|
| ViT Training | Caption/Grounding/OCR | 2T + 0.1T | ViT (+ projector) |
| Joint Pre-training | +Text/Knowledge/Interleaving/Video/Agent | 1.4T | ViT & LLM |
| Joint Cooldown | +High-quality text/multimodal/QA | 0.6T | ViT & LLM |
| Joint Long-context | +Long text/video/document | 0.3T | ViT & LLM |

- CoCa-alike ViT 训练：$\mathcal{L}=\mathcal{L}_{siglip}+2\cdot\mathcal{L}_{caption}$
- Joint Cooldown 中引入合成 QA pair 激活特定能力（低比例防止过拟合）
- Long-context 两阶段扩展到 128K（RoPE 从 50k 到 800k）

![](https://arxiv.org/html/2504.07491/x4.png)

*Figure 4: Kimi-VL 的四阶段预训练流程。ViT 训练 → Joint Pre-training → Joint Cooldown → Joint Long-context，共 4.4T tokens。*

### 3. 后训练

**Joint SFT**：
- 32K（1 epoch）→ 128K（1 epoch），使用 ChatML 格式
- 纯文本和视觉 SFT 数据混合，多样本 packing 提升训练效率

**Long-CoT SFT（Thinking 变体）**：
- Prompt engineering 构建小规模高质量 long-CoT warmup 数据集
- 包含规划、评估、反思和探索等认知模式

**RL（Thinking 变体）**：
- 在线 policy mirror descent（类似 K1.5），相对熵正则化
- Length-based reward 惩罚过长响应（减轻 overthinking）
- Curriculum sampling + prioritized sampling 提升效率

### 4. 训练基础设施

- 4D 并行：DP + EP + PP + CP（Context Parallelism）
- Vision Tower 与部分 decoder 层共置 Stage-0 减少 pipeline bubble
- MoE 架构训练吞吐比 7B dense VLM 高约 60%

---

## 实验/评估/结果

### 主结果（Kimi-VL-A3B vs 竞品）

| Benchmark | Kimi-VL-A3B | GPT-4o-mini | Qwen2.5-VL-7B | Gemma3-12B |
|-----------|-------------|-------------|---------------|------------|
| MMMU val | 57.0 | 60.0 | 58.6 | 59.6 |
| MathVista | 68.7 | 52.5 | 68.2 | 56.1 |
| InfoVQA | 83.2 | 57.9 | 82.6 | 43.8 |
| OCRBench | 867 | 785 | 864 | 702 |
| OSWorld | 8.22 | 5.03 | 2.5 | - |
| ScreenSpot-Pro | 34.5 | 0.8 | 29.0 | - |
| LongVideoBench | 64.5 | 58.2 | 56.0 | 51.5 |
| MMLongBench-Doc | 35.1 | 29.0 | 29.6 | 21.3 |

- 19/24 benchmark 超越 Qwen2.5-VL-7B（后者激活参数 2.59x）

### Thinking 变体

| Benchmark | Kimi-VL-Thinking-2506 |
|-----------|----------------------|
| MMMU | 64.0 |
| MMMU-Pro | 46.3 |
| MathVision | 56.9 |
| MathVista | 80.1 |
| VideoMMMU | 65.2 |

### NIAH (Needle-in-a-Haystack)

- 128K 文本 haystack：87.0% recall
- 128K 视频 haystack：91.7% recall

---

## 结论

Kimi-VL 证明了极小 MoE VLM（2.8B 激活）通过精心设计的训练策略可以在多模态理解、OCR、agent 和长上下文任务上达到甚至超越更大模型的性能。Thinking 变体进一步展现了长 CoT 推理对小模型多模态能力的显著提升。

---

## 思考

### 优点

1. **参数效率令人惊叹**：2.8B 激活在 OCRBench（867）、InfoVQA（83.2）等关键 benchmark 上达到 7B+ 水平
2. **MoonViT 的原生分辨率设计消除子图拆分**：NaViT packing + 2D RoPE 简化了训练和推理流程
3. **Joint Cooldown + 合成 QA 的策略精巧**：低比例注入 QA pair 激活能力而不过拟合
4. **128K 长上下文扩展完整**：NIAH 测试验证了文本和视频 haystack 上的有效性
5. **Scale 与训练效率考虑周全**：4D 并行 + Vision Tower 与 PP Stage-0 共置提升 60% 吞吐

### 缺点

1. **视频理解的绝对水平仍然有限**：VideoMMMU 52.6（vs GPT-4o 61.2），动态理解能力弱
2. **OS agent 成功率低**：OSWorld 仅 8.22%（虽推理能力强，但实际操作成功率仍很低）
3. **CoCa-alike ViT 训练的复杂性**：对比 loss + caption loss 的联合训练增加了实现复杂度
4. **训练细节披露不足**：Moonlight 的具体架构参数、数据配比等未详细展开
5. **Thinking 变体的评估不全面**：主要聚焦推理 benchmark，缺乏 agent 和通用能力评估

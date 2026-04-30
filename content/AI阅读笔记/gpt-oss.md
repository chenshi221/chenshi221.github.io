---
title: "gpt-oss-120b & gpt-oss-20b Model Card"
authors:
  - OpenAI
institutions: OpenAI
aliases:
  - gpt-oss
  - gpt-oss-120b
  - gpt-oss-20b
date: '2026-04-30'
publish_date: 2025-08
venue: 'arXiv:2508.10925'
tags:
  - 论文
  - 推理模型
  - MoE
  - 开源
  - MXFP4
  - 安全对齐
  - Agent
url: 'https://arxiv.org/abs/2508.10925'
code: https://github.com/openai/gpt-oss
rating: ⭐⭐⭐⭐
status: Read
---
**一句话总结**：OpenAI 发布的两个开源权重推理模型（gpt-oss-120b: 116.8B/5.1B 激活，gpt-oss-20b: 20.9B/3.6B 激活），采用 MoE + MXFP4 量化（单卡 80GB 可部署），通过 CoT RL 训练支持三档可变推理努力、工具使用（浏览/Python/自定义函数）和结构化输出，在 AIME 2025 上 120b 达到 92.5%（no tools），SWE-bench Verified 62.4，性能接近 o3-mini/o4-mini。

![](https://arxiv.org/html/2508.10925v1/x1.png)

*Figure 1: gpt-oss 模型与 o3、o3-mini、o4-mini 在主 benchmark 上的对比。*

---

## Intro

### Motivation

OpenAI 首次发布开源权重模型，旨在为社区提供可定制、可微调的推理模型，同时保持安全可控。开源模型与闭源 API 模型的安全风险不同——一旦发布，攻击者可通过微调绕过安全拒绝。

### 贡献

1. **两个开源推理模型**：Apache 2.0 许可，可用作 agentic workflow
2. **MXFP4 量化**：MoE 权重量化至 4.25 bits（90%+ 参数），120b 模型单 80GB GPU 可部署
3. **多级推理努力**：low/medium/high 三档，CoT 长度和准确率平滑缩放
4. **Harmony Chat Format**：专业 token 划定消息边界、角色层次和推理/工具通道
5. **全面安全评估**：默认模型 + 对抗微调版本，Preparedness Framework 评估

---

## Method 核心方法

### 1. 模型架构

| 参数 | gpt-oss-120b | gpt-oss-20b |
|------|-------------|-------------|
| 层数 | 36 | 24 |
| 总参数 | 116.8B | 20.9B |
| 激活参数 | 5.1B | 3.6B |
| Experts | 128 | 32 |
| Top-K experts | 4 | 4 |
| 残差维度 | 2880 | 2880 |
| 注意力头 | 64 Q, 8 KV (GQA) | 64 Q, 8 KV (GQA) |
| 上下文长度 | 131,072 (YaRN) | 131,072 (YaRN) |

**关键设计**：
- Banded window + full dense attention 交替（128 tokens 带宽）
- Learned bias in softmax denominator（类似 off-by-one attention / attention sink）
- Pre-LN（类似 GPT-2）+ RMSNorm + SwiGLU MoE
- o200k_harmony tokenizer（201,088 tokens）

**MXFP4 量化**：
- MoE 权重以 4.25 bits/param 存储
- 120b checkpoint 仅 60.8 GiB

### 2. 训练

- **预训练**：数万亿 tokens，聚焦 STEM/代码/通用知识，H100 GPU，120b 需 2.1M H100 小时
- **后训练**：CoT RL（类似 o3），在编程、数学、科学等广泛问题上训练
- **知识截止**：2024 年 6 月

### 3. Harmony Chat Format

- 特殊 token 标示消息边界、角色和可见性 channel
- 角色层次：System > Developer > User > Assistant > Tool
- Channel 分类：analysis（CoT）、commentary（function tool calling）、final（用户可见答案）
- 支持工具调用穿插在 CoT 中

### 4. 可变推理努力

- System prompt 中插入关键词（如 "Reasoning: low/medium/high"）
- 推理努力增加 → 平均 CoT 长度增长 → 准确率对数线性提升
- AIME 2025: low 50.4% → medium 80.0% → high 92.5%（120b, no tools）

![](https://arxiv.org/html/2508.10925/x3.png)

*Figure 3: AIME 和 GPQA 上的 test-time scaling。low/medium/high 三档推理努力下，accuracy 与 CoT+Answer 长度呈对数线性关系。*

### 5. Agentic 工具

- **Browsing**：search + open 函数与网络交互
- **Python**：Jupyter notebook 环境执行代码
- **自定义函数**：开发者通过 Developer message 定义函数 schema
- 工具可指定启用/禁用（system prompt 控制）

---

## 实验/评估/结果

### 主结果（high reasoning）

| Benchmark | gpt-oss-120b | gpt-oss-20b | o3-mini | o4-mini |
|-----------|-------------|-------------|---------|---------|
| AIME 2024 (no tools) | 95.8 | 92.1 | - | - |
| AIME 2025 (no tools) | 92.5 | 91.7 | - | - |
| AIME 2025 (with tools) | 97.9 | 98.7 | - | - |
| GPQA Diamond | 80.1 | 71.5 | - | - |
| MMLU | 90.0 | 85.3 | - | - |
| HLE (no tools) | 14.9 | 10.9 | - | - |
| SWE-Bench Verified | 62.4 | 60.7 | - | - |
| Codeforces Elo (w/ tools) | 2622 | 2516 | - | - |
| tau-Bench Retail | 67.8 | 54.8 | - | - |

### Test-time Scaling

- 所有 benchmark 上从 low → high 推理的一致性对数线性提升
- 20b 在 AIME 上用平均 20K+ CoT tokens/题

### 安全评估

- **默认模型**：Biological/Chemical/Cyber/AI Self-Improvement 均未达到 Preparedness Framework High 阈值
- **对抗微调后**：经 OpenAI SAG 审查，也未达到 High 阈值
- **与已有开源模型对比**：多数 bio 评估上已有开源模型接近甚至匹配 gpt-oss-120b 的对抗微调后性能
- **指令层次**：较 o4-mini 弱（system prompt extraction 0.832 vs 0.993）
- **幻觉**：SimpleQA accuracy 0.168（120b），o4-mini 0.234

---

## 结论

gpt-oss 模型是 OpenAI 开源策略的重要一步：提供接近 o3-mini/o4-mini 的推理能力，支持可变推理努力、工具使用和结构化输出。同时通过 Preparedness Framework 评估和对抗微调测试确保发布安全性。

---

## 思考

### 优点

1. **MXFP4 量化的实用价值**：120B MoE 模型单 80GB GPU 可部署，极大降低了开源模型使用门槛
2. **可变推理努力设计巧妙**：同一个模型通过 system prompt 控制计算量，兼顾时效性和深度推理
3. **Harmony Chat Format 是最完整的开源 agentic chat 格式**：角色层次 + channel 分类实现工具调用与 CoT 的优雅交织
4. **安全评估全面**：默认 + 对抗微调两轮评估，参考已有开源模型做基线比较
5. **20b 模型的小模型高推理能力**：AIME 2025 91.7%（no tools），20B 总参数做到这个水平很惊人

### 缺点

1. **技术细节极度贫乏**：作为 Model Card 而非 Technical Report，无预训练数据构成、超参数、训练曲线等
2. **知识密集型任务较弱**：SimpleQA 仅 16.8%（120b），HLE 14.9%，明显弱于更大模型
3. **幻觉率高**：SimpleQA hallucination rate 78.2%（120b），无 browsing 时事实性不可靠
4. **指令层次弱于闭源模型**：system prompt extraction 0.832（vs o4-mini 0.993），开发者需自行微调加固
5. **SWE-bench 与专用 agentic 模型差距大**：62.4 远低于 Kimi K2 的 65.8 和 Claude Opus 的 72.5

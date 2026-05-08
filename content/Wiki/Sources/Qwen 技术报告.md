---
type: source
status: complete
created: '2026-05-07'
updated: '2026-05-07'
tags: [llm, qwen, alibaba, pretraining, alignment, rlhf, code-generation, math-reasoning]
sources:
  - "[[Clippings/Qwen Technical Report]]"
---

# Qwen 技术报告

## 基本信息

- **标题**: Qwen Technical Report
- **作者**: Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang 等 50+ 位作者
- **机构**: Qwen Team, Alibaba Group（阿里巴巴集团通义千问团队）
- **年份**: 2023
- **来源**: arXiv:2309.16609
- **模型系列**: Qwen（千问）第一代

## 核心论点

1. **Qwen 是一个完整的语言模型系列**，涵盖基础预训练模型（QWEN）、对齐后的聊天模型（QWEN-CHAT，含 RLHF 版本）、代码专用模型（CODE-QWEN）和数学专用模型（MATH-QWEN-CHAT），参数规模包括 1.8B、7B 和 14B。
2. **大规模高质量数据是关键驱动力**：预训练数据达到 3 万亿 token，涵盖网页文档、百科、书籍、代码等多语种（中英为主）数据，并通过精细的数据清洗流程保障质量。
3. **小模型也能有强表现**：QWEN-1.8B 在部分任务上甚至超越了更大的模型，说明数据质量和训练策略的重要性。
4. **RLHF 对齐显著提升聊天质量**：人类评估表明 QWEN-CHAT-14B（RLHF）显著优于纯 SFT 版本，但仍落后于 GPT-4。
5. **Qwen 具备出色的工具使用和 Agent 能力**：在 ReAct 提示下使用未见工具、Python 代码解释器执行数学推理和数据分析、以及作为 HuggingFace Agent 调用多模态工具等方面，表现接近 GPT-4 水平。

## 关键技术方法

### 架构

- 基于 LLaMA 的 Transformer 架构，做了以下改进：
  - **解耦嵌入**：输入嵌入和输出投影不共享权重（untied embedding），以提升性能。
  - **位置编码**：使用 RoPE（Rotary Positional Embedding），逆频率矩阵采用 FP32 精度。
  - **偏置**：大部分层移除 bias，但在注意力的 QKV 层保留 bias 以增强外推能力。
  - **归一化**：Pre-Norm + RMSNorm，替代传统 LayerNorm。
  - **激活函数**：使用 SwiGLU（Swish + Gated Linear Unit），FFN 维度从 4 倍隐藏大小降至 8/3 倍。
  - **注意力**：使用 Flash Attention 加速训练。

### 训练数据

- 预训练数据规模：最多 3 万亿 token。
- 数据来源：公共网页文档、百科、书籍、代码等，中英双语为主。
- 数据清洗流程：语言识别 → 精确去重 + MinHash/LSH 模糊去重 → 规则 + 机器学习方法过滤低质量数据 → 多模型评分 → 人工抽检 → 选择性上采样高质量来源。
- 指令数据混入预训练以增强 zero-shot 和 few-shot 能力，同时采用 13-gram 重叠过滤避免基准测试数据泄露。

### Tokenizer

- 基于 BPE（byte pair encoding），以 tiktoken 的 cl100k base 为起点。
- 扩充词汇：添加常用中文字词和其他语言词汇，数字拆分为单个数字。
- 最终词汇量约 152K。
- 压缩效率在大多数语言上优于 XLM-R、LLaMA、Baichuan、InternLM 等竞品。

### 训练策略

- 自回归语言建模（next-token prediction），上下文长度 2048。
- 优化器：AdamW（β1=0.9, β2=0.95, ε=10^-8）。
- 余弦学习率调度，BFloat16 混合精度训练。

### 上下文长度扩展

- 推理阶段采用免训练技术扩展上下文：
  - **NTK-aware interpolation**：动态调整 RoPE 基频，避免高频信息损失。
  - **Dynamic NTK-aware interpolation**：按 chunk 动态改变缩放比例。
  - **LogN-Scaling**：按上下文长度与训练长度的比值缩放 query-value 点积，保持注意力熵稳定。
  - **窗口注意力**：限制注意力范围，低层使用更短窗口、高层使用更长窗口。

### 对齐（Alignment）

- **SFT**：使用 ChatML 格式的多风格对话数据，包含任务执行、聊天、工具使用、安全等类别。损失函数仅对 assistant 回复计算。
- **RLHF**：
  - 奖励模型：先进行偏好模型预训练（PMP），再用高质量对比数据微调。使用约 6600 个详细标签的分类体系和平衡采样算法。
  - PPO 训练：包含策略模型、价值模型、参考模型和奖励模型四个模型。每条 query 采样两个回复，KL 散度系数 0.04。使用预训练梯度缓解 alignment tax。

### 模型家族

| 模型 | 参数量 | 说明 |
|------|--------|------|
| QWEN-1.8B / 7B / 14B | 基础预训练模型 |  |
| QWEN-CHAT-1.8B / 7B / 14B | SFT 对齐后聊天模型 |  |
| QWEN-CHAT-14B-RLHF | RLHF 对齐后聊天模型 |  |
| CODE-QWEN-7B / 14B | 代码继续预训练 | 基于 QWEN + 900 亿 token 代码数据 |
| CODE-QWEN-CHAT-7B / 14B | 代码 SFT 模型 | 多阶段 SFT 策略 |
| MATH-QWEN-CHAT-7B / 14B | 数学专用聊天模型 |  |

## 主要结果

### 基础模型

- **QWEN-14B** 在全部 7 个基准测试上超越了此前同参数量级的 SOTA（包括 LLaMA2-70B 在 3 个任务上的表现）。
- **QWEN-7B** 超越 LLaMA2-13B，与 Baichuan2-13B 持平。
- **QWEN-1.8B** 在某些任务上超越了更大的模型。
- 核心基准：MMLU (66.3), C-Eval (72.1), GSM8K (61.3), MATH (24.8), HumanEval (32.3), MBPP (40.8), BBH (53.4)（QWEN-14B，5-shot/8-shot 设置）。

### 对齐模型

- **QWEN-14B-Chat** 在所有基准上超越除 GPT-4 和 LLaMA2-70B-Chat 外的所有模型。
- 人类评估（300 条中文指令）：RLHF 版本显著优于 SFT 版本，整体接近但未达到 GPT-4 水平。
- HumanEval 代码生成质量显著高于其他开源模型。

### 工具使用与 Agent

- **ReAct 工具调用**：QWEN-14B-CHAT 工具选择准确率 98%，参数匹配 Rouge-L 93%，假阳性率仅 2.4%，优于 GPT-3.5。
- **代码解释器**：QWEN-14B-CHAT 代码可执行率 81.7%，最终答案正确率 56.4%，大幅领先其他开源模型。
- **HuggingFace Agent**：QWEN-14B-CHAT 在 Run Mode 下工具选择 93.5%、代码正确率 87.0%，接近 GPT-4。

### 代码专用模型

- **CODE-QWEN-CHAT-14B** 在 HumanEval 上达到 66.4% pass@1，大幅超越同规模开源模型，接近 GPT-3.5 (73.2%)。
- 多语言代码生成（HUMANEVALPACK）：CODE-QWEN-CHAT-14B 平均 51.9%，领先所有同规模开源模型。

### 数学专用模型

- **MATH-QWEN-14B-CHAT**：GSM8K 69.8%，MATH 24.2%，Math401 85.0%，Math23K 78.4%。
- 在算术能力和中文数学题上超越 GPT-3.5，逼近 Minerva-62B 水平。

## 局限性

1. **与闭源模型仍有差距**：在人类评估和部分基准测试中，QWEN 仍落后于 GPT-4，特别是在复杂推理任务上。
2. **评估方法的局限**：作者明确指出传统基准测试难以准确衡量对齐后聊天模型的真实能力，呼吁开发更适合的评估方法。
3. **上下文长度限制**：基础训练上下文仅 2048 token，扩展至 8192+ 依赖推理阶段的免训练技术，可能在极端长文本场景下表现受限。
4. **代码和数学专用模型的通用性**：CODE-QWEN 仅在代码数据上继续预训练，可能损失部分通用能力；MATH-QWEN 仅通过数学 SFT 构建，未进行数学领域继续预训练。
5. **数据透明度**：虽然描述了数据处理流程，但具体数据来源和构成比例未完全公开。

## 与相关工作的关系

- **LLaMA / LLaMA 2**：Qwen 架构以 LLaMA 为基础，进行了多项改进（解耦嵌入、QKV bias、SwiGLU 等）。QWEN-14B 在多数基准上超越 LLaMA2-70B，说明在更小参数量下通过数据和训练优化可实现更好效果。
- **Baichuan2 / InternLM / ChatGLM2**：同为中国团队开发的开源大模型。QWEN-14B 在所有基准上超越这些模型的同参数量版本。
- **GPT-3.5 / GPT-4**：作为闭源基准，QWEN-14B-Chat 在自动评估中接近 GPT-3.5，在人类评估中介于 GPT-3.5 和 GPT-4 之间。
- **Code Llama / StarCoder**：代码专用模型。CODE-QWEN-CHAT-14B 在 HumanEval 上显著超越 Code Llama-34B 和 StarCoder-15B。
- **WizardMath / GAIRMath-Abel**：数学专用开源模型。MATH-QWEN-14B-CHAT 在 GSM8K 和 MATH 上超越 WizardMath-70B。
- **QWEN-VL**：Qwen 团队同期发布的多模态模型，基于同一基础模型构建，本报告中用于评估代码解释器生成的可视化结果质量。
- **RLHF 生态**：采用 PPO 进行强化学习，与 InstructGPT 路线一致；同时使用预训练梯度缓解 alignment tax，是该报告的一个技术创新点。

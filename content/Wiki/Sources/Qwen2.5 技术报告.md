---
type: source
status: complete
created: '2026-05-07'
updated: '2026-05-07'
tags: [llm, qwen, alibaba, pretraining, coding, math]
sources:
  - "[[Clippings/Qwen2.5 Technical Report]]"
---

# Qwen2.5 技术报告

## 基本信息

- **标题**：Qwen2.5 Technical Report
- **作者**：Qwen Team（核心贡献者：An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Junyang Lin, Jingren Zhou 等）
- **机构**：阿里巴巴集团
- **年份**：2024
- **来源**：arXiv:2412.15115
- **许可**：Apache 2.0（0.5B/1.5B/7B/14B/32B），Qwen Research（3B），Qwen（72B）

## 核心论点

1. **数据规模是关键驱动力**：预训练数据从 Qwen2 的 7 万亿 token 扩展到 18 万亿 token，在知识、编码和数学领域的能力显著提升，证明了数据规模与多样性对模型能力的重要性。
2. **小模型可以匹敌大模型**：Qwen2.5-72B-Instruct 的性能可与约 5 倍参数量的 Llama-3-405B-Instruct 竞争，体现了数据质量和训练策略对模型效率的关键作用。
3. **后训练同等重要**：通过 100 万+ 样本的 SFT 和两阶段强化学习（离线 DPO + 在线 GRPO），在人类偏好对齐、长文本生成、结构化数据分析和指令遵循方面取得显著提升。
4. **完整尺寸覆盖生态需求**：从 0.5B 到 72B 的密集模型加上 MoE 变体（Turbo/Plus），覆盖从边缘设备到 API 服务的全场景需求。
5. **专用模型可叠加增益**：Qwen2.5 作为基座模型，成功衍生出 Qwen2.5-Math、Qwen2.5-Coder、QwQ 等专用模型，验证了通用基座的可扩展性。

## 关键技术方法（相比 Qwen2 的改进）

### 预训练阶段

- **数据质量提升**：使用 Qwen2-Instruct 作为数据质量过滤器，对训练样本进行多维度评估和打分，替代之前的方法。
- **数学与代码数据整合**：直接引入 Qwen2.5-Math 和 Qwen2.5-Coder 的训练数据，让基座模型在预训练阶段就获得强大的数学推理和代码生成能力。
- **合成数据增强**：利用 Qwen2-72B-Instruct 和 Qwen2-Math-72B-Instruct 生成高质量合成数据，并通过专用奖励模型过滤。
- **数据配比优化**：对过度代表的领域（电商、社交媒体、娱乐）进行降采样，对高质量但代表不足的领域（技术、科学、学术研究）进行升采样。
- **Scaling Law 指导超参数**：基于 44M 到 14B 参数的密集模型和 44M 到 1B 激活参数的 MoE 模型，建立学习率和 batch size 与模型规模的关系。
- **长上下文预训练**：分阶段扩展上下文长度，从 4,096 到 32,768 token；Qwen2.5-Turbo 进一步通过 4 阶段训练达到 262,144 token，RoPE 基频从 10,000 提升到 10,000,000。

### 后训练阶段

- **大规模 SFT**：超过 100 万样本，覆盖长序列生成（最长 8,192 token 输出）、数学（链式思维数据）、编码（近 40 种编程语言）、指令遵循（代码验证框架）、结构化数据理解、逻辑推理（7 万新查询）、跨语言迁移和鲁棒系统指令。
- **两阶段强化学习**：
  - **离线 RL（DPO）**：约 15 万训练对，聚焦推理、事实性和指令遵循等难以用奖励模型评估的能力。
  - **在线 RL（GRPO）**：基于奖励模型检测输出质量细微差异，每个查询采样 8 个回复，使用 2048 全局 batch size。
- **长上下文微调**：Qwen2.5-Turbo 采用两阶段 SFT（短指令 + 混合长短指令），RL 阶段仅使用短指令但仍能提升长上下文对齐效果。

### 推理增强

- **YARN + DCA**：通过 YARN 和 Dual Chunk Attention 将序列长度容量提升 4 倍，Qwen2.5-Turbo 支持 100 万 token，其他模型支持 131,072 token。
- **稀疏注意力加速**：基于 Minference 的稀疏注意力机制，将 100 万 token 序列的注意力计算量降低 12.5 倍，TTFT 加速 3.2 到 4.3 倍。

### 架构特点

- 沿用 Qwen2 的 Transformer 解码器架构：GQA、SwiGLU、RoPE + QKV Bias、RMSNorm。
- MoE 模型采用细粒度专家分割和共享专家路由。
- 词表统一为 151,643 个常规 token + 22 个控制 token（含 2 个工具功能 token）。

## 主要结果

### 基座模型

- Qwen2.5-72B 在 MMLU（86.1）、MATH（62.1）、GSM8K（91.5）等基准上显著超越 Qwen2-72B，与 Llama-3-405B 相当。
- Qwen2.5-32B 在 MATH（57.7）和 MBPP（84.5）上大幅超越 Qwen1.5-32B。
- Qwen2.5-Turbo（MoE）以远低于 Qwen2.5-14B 的训练和推理成本，达到可比性能，MMLU-Pro 甚至超过 Qwen2.5-32B。

### 指令微调模型

- Qwen2.5-72B-Instruct 在 MATH（83.1）、MBPP（88.2）、LiveCodeBench（55.5）、Arena-Hard（81.2）等多项基准上超越 Llama-3.1-405B-Instruct。
- Qwen2.5-7B-Instruct 在 MATH（75.5）和 HumanEval（84.8）上显著领先 Gemma2-9B-IT 和 Llama3.1-8B-Instruct。
- Qwen2.5-0.5B 的性能接近甚至超越 Qwen2-1.5B，Qwen2.5-3B 接近 Qwen2-7B。
- Qwen2.5-Turbo 在 10 个基准中有 8 个超越 Qwen2.5-14B-Instruct，成本显著更低。
- Qwen2.5-Plus 在 13 个基准中有 9 个超越 Qwen2.5-72B-Instruct。

### 长上下文能力

- Qwen2.5-72B-Instruct 在 RULER 上平均 95.1，128K 上下文下 88.4，显著优于 GPT-4（91.6/81.2）和 Llama-3.1-70B-Instruct（89.6/66.6）。
- Qwen2.5-Turbo 在 100 万 token passkey retrieval 任务上达到 100% 准确率。
- YARN + DCA 在 128K 以上长度带来显著增益，32K 以内无影响。

### 多语言能力

- Qwen2.5-72B-Instruct 在多语言指令遵循（IFEval multilingual 86.98）、知识（JMMLU 80.56、TurkishMMLU 76.12）等任务上领先。
- 文化细微理解（BLEnD）仍有提升空间。

### 奖励模型

- Qwen2.5-RM-72B 在 PPE 和中文人类偏好基准上排名第一，RMB 上排名第二。
- 发现：在单一 RM 基准上过度优化会触发 Goodhart 定律，RM 基准分数不能准确预测下游 RL 模型性能。

## 局限性

1. **文化细微理解有限**：在 BLEnD 基准上的表现虽有改进但仍不理想，跨文化知识理解需进一步增强。
2. **指令遵循仍有差距**：与 Llama-3.1-405B-Instruct 相比，Qwen2.5-72B 在 IFEval 上略有差距（84.1 vs 86.0）。
3. **奖励模型评估瓶颈**：当前 RM 评估基准不能准确预测 RL 模型性能，需要更可靠的评估方法。
4. **长上下文 RL 训练缺失**：由于计算成本和缺乏合适的长上下文奖励模型，Qwen2.5-Turbo 的 RL 阶段仅使用短指令。
5. **多语言翻译任务偏弱**：Flores-101 翻译任务分数相对较低（Qwen2.5-72B 为 39.0），多语言翻译能力有待提升。
6. **部分小模型许可限制**：Qwen2.5-3B 使用 Qwen Research 许可，Qwen2.5-72B 使用 Qwen 许可，均非完全开源许可。

## 与相关工作的关系

- **Qwen2**：Qwen2.5 是 Qwen2 的直接迭代，预训练数据从 7T 扩展到 18T token，后训练数据和方法均有大幅改进，所有尺寸模型在几乎所有基准上均有提升。
- **Qwen2.5-Math / Qwen2.5-Coder**：作为专用模型的训练数据被反向整合到 Qwen2.5 基座预训练中，形成正反馈循环。
- **Llama-3.1 系列**：主要竞争对手。Qwen2.5-72B 以约 1/5 参数量匹敌 Llama-3-405B，Qwen2.5-7B 在数学和编码上显著领先 Llama3.1-8B。
- **Gemma2 系列**：在 7B-9B 和 27B-32B 级别上，Qwen2.5 全面超越 Gemma2 对应模型。
- **GPT-4o / GPT-4o-mini**：Qwen2.5-Plus 竞争性能对标 GPT-4o，Qwen2.5-Turbo 和 Qwen2.5-14B 竞争性能对标 GPT-4o-mini。
- **GRPO（Shao et al., 2024）**：在线 RL 采用的 Group Relative Policy Optimization 方法，源自 DeepSeekMath。
- **DPO（Rafailov et al., 2023）**：离线 RL 阶段使用 Direct Preference Optimization。
- **YARN / DCA**：长上下文扩展采用已有的 YaRN 和 Dual Chunk Attention 技术。

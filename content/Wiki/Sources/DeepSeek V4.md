---
type: source
status: complete
created: '2026-05-07'
updated: '2026-05-07'
tags: [llm, deepseek, moe, reasoning, long-context, attention, quantization]
sources:
  - "[[Clippings/DeepSeek V4]]"
---

# DeepSeek V4: Towards Highly Efficient Million-Token Context Intelligence

## 基本信息

- **标题**: DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
- **作者**: DeepSeek-AI
- **机构**: DeepSeek
- **年份**: 2025（Preview 版本）
- **模型**: DeepSeek-V4-Pro（1.6T 参数，49B 激活）和 DeepSeek-V4-Flash（284B 参数，13B 激活），均支持 100 万 token 上下文
- **开源**: 模型权重发布于 HuggingFace

## 核心论点

1. **长上下文效率突破**: 通过混合注意力架构（CSA + HCA），在 100 万 token 上下文场景下，DeepSeek-V4-Pro 的单 token 推理 FLOPs 仅为 DeepSeek-V3.2 的 27%，KV 缓存仅为 10%，从根本上突破了超长上下文的效率瓶颈。
2. **开源模型新 SOTA**: DeepSeek-V4-Pro-Max（最大推理努力模式）在知识、推理、编码、长上下文和智能体等任务上全面超越此前所有开源模型，在部分基准上接近甚至匹配闭源前沿模型。
3. **测试时计算扩展的基础设施**: 高效的百万级上下文支持为推理时扩展（test-time scaling）、长周期任务和在线学习等新范式奠定了基础。
4. **小模型高效推理**: DeepSeek-V4-Flash-Max 以 13B 激活参数在推理任务上达到与 GPT-5.2、Gemini-3.0-Pro 相当的水平，展示了极高性价比架构的可行性。
5. **两阶段后训练范式**: 采用"领域专家独立培养 + 统一模型整合"的后训练流程，以 On-Policy Distillation（OPD）替代混合 RL，实现多领域能力的高效融合。

## 关键技术方法（相比 V3/V3.2 的改进）

### 架构创新

- **混合注意力架构（CSA + HCA）**：
  - Compressed Sparse Attention（CSA）：将每 m 个 token 的 KV 缓存压缩为 1 个条目，再通过 Lightning Indexer 做 top-k 稀疏选择 + MQA 注意力
  - Heavily Compressed Attention（HCA）：以更大压缩率（m' >> m）压缩 KV 缓存，保持密集注意力
  - 两者交替配置，配合滑动窗口注意力分支和 Attention Sink 技术
  - 相比 V3/V3.2 的 DeepSeek Sparse Attention（DSA），大幅降低了长上下文的计算和存储开销
- **流形约束超连接（mHC）**：替代传统残差连接，将残差映射矩阵约束到双随机矩阵流形上（Birkhoff 多面体），确保谱范数不超过 1，增强深层网络的数值稳定性。通过 Sinkhorn-Knopp 迭代实现约束投影，开销仅约 6.7%
- **MoE 微调**：继承 DeepSeekMoE 框架，但将亲和力计算从 Sigmoid 改为 Sqrt(Softplus)，移除路由目标节点数约束，前 3 层使用 Hash 路由替代密集 FFN

### 优化器

- **Muon 优化器**：替代 AdamW 用于大部分参数（嵌入层、预测头、RMSNorm 等仍用 AdamW），通过混合 Newton-Schulz 迭代实现矩阵正交化，带来更快收敛和更好训练稳定性。采用混合 ZeRO 策略适配 Muon 的全梯度矩阵需求

### 训练稳定性

- **Anticipatory Routing**：解耦主干网络和路由网络的同步更新，使用历史参数计算路由索引，避免 loss spike 的恶性循环
- **SwiGLU Clamping**：将 SwiGLU 的线性部分限制在 [-10, 10]，门控部分上限为 10，有效消除异常值

### 基础设施

- **FP4 量化感知训练（QAT）**：对 MoE 专家权重和 CSA 索引器的 QK 路径应用 MXFP4 量化，推理时直接使用 FP4 权重；FP4→FP8 反量化无损，可复用现有 FP8 训练框架
- **细粒度 EP 融合核**：将 MoE 层的通信和计算融合为单一流水线化 kernel，专家按波次调度，实现 1.50~1.73x 加速（RL rollout 场景最高 1.96x）
- **TileLang DSL**：用于灵活高效地开发融合 kernel，集成 Z3 SMT 求解器做形式化整数分析，支持按位可复现性
- **批不变且确定性的 kernel 库**：端到端确保训练和推理的按位可复现性，包括注意力、矩阵乘法、mHC 等模块
- **扩展自动微分**：实现张量级激活检查点，支持细粒度重计算控制
- **磁盘 KV 缓存**：支持共享前缀复用，避免重复 prefill

### 后训练流程

- **专家训练**：对数学、编码、智能体、指令遵循等独立训练专家模型，SFT + GRPO 强化学习
- **三种推理模式**：Non-think（快速）、Think High（深入推理）、Think Max（最大推理努力）
- **生成式奖励模型（GRM）**：actor 网络本身兼任 GRM，统一优化生成和评估能力
- **On-Policy Distillation（OPD）**：用十余个教师模型通过全词汇表 KL 散度蒸馏合并为一个学生模型，替代 V3.2 的混合 RL 阶段
- **交错思考（Interleaved Thinking）**：工具调用场景下完整保留推理历史；一般对话场景下新用户消息到达时清除推理痕迹
- **Quick Instruction**：通过专用特殊 token 在输入序列末尾并行执行辅助任务（搜索判断、意图识别等），避免额外小模型的重复 prefill
- **XML 工具调用格式**：使用 `|DSML|` 特殊 token 和 XML 格式，减少转义失败和工具调用错误

## 主要结果

### 基座模型

- DeepSeek-V4-Flash-Base（13B 激活）在多数基准上超越 DeepSeek-V3.2-Base（37B 激活），体现架构和数据改进带来的效率收益
- DeepSeek-V4-Pro-Base 在知识、推理、编码和长上下文能力上全面领先，成为 DeepSeek 系列最强基座模型

### 后训练模型

- **知识**: V4-Pro-Max 在 SimpleQA 上比此前最佳开源模型高出 20 个百分点，但仍在 Gemini-3.1-Pro 之后
- **推理**: 在 LiveCodeBench 和 Codeforces 上与 GPT-5.4 相当，Codeforces 排名人类选手第 23 位；形式数学推理在 agentic 设置下达到 SOTA
- **智能体**: 代码智能体性能接近 Claude Opus 4.5 水平；在 MCPAtlas 和 Toolathlon 上表现优异
- **长上下文**: 在 MRCR 和 CorpusQA 100 万 token 基准上超越 Gemini-3.1-Pro
- **中文写作**: 功能写作和创意写作均优于 Gemini-3.1-Pro
- **白色领任务**: 在 30 项中文企业生产力任务上超越 Opus-4.6-Max

### 效率对比（100 万 token 上下文）

| 模型 | 单 token FLOPs（vs V3.2） | KV 缓存大小（vs V3.2） |
|---|---|---|
| DeepSeek-V4-Pro | 27% | 10% |
| DeepSeek-V4-Flash | 10% | 7% |

## 局限性

1. **架构复杂性**: 为追求极端长上下文效率，采用了大胆且相对复杂的架构设计，保留了大量初步验证的组件和技巧，不够简洁优雅
2. **训练稳定性机理不明确**: Anticipatory Routing 和 SwiGLU Clamping 虽然有效，但底层原理理解不足，缺乏原则性的预测方法
3. **与前沿闭源模型仍有差距**: 在部分知识基准上落后 Gemini-3.1-Pro，在高难度推理任务上落后 GPT-5.4 和 Gemini-3.1-Pro 约 3-6 个月
4. **格式美学不足**: 在 PPT 等展示类任务的视觉排版方面仍有较大改进空间
5. **指令遵循偶有疏漏**: 偶尔忽略特定格式约束
6. **尚无多模态能力**: 论文明确表示正在开发多模态能力，当前版本仅支持文本
7. **Flash 模型在复杂任务上的局限**: DeepSeek-V4-Flash-Max 在高难度智能体任务和知识密集型任务上仍落后于 V4-Pro

## 与 DeepSeek 系列模型的关系

- **继承**: 沿用 DeepSeekMoE 框架和 Multi-Token Prediction（MTP）策略，训练数据在 V3 基础上扩展至 32T+ token，tokenizer 词汇表保持 128K
- **演进路径**: DeepSeek-V2（MoE 架构）→ DeepSeek-V3（MTP、辅助损失-free 负载均衡）→ DeepSeek-V3.2（推理能力增强）→ DeepSeek-V4（百万级上下文效率突破）
- **关键差异**: V4 相比 V3/V3.2 最核心的变化是注意力机制（CSA+HCA 替代 DSA）、残差连接（mHC 替代标准残差）和优化器（Muon 替代 AdamW），以及引入 FP4 量化感知训练
- **后训练范式转变**: 从 V3.2 的混合 RL 阶段转变为 OPD 为主，用十余个领域专家教师模型蒸馏统一学生模型
- **基础设施积累**: 在 V3 的 DualPipe、3FS 等基础上，新增 TileLang、MegaMoE 融合核、DSec 沙箱平台等基础设施

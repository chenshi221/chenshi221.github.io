---
type: source
status: complete
created: '2026-05-07'
updated: '2026-05-07'
tags: [llm, agent, coding, glm, zhipu, agentic-engineering, reinforcement-learning, moe]
sources:
  - "[[Clippings/GLM-5: from Vibe Coding to Agentic Engineering]]"
---

# GLM-5: 从 Vibe Coding 到 Agentic Engineering

## 基本信息

- **作者**：Zhipu AI & 清华大学（核心贡献者包括 Aohan Zeng、Xin Lv、Zhenyu Hou 等，顾问包括 Jie Tang、Yuxiao Dong 等）
- **发布年份**：2026
- **来源**：[arXiv:2602.15763](https://arxiv.org/abs/2602.15763)
- **代码与模型**：[github.com/zai-org/GLM-5](https://github.com/zai-org/GLM-5)
- **定位**：智谱 AI 下一代旗舰基础模型，GLM-4.5 的后继者

## 核心论点

GLM-5 提出从 **Vibe Coding**（人类提示、AI 写代码）向 **Agentic Engineering**（AI Agent 自主规划、实现、迭代）的范式转变。其核心主张：

1. **开放权重模型可以媲美顶级闭源系统**：GLM-5 在主流基准上达到开源模型 SOTA，与 Claude Opus 4.5、GPT-5.2 (xhigh) 可比，超过 Gemini 3 Pro。
2. **真实软件工程能力比静态基准更重要**：GLM-5 在端到端软件开发任务（SWE-bench、Terminal-Bench、CC-Bench-V2）上展示了前所未有的能力。
3. **效率与性能可以兼得**：通过 DSA（DeepSeek Sparse Attention）等架构创新，在不牺牲长上下文能力的前提下显著降低训练和推理成本。

## 关键技术方法

### 模型架构

- **规模**：744B 总参数，40B 激活参数（MoE 架构，256 个专家，80 层，较 GLM-4.5 的 355B/32B �倍）
- **训练数据**：总计 28.5 万亿 tokens（预训练 27T + 中期训练 1.5T）
- **上下文长度**：从 4K 逐步扩展到 200K
- **Multi-Latent Attention (MLA)**：采用 Muon Split 技术解决了 MLA 在 Muon 优化器下的性能差距，并优化了 MLA-256 变体以降低解码计算量
- **Multi-Token Prediction (MTP)**：提出参数共享的 3 层 MTP，提高推测解码接受率（2.76 vs DeepSeek-V3.2 的 2.55）

### DSA（DeepSeek Sparse Attention）

- 将传统 O(L^2) 密集注意力替换为动态细粒度选择机制，根据 token 重要性分配注意力资源
- 通过"密集预热 + 稀疏训练适应"两阶段策略从密集模型迁移，仅需 20B tokens 适配（远少于 DeepSeek-V3.2 的 943.7B）
- 长序列注意力计算降低约 1.5-2 倍，128K 上下文仅需一半 GPU 成本
- 相比滑动窗口注意力 (SWA)、Gated DeltaNet (GDN) 等替代方案，DSA 在长上下文任务上无损

### 训练流程

1. **预训练**：27T tokens 语料，优先代码和推理数据
2. **中期训练 (Mid-Training)**：三阶段上下文扩展（32K/128K/200K），重点注入长上下文 Agent 数据和软件工程数据（约 1000 万个 issue-PR 对）
3. **监督微调 (SFT)**：覆盖通用对话、推理、编码与 Agent 三大类，支持三种思维模式：
   - **Interleaved Thinking**：每次响应和工具调用前思考
   - **Preserved Thinking**：编码 Agent 场景中跨轮次保留所有思维块
   - **Turn-level Thinking**：按轮次控制是否启用推理
4. **推理 RL**：基于 GRPO + IcePop 技术，混合数学、科学、代码和工具集成推理 (TIR) 四领域训练
5. **Agent RL**：全异步解耦 RL 框架，编码和搜索 Agent 任务
6. **通用 RL**：三维度优化（基础正确性、情感智能、任务质量），混合奖励系统（规则、ORM、GRM），人类对齐
7. **On-Policy 跨阶段蒸馏**：最终阶段，防止多阶段 RL 导致的能力退化

### 异步 RL 基础设施（slime 框架）

- 解耦训练引擎和推理引擎，减少 Agent rollout 长尾导致的 GPU 空闲
- **Token-in-Token-out (TITO) 网关**：避免重新分词导致的不匹配
- **直接双侧重要性采样**：token 级裁剪机制控制 off-policy 偏差
- **DP-aware 路由**：最大化 MoE 推理中 KV-cache 复用
- 支持 1000+ 并发 rollout，自动调整任务采样比例

### Vibe Coding vs Agentic Engineering

| 维度 | Vibe Coding | Agentic Engineering |
|------|------------|-------------------|
| 交互模式 | 人类提示，AI 写代码 | AI Agent 自主规划、实现、迭代 |
| 任务范围 | 单轮、局部代码生成 | 长时程、端到端软件工程 |
| 关键能力 | 代码补全、指令遵循 | 长上下文一致性、自我纠错、环境交互 |
| 评估方式 | 静态基准（SWE-bench 等） | 真实工程环境（CC-Bench-V2） |

### Agent 环境构建

- **软件工程环境**：基于 RepoLaunch 框架，从真实 SWE issue 构建超过 1 万个可验证环境，覆盖 9 种编程语言
- **终端环境**：从种子数据和网络语料两条管线合成，Docker 构建准确率超过 90%
- **搜索任务**：构建 Web Knowledge Graph (WKG)，自动生成高难度多跳 QA 对，三阶段过滤确保质量
- **幻灯片生成**：多层级奖励设计（静态标记、运行时渲染、视觉感知），自改进管线

### 搜索 Agent 上下文管理

- 提出 **Hierarchical Context Management** 策略：Keep-recent-k（保留最近 k 轮）+ Discard-all（超阈值 T 时清空）
- BrowseComp 从 55.3% 提升至 62.0%（keep-recent），最终达 75.9%（分层管理）

### 中国芯片适配

- 全栈适配 7 个主流国产芯片平台（华为昇腾、摩尔线程、海光、寒武纪、昆仑芯、天数智芯、壁仞）
- 混合精度 W4A8 量化、高性能融合算子（Lightning Indexer、Sparse Flash Attention、MLAPO）
- 单节点国产芯片性能对标双卡国际集群，长序列场景部署成本降低 50%

## 主要结果

### ARC 基准对比（Table 7 摘要）

| 基准 | GLM-5 | Claude Opus 4.5 | GPT-5.2 (xhigh) | Gemini 3 Pro |
|------|-------|-----------------|------------------|--------------|
| HLE (w/ Tools) | 50.4 | 43.4* | 45.5* | 45.8* |
| SWE-bench Verified | 77.8 | 80.9 | 80.0 | 76.2 |
| SWE-bench Multilingual | 73.3 | 77.5 | 72.0 | 65.0 |
| Terminal-Bench 2.0 | 60.7† | 59.3 | 54.0 | 54.2 |
| BrowseComp (w/ CM) | 75.9 | 57.8 | 65.8 | 59.2 |
| τ²-Bench | 89.7 | 91.6 | 85.5 | 90.7 |
| Vending-Bench 2 | $4,432 | $4,967 | $3,591 | $5,478 |

- **Artificial Analysis Intelligence Index v4.0**：GLM-5 得分 50，首次有开放权重模型达到此分数
- **LMArena**：Text Arena 和 Code Arena 双料开源第一
- **长时程任务**：Vending-Bench 2 开源第一，CC-Bench-V2 显著超越 GLM-4.7

### CC-Bench-V2（真实工程评估）

- **前端**：构建成功率 98%（React/Vue/Svelte/Next.js），CSR 接近 Claude Opus 4.5
- **后端**：85 个任务，6 种语言，Pass@1 为 25.8（与 Claude Opus 4.5 的 26.9 可比）
- **长时程**：Repo Exploration 65.6%（超过 Claude Opus 4.5 的 64.5%），Chained Tasks 52.3%

### SWE-rebench（持续更新的去污染评估）

- GLM-5 达 42.1% resolved rate，有效泛化到新的 SWE 问题

### "Pony Alpha" 匿名发布

- GLM-5 以 "Pony Alpha" 名义匿名发布在 OpenRouter，社区猜测 25% 认为是 Claude Sonnet 5、20% 认为是 DeepSeek、10% 认为是 Grok

## 局限性

1. **端到端任务完成率 (ISR) 差距**：前端评估中 GLM-5 虽然 CSR 接近 Claude Opus 4.5，但在完整任务端到端完成率上仍有明显差距
2. **长时程链式任务**：CC-Bench-V2 的 Chained Tasks 中 GLM-5（52.3%）与 Claude Opus 4.5（61.6%）存在显著差距，错误会在链中累积
3. **工具调用**：Tool-Decathlon（39.2%）落后于 Claude Opus 4.5（43.5%）和 GPT-5.2（46.3%）
4. **部分推理基准**：GPQA-Diamond（86.0%）落后于 Gemini 3 Pro（91.9%）和 GPT-5.2（92.4%）
5. **SWE-rebench**：42.1% 落后于 Claude Opus 4.6（52.9%）和 GPT-5.2（51.7%）

## 与相关工作的关系

- **[[Wiki/Sources/DeepSeek-V3 技术报告|DeepSeek-V3/V3.2]]**：GLM-5 架构上采用 DeepSeek 提出的 MLA 和 DSA，但在 Muon Split、MTP 参数共享等方面做了改进；DSA 适配仅需 20B tokens（vs DeepSeek-V3.2 的 943.7B）
- **[[Wiki/Sources/DeepSeek-R1 强化学习推理|DeepSeek-R1]]**：推理 RL 阶段借鉴 GRPO 算法并引入 IcePop 技术
- **Claude Opus 4.5**：主要对标模型，在多数基准上互有胜负
- **GPT-5.2 (xhigh)**：在部分推理和工具调用基准上领先 GLM-5
- **Kimi K2.5**：开源竞品，在 HLE 等推理任务上略优于 GLM-5
- **slime 框架**：GLM-4.5 时期建立的统一后训练基础设施，GLM-5 进一步扩展其异步能力
- **RepoLaunch / Harbor**：Agent 环境构建的基础设施框架

## 综合判断

GLM-5 是目前开源权重模型中综合能力最强的之一，其核心贡献在于：（1）证明了从 vibe coding 到 agentic engineering 的范式转变在工程上可行；（2）通过 DSA 等技术实现了效率与性能的平衡；（3）建立了异步 Agent RL 的完整训练管线。但与顶级闭源模型相比，在端到端任务完成率和长时程链式任务上仍有差距，这也是未来研究的重点方向。

---
type: source
status: complete
created: '2026-05-07'
updated: '2026-05-07'
tags: [agent, llm, survey, autonomous-agent]
sources:
  - "[[Clippings/The Rise and Potential of Large Language Model Based Agents: A Survey]]"
---
# The Rise and Potential of Large Language Model Based Agents: A Survey

## 基本信息

- **作者**：Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He 等（复旦大学 NLP 组）
- **发表**：arXiv 2309.07864，2023 年 9 月
- **类型**：综述论文，LLM-based Agent 领域首篇全面系统综述之一
- **配套资源**：GitHub 论文列表 [LLM-Agent-Paper-List](https://github.com/WooooDyy/LLM-Agent-Paper-List)

## 核心论点

1. **LLM 是构建通用 AI Agent 的理想基础**：LLM 在知识获取、指令理解、泛化、推理规划和自然语言交互方面展现出强大能力，被认为是 AGI 的火花，适合作为 Agent 大脑的核心组件。
2. **三模块通用框架**：提出 Brain-Perception-Action 概念框架——Brain（LLM 驱动的知识、记忆、推理、规划）、Perception（多模态感知扩展）、Action（文本输出、工具使用、具身动作）。
3. **从个体到社会**：Agent 不仅能独立执行任务，还能通过多智能体协作/对抗产生集体智能，甚至涌现出社会现象，形成"Agent 社会"。
4. **哲学根源与 AI 实体化**：从亚里士多德和休谟的"agent"哲学概念出发，追溯到图灵测试，论证 AI agent 是哲学概念在 AI 领域的具体化。

## 关键技术方法

### Agent 框架（Brain-Perception-Action）

**Brain（大脑模块）**：
- **自然语言交互**：多轮对话、高质量生成、意图与隐含意义理解
- **知识**：语言知识、常识知识、专业领域知识；存在知识过时和幻觉问题
- **记忆**：提升 Transformer 长度限制、记忆摘要、向量/数据结构压缩；检索策略基于 Recency-Relevance-Importance 三指标加权
- **推理与规划**：CoT 引导推理；规划分为 Plan Formulation（一次性分解/逐步分解/层级规划/树搜索）和 Plan Reflection（内部反馈/人类反馈/环境反馈）
- **迁移与泛化**：零样本任务泛化（instruction tuning）、In-context Learning、持续学习（Voyager 的技能库机制对抗灾难性遗忘）

**Perception（感知模块）**：
- 文本输入 → 视觉输入（图像描述、ViT 编码器、Q-Former 对齐、视频时序理解） → 听觉输入（AudioGPT 级联范式、AST 音频频谱 Transformer） → 其他输入（触觉、手势、3D 地图、GPS、LiDAR）

**Action（行动模块）**：
- 文本输出
- 工具使用：理解工具 → 学习使用（示范学习/反馈学习） → 自主制造工具；工具扩展了专业知识、可解释性、鲁棒性
- 具身动作：观察（多模态感知）、操控（物体重排、桌面操作）、导航（内部地图构建、空间定位）

### 应用场景

**单智能体**：
- 任务导向：网页导航（Mind2Web、WebGum）、生活场景（PET 框架）
- 创新导向：化学/材料合成、药物发现、代码开发
- 生命周期导向：Minecraft 中 Voyager 实现终身学习探索

**多智能体系统**：
- 协作交互：无序协作（ChatLLM network、多数投票）和有序协作（CAMEL 双智能体、MetaGPT 瀑布模型、AgentVerse 动态团队）
- 对抗交互：辩论机制提升推理质量（Du et al. 的 debate、ChatEval 评估）

**人机协作**：
- 指导者-执行者范式：量化反馈（二值/评分/比较）和质性反馈（文本建议/视觉批评）
- 平等伙伴范程：共情沟通者（情感对话）、人类级参与者（游戏协作、说服能力）

### Agent 社会

- **行为与人格**：个体行为（输入/内化/输出）、群体行为（积极/中性/消极）、认知能力、情商、性格刻画（Big Five、MBTI）
- **社会环境**：文本环境、虚拟沙盒（Generative Agents 小镇）、物理环境
- **社会仿真**：开放性、持久性、情境性、组织性四大属性；涌现现象包括有序合作、网络传播、伦理决策、政策制定

## 主要结果

- 系统梳理了 LLM-based Agent 从哲学起源到技术实现的完整脉络
- 提出的 Brain-Perception-Action 框架成为后续研究的重要参考架构
- 全面覆盖 2023 年及之前的主要 Agent 工作，包括 AutoGPT、CAMEL、Voyager、MetaGPT、Generative Agents 等
- 讨论了评估维度（Utility/Sociability/Values/Evolution）、安全风险（对抗鲁棒性/可信度/滥用/失业/人类福祉威胁）和开放问题（AGI 路径争论/虚拟到物理的鸿沟/集体智能/Agent as a Service）

## 局限性

- **时效性**：发表于 2023 年 9 月，未覆盖 2024 年以后的快速发展（如 GPT-4o、Claude 3、DeepSeek 系列、Kimi Agent 等）
- **深度不足**：作为综述覆盖面广但对各子领域的技术细节讨论有限，特别是工具使用和具身动作部分
- **评估框架偏抽象**：提出的四维评估（Utility/Sociability/Values/Evolution）缺乏具体 benchmark 和量化标准
- **未充分讨论开源 Agent 框架**：对 LangChain、LlamaIndex 等工程生态着墨不多
- **多模态 Agent 讨论较浅**：视觉/听觉感知部分主要是方向性介绍，缺乏对具体多模态模型架构的深入分析

## 与相关工作的关系

### 与 [[Wiki/Sources/LLM Agent 综述 2024]]（人大综述）的关系

| 维度 | 本综述（复旦 2023） | 人大综述（2024） |
|------|-------------------|-----------------|
| 框架 | Brain-Perception-Action 三模块 | Profile-Memory-Planning-Action 四模块 |
| 侧重 | 哲学溯源 + 社会仿真 + 通用框架 | 架构分类 + 能力获取方法 + 工程实践 |
| 独特贡献 | Agent 社会、人格涌现、集体智能 | Profile 角色设定、能力获取分类（微调 vs 非微调） |
| 互补性 | 本综述提供更宏观的视野和哲学基础 | 人大综述提供更细粒度的架构分析和实现路径 |

两篇综述框架有重叠但视角不同：复旦综述强调从个体智能到社会智能的演进，人大综述强调从架构设计到工程落地的方法论。

### 与 [[Wiki/Sources/Agent AI Survey 2024]]（Agent AI 综述）的关系

| 维度 | 本综述（复旦 2023） | Agent AI 综述（2024） |
|------|-------------------|---------------------|
| 范围 | 以 LLM 为核心的通用 Agent | 多模态交互为核心的 Agent AI |
| 模态 | 重点在语言，感知模块作为扩展 | 视觉-语言-环境数据深度融合 |
| 独特贡献 | Agent 社会仿真、哲学讨论 | 跨现实训练框架、Embodied AI、游戏/机器人/医疗 |
| 互补性 | 本综述是 LLM Agent 的"总纲" | Agent AI 是多模态/具身方向的深入拓展 |

复旦综述为整个 LLM Agent 领域提供了概念基础和分类框架，Agent AI 综述则在多模态交互和具身智能方向做了更深入的探索。三篇综述共同构成了 2023-2024 年 LLM Agent 研究的知识图谱。

## 后续问题

- Brain-Perception-Action 框架在 2024-2025 年的实践中是否仍然适用？与人大四模块框架相比哪个更好用？
- Agent 社会仿真的规模瓶颈在哪里？从 Generative Agents 的 25 个 Agent 到更大规模需要什么突破？
- LLM-based Agent 到底是不是 AGI 的可行路径？2024-2025 年的进展（推理模型、工具使用、Agent 框架）对这个问题提供了什么新证据？

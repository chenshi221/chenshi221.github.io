---
type: log
status: active
created: '2026-04-29'
updated: '2026-05-06'
tags:
  - meta
---

# Wiki Log

## [2026-04-29] init | Wiki 初始化

- Created: Wiki 目录结构及基础页面
- Created: [[Wiki/overview]], [[Wiki/index]], [[Wiki/log]]
- Created: [[Wiki/Contradictions]]
- Notes: 初始化 Wiki 结构，准备来源摄取工作流。

## [2026-04-29] ingest | llm-wiki

- Source: [[Clippings/llm-wiki.md]]
- Created: [[Wiki/Sources/llm-wiki]]
- Created: [[Wiki/Concepts/llm-wiki-pattern]]
- Updated: [[Wiki/overview]], [[Wiki/index]]
- Notes: Karpathy 的 LLM Wiki 模式——本知识库的架构蓝图。该文档描述了 LLM 增量维护个人知识库的方法，与本 vault 的 CLAUDE.md 高度一致，构成元层指导。

## [2026-04-29] ingest | DreamOmni2

- Source: [[Clippings/DreamOmni2 Multimodal Instruction-based Editing and Generation.md]]
- Created: [[Wiki/Sources/DreamOmni2]]
- Created: [[Wiki/Entities/DreamOmni2]]
- Created: [[Wiki/Concepts/多模态指令编辑与生成]]
- Created: [[Wiki/Topics/扩散模型图像编辑与生成]]
- Updated: [[Wiki/overview]], [[Wiki/index]]
- Notes: DreamOmni2 提出了多模态指令编辑和生成两个新任务，通过三步数据合成管线 + 索引编码/位置编码偏移 + VLM 联合训练的框架，在 benchmark 上超越 GPT-4o 和 Nano Banana。该工作将统一图像编辑生成从单图像+文本扩展到多图像+多模态指令。

## [2026-04-30] ingest | 统一多模态模型批量摄取

- Sources:
  - [[Clippings/Emerging Properties in Unified Multimodal Pretraining.md]]
  - [[Clippings/UniWorld-V1 High-Resolution Semantic Encoders for Unified Visual Understanding and Generation.md]]
  - [[Clippings/Tuna-2 Pixel Embeddings Beat Vision Encoders for Multimodal Understanding and Generation.md]]
  - [[Clippings/L u m i n a - D i M O O An Omni Diffusion Large Language Model for Multi-Modal Generation and Understanding.md]]
  - [[Clippings/OmniGen2 Towards Instruction-Aligned Multimodal Generation.md]]
  - [[Clippings/Show-o2 Improved Native Unified Multimodal Models.md]]
  - [[Clippings/Unified Multimodal Understanding and Generation Models Advances, Challenges, and Opportunities.md]]
- Created:
  - Sources: [[Wiki/Sources/BAGEL]], [[Wiki/Sources/UniWorld-V1]], [[Wiki/Sources/Tuna-2]], [[Wiki/Sources/Lumina-DiMOO]], [[Wiki/Sources/OmniGen2]], [[Wiki/Sources/Show-o2]], [[Wiki/Sources/Unified Multimodal Survey]]
  - Entities: [[Wiki/Entities/BAGEL]], [[Wiki/Entities/UniWorld-V1]], [[Wiki/Entities/Tuna-2]], [[Wiki/Entities/Lumina-DiMOO]], [[Wiki/Entities/OmniGen2]], [[Wiki/Entities/Show-o2]]
  - Comparisons: [[Wiki/Comparisons/统一多模态模型架构比较]], [[Wiki/Comparisons/README]]
  - Questions: [[Wiki/Questions/关于统一多模态模型的问题]]
- Updated:
  - [[Wiki/Topics/扩散模型图像编辑与生成]] — 全面扩展，加入统一多模态模型架构对比表格和涌现能力描述
  - [[Wiki/index]]
- Notes: 批量摄取了 7 个 Clippings，全部围绕同一主题——统一多模态理解与生成模型（UMMs）。这些来源共同勾勒了 2025 年这一方向的完整图景：从 BAGEL 的超大规模涌现能力到 Tuna-2 的极致简化，从 UniWorld-V1 的数据效率到 Lumina-DiMOO 的离散扩散速度优势。一次处理后，Wiki 的知识从单一 DreamOmni2 扩展为覆盖 7 个主要 UMM 模型、1 篇综述和 1 份比较的结构化知识体系。

## [2026-04-30] lint | 基于 CLAUDE.md 的 Wiki 体检与更新

- Checked: 全部 Wiki 页面 frontmatter、overview 内容完整性、Clippings 处理状态
- Fixed:
  - [[Wiki/index]] — 添加 frontmatter (type: index)
  - [[Wiki/log]] — 添加 frontmatter (type: log)
  - [[Wiki/overview]] — 修正来源计数（10→9）、添加操作框架说明、引用 [[CLAUDE.md]]、更新最近动态
- Open issues:
  - 未处理 Clippings：20 个（含 1 个疑似重复：DreamOmni2 有两个版本，创建时间差 2 天，大小微差）
  - overview.md 的 `type: overview` 不在 CLAUDE.md 定义的 frontmatter 类型列表中（source/topic/concept/entity/question/comparison/synthesis/contradiction）
- Suggested: 下一步可逐个摄取未处理的 Clippings，优先处理与已有主题（图像编辑、情感计算、统一多模态）直接相关的文件


## [2026-04-30] ingest | 情感计算与图像生成批量摄取

- Sources:
  - [[Clippings/EmoEdit Evoking Emotions through Image Manipulation.md]]
  - [[Clippings/EmoArt A Multidimensional Dataset for Emotion-Aware Artistic Generation.md]]
  - [[Clippings/EmotiCrafter Text-to-Emotional-Image Generation based on Valence-Arousal Model.md]]
  - [[Clippings/Affective Image Editing Shaping Emotional Factors via Text Descriptions.md]]
  - [[Clippings/Generating Fearful Images Investigating Potential Emotional Biases in Image-Generation Models.md]]
- Created:
  - Sources: [[Wiki/Sources/EmoEdit]], [[Wiki/Sources/EmoArt]], [[Wiki/Sources/EmotiCrafter]], [[Wiki/Sources/Affective Image Editing]], [[Wiki/Sources/Generating Fearful Images]]
  - Concepts: [[Wiki/Concepts/情感图像编辑]], [[Wiki/Concepts/Valence-Arousal 情感模型]], [[Wiki/Concepts/AI 生成图像的情感偏差]]
  - Entities: [[Wiki/Entities/EmoEdit]], [[Wiki/Entities/EmoArt]], [[Wiki/Entities/EmotiCrafter]]
  - Topics: [[Wiki/Topics/情感计算与图像生成]]
- Updated: [[Wiki/index]]
- Notes: 一次性摄入了 5 篇情感计算相关 Clippings，新建了「情感计算与图像生成」主题。这批文献展示了从离散到连续的情感表示演进（EmoEdit 的 8 类 → EmotiCrafter 的 V-A 坐标），以及从方法到审计的视角拓展（Generating Fearful Images 的情感偏差问题）。EmoArt 的 132K 艺术数据集为该方向提供了重要的 benchmark 基础。剩余 15 个 Clippings 待处理。


## [2026-04-30] ingest | 图像编辑全栈批量摄取（15 篇）

- Sources:
  - 编辑基础: [[Clippings/InstructPix2Pix Learning to Follow Image Editing Instructions.md]], [[Clippings/Adding Conditional Control to Text-to-Image Diffusion Models.md]]
  - 编辑数据: [[Clippings/UltraEdit Instruction-based Fine-Grained Image Editing at Scale.md]], [[Clippings/AnyEdit Mastering Unified High-Quality Image Editing for Any Idea.md]], [[Clippings/ImgEdit A Unified Image Editing Dataset and Benchmark.md]], [[Clippings/OpenGPT-4o-Image A Comprehensive Dataset for Advanced Image Generation and Editing.md]]
  - 编辑方法: [[Clippings/EditWorld Simulating World Dynamics for Instruction-Following Image Editing.md]], [[Clippings/Step1X-Edit A Practical Framework for General Image Editing.md]]
  - 推理生成: [[Clippings/GoT Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing.md]], [[Clippings/Mind-Brush Integrating Agentic Cognitive Search and Reasoning into Image Generation.md]], [[Clippings/VisionCreator A Native Visual-Generation Agentic Model with Understanding, Thinking, Planning and Creation.md]]
  - Benchmark: [[Clippings/Envisioning Beyond the Pixels Benchmarking Reasoning-Informed Visual Editing.md]], [[Clippings/WEAVE Unleashing and Benchmarking the In-context Interleaved Comprehension and Generation.md]]
  - 控制+审美: [[Clippings/OminiControl Minimal and Universal Control for Diffusion Transformer.md]], [[Clippings/Unlocking the Essence of Beauty Advanced Aesthetic Reasoning with Relative-Absolute Policy Optimization.md]]
- Created:
  - Sources (15): [[Wiki/Sources/InstructPix2Pix]], [[Wiki/Sources/ControlNet]], [[Wiki/Sources/UltraEdit]], [[Wiki/Sources/AnyEdit]], [[Wiki/Sources/EditWorld]], [[Wiki/Sources/Step1X-Edit]], [[Wiki/Sources/GoT]], [[Wiki/Sources/Mind-Brush]], [[Wiki/Sources/VisionCreator]], [[Wiki/Sources/RISEBench]], [[Wiki/Sources/ImgEdit]], [[Wiki/Sources/WEAVE]], [[Wiki/Sources/OpenGPT-4o-Image]], [[Wiki/Sources/OminiControl]], [[Wiki/Sources/Aes-R1]]
- Updated:
  - [[Wiki/Topics/扩散模型图像编辑与生成]] — 大幅扩展，新增指令编辑演进、推理 Agent、Benchmark 生态、审美评估等章节
  - [[Wiki/index]], [[Wiki/overview]]
- Notes: 一次性将所有 15 个剩余 Clippings 全部摄入。完整的图像编辑知识脉络已构建：从 InstructPix2Pix 奠基 → ControlNet 控制 → UltraEdit/AnyEdit 数据扩展 → EditWorld 物理感知 → GoT/Mind-Brush/VisionCreator 推理 Agent → RISEBench/WEAVE 等 benchmark 生态 → Aes-R1 审美指导。所有 Clippings（29 个）已全部处理完毕。Wiki 现覆盖 29 个来源、两个主题（统一多模态 + 情感计算），共 50+ 页面。


## [2026-04-30] refactor | 深化分析与清理

- Deleted: `Clippings/DreamOmni2 Multimodal Instruction-based Editing and Generation 1.md`（重复文件）
- Enhanced:
  - [[Wiki/Comparisons/统一多模态模型架构比较]] — 重写为深度分析页，新增编码器之争、离散vs连续、涌现vs高效、统一vs专用的分析判断
  - [[Wiki/Comparisons/指令编辑数据集比较]] — 新建，7 个编辑数据集全维度对比 + 五种范式演进 + 合成vs真实思考
  - [[Wiki/Comparisons/情感表示方法比较]] — 新建，离散vs连续深度对比 + Arousal 为什么更难 + 融合方案
  - [[Wiki/Comparisons/编辑方法能力演进]] — 新建，五代能力矩阵 + 四个关键转折点 + 推理必要性/Agent边界分析
  - [[Wiki/Topics/扩散模型图像编辑与生成]] — 新增"三个元问题"思考：编辑的本质、统一模型的终点、Agent 是过度设计还是必然
  - [[Wiki/Topics/情感计算与图像生成]] — 新增深度分析：情感编辑真正解决的问题、数据集方向、评估困局、情感偏差被低估
- Updated: [[Wiki/index]], [[Wiki/Comparisons/README]]
- Notes: 从"事实整理"升级为"分析思考"。所有比较页和主题页现在包含我的独立判断，而非仅罗列论文结论。Wiki 质量从"信息聚合"提升为"知识 + 洞见"。


## [2026-04-30] query | BAGEL 是否支持图文交错生成

- Question: 用户询问 BAGEL 是否支持图文交错生成
- Answer: ✅ 支持。BAGEL 的 Generalized Causal Attention + Diffusion Forcing 策略使其能在单次推理中输出文本-图像-文本交错序列。这是 MoT 集成架构相对于 VLM+Diff 组装式方案的独特优势。
- Created: [[Wiki/Questions/BAGEL 图文交错生成能力]]
- Updated:
  - [[Wiki/Entities/BAGEL]] — 大幅扩展，新增交错生成详解、Generalized Causal Attention 机制、涌现模式表
  - [[Wiki/Sources/BAGEL]] — 补充交错生成能力条目
  - [[Wiki/Comparisons/统一多模态模型架构比较]] — 新增"交错生成：架构决定能力边界"分析节 + 对比行
  - [[Wiki/Topics/扩散模型图像编辑与生成]] — 核心对比表新增交错生成行
  - [[Wiki/index]]
- Notes: 这次查询揭示了一个重要洞察——架构选择（集成 vs 组装）直接决定了能力边界（是否支持交错生成），而不仅仅是性能差异。组装式方案无论如何优化都无法支持单次推理中的交错输出。


## [2026-04-30] ingest | LLM 基础模型与推理方法批量摄取（10 篇）

- Sources:
  - [[Clippings/Attention Is All You Need.md]]
  - [[Clippings/BERT Pre-training of Deep Bidirectional Transformers for Language Understanding.md]]
  - [[Clippings/Language Models are Few-Shot Learners.md]]
  - [[Clippings/Training language models to follow instructions with human feedback.md]]
  - [[Clippings/LLaMA Open and Efficient Foundation Language Models.md]]
  - [[Clippings/Training Compute-Optimal Large Language Models.md]]
  - [[Clippings/GPT-4 Technical Report.md]]
  - [[Clippings/GPT-4o System Card.md]]
  - [[Clippings/The Llama 3 Herd of Models.md]]
  - [[Clippings/RoFormer Enhanced Transformer with Rotary Position Embedding.md]]
- Created:
  - Sources (10): [[Wiki/Sources/Transformer]], [[Wiki/Sources/BERT]], [[Wiki/Sources/GPT-3]], [[Wiki/Sources/InstructGPT]], [[Wiki/Sources/LLaMA]], [[Wiki/Sources/Chinchilla 缩放定律]], [[Wiki/Sources/GPT-4]], [[Wiki/Sources/GPT-4o]], [[Wiki/Sources/Llama 3]], [[Wiki/Sources/RoPE]]
  - Topics (1): [[Wiki/Topics/大语言模型基础]]
  - Entities (1): [[Wiki/Entities/Transformer]]
  - Concepts (4): [[Wiki/Concepts/Scaling Laws]], [[Wiki/Concepts/RLHF]], [[Wiki/Concepts/RoPE 旋转位置编码]], [[Wiki/Concepts/GPT 系列模型]]
- Updated: [[Wiki/index]]
- Notes: 一次摄入了 10 篇 LLM 基础核心文献，覆盖从 Transformer (2017) 到 Llama 3 (2024) 的完整发展脉络。新建了「大语言模型基础」主题，串联架构演进、缩放定律和对齐方法三条主线。4 个新概念页深度覆盖了 Scaling Laws、RLHF、RoPE 和 GPT 系列模型。这是 Wiki 首次系统性地整理 NLP/LLM 基础，与已有的图像编辑和情感计算主题形成互补——理解 MLLM 图像编辑方法的前提是理解这些 LLM 基础。

## [2026-04-30] ingest | Agent + Embedding + Benchmark 批量摄取（10 篇）

- Sources:
  - Agent: [[Clippings/A Survey on Large Language Model based Autonomous Agents]], [[Clippings/Agent AI Surveying the Horizons of Multimodal Interaction]], [[Clippings/Agent Banana High-Fidelity Image Editing with Agentic Thinking and Tooling]]
  - Embedding: [[Clippings/Magic-MM-Embedding Towards Visual-Token-Efficient Universal Multimodal Embedding with MLLMs]], [[Clippings/ObjEmbed Towards Universal Multimodal Object Embeddings]], [[Clippings/RzenEmbed Towards Comprehensive Multimodal Retrieval]], [[Clippings/SAIL-Embedding Technical Report Omni-modal Embedding Foundation Model]]
  - Benchmark: [[Clippings/OCRBench v2 An Improved Benchmark for Evaluating Large Multimodal Models on Visual Text Localization and Reasoning]], [[Clippings/OmniDocBench Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations]], [[Clippings/pipeline Unlocking Trillions of Tokens in PDFs with Vision Language Models]]
- Created:
  - Sources (10): [[Wiki/Sources/LLM Agent 综述 2024]], [[Wiki/Sources/Agent AI Survey 2024]], [[Wiki/Sources/Agent Banana]], [[Wiki/Sources/Magic-MM-Embedding]], [[Wiki/Sources/ObjEmbed]], [[Wiki/Sources/RzenEmbed]], [[Wiki/Sources/SAIL-Embedding]], [[Wiki/Sources/OCRBench v2]], [[Wiki/Sources/OmniDocBench]], [[Wiki/Sources/olmOCR]]
  - Topics (3): [[Wiki/Topics/LLM Agent 与工具使用]], [[Wiki/Topics/多模态 Embedding 与检索]], [[Wiki/Topics/多模态 Benchmark 与评估]]
  - Concepts (3): [[Wiki/Concepts/LLM Agent 架构]], [[Wiki/Concepts/多模态 Embedding 模型]], [[Wiki/Concepts/PDF 文档解析]]
  - Entities (1): [[Wiki/Entities/Agent Banana]]
  - Comparisons (1): [[Wiki/Comparisons/多模态 Embedding 模型比较]]
- Updated: [[Wiki/index]]
- Notes: 一次性摄入了 10 个 Clippings，覆盖三个新方向：LLM Agent 架构理论、多模态 Embedding 模型、多模态 Benchmark 评估体系。Agent Banana 特别链接到已有的 [[Wiki/Topics/扩散模型图像编辑与生成]]，形成 Agentic 编辑与扩散模型的交叉。4 个 Embedding 模型覆盖了效率优化、物体级表示、多模态扩展和工业落地四个维度，通过比较页进行系统性对比。OCRBench v2 + OmniDocBench + olmOCR 形成了"测什么 → 怎么测 → 用什么工具"的完整 PDF 文档处理知识链。

## [2026-04-30] ingest | 国产大模型系列批量摄取（9 篇）

- Sources:
  - [[Clippings/DeepSeek LLM Scaling Open-Source Language Models with Longtermism.md]]
  - [[Clippings/DeepSeek-V3 Technical Report.md]]
  - [[Clippings/DeepSeek-V3.2 Pushing the Frontier of Open Large Language Models.md]]
  - [[Clippings/DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.md]]
  - [[Clippings/Kimi k1.5 Scaling Reinforcement Learning with LLMs.md]]
  - [[Clippings/Kimi K2 Open Agentic Intelligence.md]]
  - [[Clippings/Kimi K2.5 Visual Agentic Intelligence.md]]
  - [[Clippings/Kimi-VL Technical Report.md]]
  - [[Clippings/Qwen3 Technical Report.md]]
- Created:
  - Sources (9): [[Wiki/Sources/DeepSeek LLM 开源语言模型与长期主义]], [[Wiki/Sources/DeepSeek-V3 技术报告]], [[Wiki/Sources/DeepSeek-V3.2 开源大模型前沿]], [[Wiki/Sources/DeepSeek-R1 强化学习推理]], [[Wiki/Sources/Kimi k1.5 强化学习规模化]], [[Wiki/Sources/Kimi K2 开放 Agent 智能]], [[Wiki/Sources/Kimi K2.5 视觉 Agent 智能]], [[Wiki/Sources/Kimi-VL 技术报告]], [[Wiki/Sources/Qwen3 技术报告]]
  - Topics (1): [[Wiki/Topics/国产大模型演进]]
  - Entities (3): [[Wiki/Entities/DeepSeek 系列模型]], [[Wiki/Entities/Kimi 系列模型]], [[Wiki/Entities/Qwen3]]
  - Concepts (3): [[Wiki/Concepts/MoE 混合专家模型]], [[Wiki/Concepts/推理模型与强化学习]], [[Wiki/Concepts/多模态 Agent]]
- Updated: [[Wiki/index]]
- Notes: 一次性摄入了 9 个 Clippings，覆盖 DeepSeek、Kimi、Qwen 三条国产大模型主线。三条线共同趋势：(1) MoE 成为大模型主流架构，各有特色（DeepSeekMoE 的 aux-loss-free、Kimi 的 sparsity scaling law、Qwen3 的 Dense+MoE 双线）；(2) RL 驱动推理能力训练是共同选择（GRPO vs online mirror descent vs 四阶段管线），但推理模式存在分歧（专用推理模型 vs 统一双模式 vs long2short 蒸馏）；(3) Agent 能力成为新焦点（K2 的数据合成、K2.5 的 Agent Swarm、V3.2 的 thinking-in-tool-use）。三个新概念页深度交叉对比了三家方案，包括 MoE 架构的负载均衡、训练稳定性、推理效率等关键挑战；RL 推理训练中的奖励设计、训练稳定性、推理长度控制、泛化性问题；多模态 Agent 的工具使用、多步规划、反思纠错、数据合成等核心要素。Wiki 现覆盖 49 个来源、6 个主题、18 个概念、23 个实体。


## [2026-04-30] ingest | Vision 基础模型 + 多模态对比学习 + 推理增强方法批量摄取（11 篇）

- Sources:
  - Vision 基础模型: [[Clippings/An Image is Worth 16x16 Words Transformers for Image Recognition at Scale.md]], [[Clippings/Swin Transformer Hierarchical Vision Transformer using Shifted Windows.md]], [[Clippings/MLP-Mixer An all-MLP Architecture for Vision.md]]
  - 多模态对比学习: [[Clippings/CLAP  Learning Audio Concepts From Natural Language Supervision.md]], [[Clippings/SigLIP 2 Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features.md]]
  - 原生多模态: [[Clippings/Emu3.5 Native Multimodal Models are World Learners.md]]
  - 推理方法: [[Clippings/Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.md]], [[Clippings/Tree of Thoughts Deliberate Problem Solving with Large Language Models.md]], [[Clippings/Graph of Thoughts Solving Elaborate Problems with Large Language Models.md]], [[Clippings/Competitive Programming with Large Reasoning Models.md]]
  - Benchmark: [[Clippings/WorldEdit Towards Open-World Image Editing with a Knowledge-Informed Benchmark.md]]
- Created:
  - Sources (11): [[Wiki/Sources/ViT]], [[Wiki/Sources/Swin Transformer]], [[Wiki/Sources/MLP-Mixer]], [[Wiki/Sources/CLAP]], [[Wiki/Sources/SigLIP 2]], [[Wiki/Sources/Emu3.5]], [[Wiki/Sources/Chain-of-Thought]], [[Wiki/Sources/Tree of Thoughts]], [[Wiki/Sources/Graph of Thoughts]], [[Wiki/Sources/Competitive Programming LM]], [[Wiki/Sources/WorldEdit]]
  - Topics (2): [[Wiki/Topics/Vision Transformer 演进]], [[Wiki/Topics/推理增强方法]]
  - Concepts (3): [[Wiki/Concepts/Chain-of-Thought 思维链]], [[Wiki/Concepts/多模态对比学习]], [[Wiki/Concepts/原生多模态模型]]
  - Entities (2): [[Wiki/Entities/Vision Transformer (ViT)]], [[Wiki/Entities/Swin Transformer]]
- Updated: [[Wiki/index]]
- Notes: 一次性摄入了 11 篇 Clippings，涉及三个新方向：(1) Vision Transformer 架构演进（ViT→Swin→MLP-Mixer），建立了从 2020 到 2021 的视觉基础模型叙事；(2) 多模态对比学习（CLAP/SigLIP 2/Emu3.5），补充了音频对比学习和原生多模态世界模型的视角；(3) LLM 推理增强方法（CoT→ToT→GoT→推理模型），完整覆盖了从提示工程到 RL 训练的推理演进。WorldEdit 填补了因果推理编辑 benchmark 的空白，与已有编辑生态互补。


## [2026-04-30] ingest | 全量 Clippings 摄取完成（40 篇新摄入）

- Sources: 全部 40 个新 Clippings 已通过 4 个并行 Agent 处理完毕
  - Agent 1: LLM 基础模型（10 篇：Transformer/BERT/GPT-3/InstructGPT/LLaMA/Chinchilla/GPT-4/GPT-4o/Llama 3/RoPE）
  - Agent 2: 国产大模型（9 篇：DeepSeek×4 + Kimi×4 + Qwen3）
  - Agent 3: Vision+多模态+推理方法+WorldEdit（11 篇：ViT/Swin/MLP-Mixer/CLAP/SigLIP 2/Emu3.5/CoT/ToT/GoT/Competitive/WorldEdit）
  - Agent 4: Agent+Embedding+Benchmark（10 篇：Agent综述×2/Agent Banana/Embedding×4/Benchmark×3）
- Created:
  - Sources (40): [[Wiki/Sources/Transformer]], [[Wiki/Sources/BERT]], [[Wiki/Sources/GPT-3]], [[Wiki/Sources/InstructGPT]], [[Wiki/Sources/LLaMA]], [[Wiki/Sources/Chinchilla 缩放定律]], [[Wiki/Sources/GPT-4]], [[Wiki/Sources/GPT-4o]], [[Wiki/Sources/Llama 3]], [[Wiki/Sources/RoPE]], [[Wiki/Sources/DeepSeek LLM 开源语言模型与长期主义]], [[Wiki/Sources/DeepSeek-V3 技术报告]], [[Wiki/Sources/DeepSeek-V3.2 开源大模型前沿]], [[Wiki/Sources/DeepSeek-R1 强化学习推理]], [[Wiki/Sources/Kimi k1.5 强化学习规模化]], [[Wiki/Sources/Kimi K2 开放 Agent 智能]], [[Wiki/Sources/Kimi K2.5 视觉 Agent 智能]], [[Wiki/Sources/Kimi-VL 技术报告]], [[Wiki/Sources/Qwen3 技术报告]], [[Wiki/Sources/ViT]], [[Wiki/Sources/Swin Transformer]], [[Wiki/Sources/MLP-Mixer]], [[Wiki/Sources/CLAP]], [[Wiki/Sources/SigLIP 2]], [[Wiki/Sources/Emu3.5]], [[Wiki/Sources/Chain-of-Thought]], [[Wiki/Sources/Tree of Thoughts]], [[Wiki/Sources/Graph of Thoughts]], [[Wiki/Sources/Competitive Programming LM]], [[Wiki/Sources/WorldEdit]], [[Wiki/Sources/LLM Agent 综述 2024]], [[Wiki/Sources/Agent AI Survey 2024]], [[Wiki/Sources/Agent Banana]], [[Wiki/Sources/Magic-MM-Embedding]], [[Wiki/Sources/ObjEmbed]], [[Wiki/Sources/RzenEmbed]], [[Wiki/Sources/SAIL-Embedding]], [[Wiki/Sources/OCRBench v2]], [[Wiki/Sources/OmniDocBench]], [[Wiki/Sources/olmOCR]]
  - Topics (7): [[Wiki/Topics/大语言模型基础]], [[Wiki/Topics/LLM Agent 与工具使用]], [[Wiki/Topics/多模态 Embedding 与检索]], [[Wiki/Topics/多模态 Benchmark 与评估]], [[Wiki/Topics/Vision Transformer 演进]], [[Wiki/Topics/推理增强方法]], [[Wiki/Topics/国产大模型演进]]
  - Concepts (13): [[Wiki/Concepts/Scaling Laws]], [[Wiki/Concepts/RLHF]], [[Wiki/Concepts/RoPE 旋转位置编码]], [[Wiki/Concepts/GPT 系列模型]], [[Wiki/Concepts/LLM Agent 架构]], [[Wiki/Concepts/多模态 Embedding 模型]], [[Wiki/Concepts/PDF 文档解析]], [[Wiki/Concepts/Chain-of-Thought 思维链]], [[Wiki/Concepts/多模态对比学习]], [[Wiki/Concepts/原生多模态模型]], [[Wiki/Concepts/MoE 混合专家模型]], [[Wiki/Concepts/推理模型与强化学习]], [[Wiki/Concepts/多模态 Agent]]
  - Entities (7): [[Wiki/Entities/Transformer]], [[Wiki/Entities/Vision Transformer (ViT)]], [[Wiki/Entities/Swin Transformer]], [[Wiki/Entities/Agent Banana]], [[Wiki/Entities/DeepSeek 系列模型]], [[Wiki/Entities/Kimi 系列模型]], [[Wiki/Entities/Qwen3]]
  - Comparisons (1): [[Wiki/Comparisons/多模态 Embedding 模型比较]]
- Updated: [[Wiki/index]], [[Wiki/overview]]
- Notes: 本次是 Wiki 建立以来最大规模的一次摄取（40 篇），通过 4 个并行 Agent 在约 12 分钟内完成。Wiki 从"图像编辑+情感计算"的垂直知识库扩展为覆盖 LLM 基础、国产大模型、推理方法、Vision 基础、Agent、Embedding、Benchmark 的全栈 AI 知识体系。总页面数从 ~50 跃升至 **120+**，全部 69 个 Clippings 已处理完毕。从 2026-04-29 初始化到 2026-04-30 全部完成，历时不到 2 天。

## [2026-04-30] deepen | 5 个核心 Concepts 深度分析

- Enhanced:
  - [[Wiki/Concepts/MoE 混合专家模型]] — 新增深度分析：Dense scaling 撞墙、推理成本压力、MoE 显存弱点、aux-loss-free 的真实价值、Dense+MoE 双线的务实逻辑
  - [[Wiki/Concepts/推理模型与强化学习]] — 新增深度分析：提示工程时代终结的边界、aha moment 的去浪漫化、MCTS 失败的问题特性不匹配、蒸馏 vs 直接 RL 的三重验证、推理能力上限、推理模型的对齐挑战
  - [[Wiki/Concepts/多模态 Agent]] — 新增深度分析：训练派 vs 推理派阵营、Agent 评估瓶颈、多模态是否伪需求、Agent Swarm 过度设计问题、安全对齐空白
  - [[Wiki/Concepts/原生多模态模型]] — 新增深度分析：统一 vs 组装的数据本质、BAGEL MoT 的第三条路、互信息瓶颈被夸大、DiDA 20x 的独立验证需求、5/10 年预测
  - [[Wiki/Concepts/RLHF]] — 新增深度分析：DPO 取代 PPO 的工程原因、RLHF 是安全帽非引擎、推理时代对齐挑战、过度对齐问题、下一个范式
- Cross-references: [[Wiki/Comparisons/国产大模型技术路线比较]], [[Wiki/Comparisons/统一多模态模型架构比较]]

## [2026-04-30] deepen | 4 个深度比较页创建

- Created:
  - [[Wiki/Comparisons/推理增强方法比较]] — CoT→ToT→GoT→推理模型：外部提示到内部训练的范式转移，有效思考量统一框架，ToT/GoT 为何未成主流，CoT 的角色重定位
  - [[Wiki/Comparisons/Vision Transformer 架构比较]] — ViT→Swin→MLP-Mixer：归纳偏置剥离实验视角，层级结构决定通用性，MLP-Mixer 的哲学追问价值，ViT 在 VLM/扩散模型中的新角色
  - [[Wiki/Comparisons/LLM 缩放定律比较]] — Kaplan vs Chinchilla：方法论偏置分析，数据质量被低估，过度训练策略崛起，MoE sparsity scaling law 扩展，缩放定律作为通用 AI 分析工具
  - [[Wiki/Comparisons/GPT 系列代际比较]] — GPT-1→GPT-4o 能力矩阵，规模→方法→模态的驱动力转移规律，GPT-4 技术保密的科学伤害与商业逻辑，开源追赶加速
- Updated: [[Wiki/index]], [[Wiki/log]]
- Cross-references: [[Wiki/Topics/推理增强方法]], [[Wiki/Topics/Vision Transformer 演进]], [[Wiki/Topics/大语言模型基础]], [[Wiki/Concepts/GPT 系列模型]], [[Wiki/Concepts/Scaling Laws]], [[Wiki/Concepts/Chain-of-Thought 思维链]], [[Wiki/Concepts/推理模型与强化学习]]
- Notes: 4 个比较页覆盖 LLM 推理、视觉架构、缩放定律、GPT 演进四个核心方向的深度分析。每个页面包含独立判断和批判性分析——从"有效思考量"统一框架、归纳偏置剥离实验、过度训练 vs Chinchilla 最优、到 GPT-4 保密对科学的伤害——而非仅罗列事实。将推理方法从"越来越好"的表面叙事升级为"外部策略→内部能力"的范式转移理解；将 ViT/Swin/MLP-Mixer 从三个独立工作重读为一场连续实验；将缩放定律从训练配方扩展为 AI 通用分析工具；将 GPT 代际从能力炫耀重构为驱动力转移故事。Wiki 比较页总计 11 个。




## [2026-04-30] deepen | 深度分析与批判性思考全量更新

- Created:
  - Comparisons (5): [[Wiki/Comparisons/国产大模型技术路线比较]], [[Wiki/Comparisons/推理增强方法比较]], [[Wiki/Comparisons/Vision Transformer 架构比较]], [[Wiki/Comparisons/LLM 缩放定律比较]], [[Wiki/Comparisons/GPT 系列代际比较]]
- Enhanced:
  - Concepts (5): [[Wiki/Concepts/MoE 混合专家模型]], [[Wiki/Concepts/推理模型与强化学习]], [[Wiki/Concepts/多模态 Agent]], [[Wiki/Concepts/原生多模态模型]], [[Wiki/Concepts/RLHF]] — 均新增"深度分析"章节
  - Entities (3): [[Wiki/Entities/DeepSeek 系列模型]], [[Wiki/Entities/Kimi 系列模型]], [[Wiki/Entities/Qwen3]] — 均新增"批判性评估"章节
  - Topics (2): [[Wiki/Topics/国产大模型演进]], [[Wiki/Topics/大语言模型基础]] — 均新增"我的思考"章节
- Updated: [[Wiki/index]], [[Wiki/overview]], [[Wiki/Comparisons/README]]
- Notes: 这是 Wiki 建立以来的第三次深度分析迭代（第一次：编辑+情感比较页深化；第二次：图像编辑 Entity 扩展；本次：全量知识体系的批判性分析）。核心变化：Wiki 从"信息聚合"升级为"知识 + 独立洞见"。所有比较页和概念页现在包含我的独立判断（非事实罗列），覆盖 MoE 的经济动因、推理模型的范式转移、Agent 的训练派 vs 推理派分歧、原生多模态的长期预测、RLHF 时代的终结、国产大模型路线的深层逻辑。Wiki 现已从"研究笔记"演变为"可独立产生洞见的知识体系"。


## [2026-04-30] ingest | 扩散模型与 Flow Matching 基础批量摄取（6 篇）

- Sources:
  - [[Clippings/Denoising Diffusion Probabilistic Models.md]]
  - [[Clippings/Flow Matching for Generative Modeling.md]]
  - [[Clippings/FLUX.1 Kontext Flow Matching for In-Context Image Generation and Editing in Latent Space.md]]
  - [[Clippings/Seedream 2.0 A Native Chinese-English Bilingual Image Generation Foundation Model.md]]
  - [[Clippings/Seedream 3.0 Technical Report.md]]
  - [[Clippings/Seedream 4.0 Toward Next-generation Multimodal Image Generation.md]]
- Created:
  - Sources (6): [[Wiki/Sources/DDPM 扩散模型奠基论文]], [[Wiki/Sources/Flow Matching 生成建模]], [[Wiki/Sources/FLUX.1 Kontext 上下文编辑]], [[Wiki/Sources/Seedream 2.0 中英双语图像生成]], [[Wiki/Sources/Seedream 3.0 技术报告]], [[Wiki/Sources/Seedream 4.0 多模态图像生成]]
  - Concepts (2): [[Wiki/Concepts/扩散模型原理]], [[Wiki/Concepts/Flow Matching]]
  - Entities (1): [[Wiki/Entities/Seedream 系列模型]]
  - Topics (1): [[Wiki/Topics/扩散模型与 Flow Matching 基础]]
- Updated:
  - [[Wiki/Topics/扩散模型图像编辑与生成]] — 新增「阶段零：生成模型底层引擎」节，补充 FLUX.1 Kontext 到指令编辑分支，更新 frontmatter sources
  - [[Wiki/index]]
- Notes: 一次摄入了 6 篇 Clippings，覆盖扩散模型从理论到工业落地的完整脉络——DDPM 奠基 -> Flow Matching 统一框架 -> FLUX.1 Kontext 编辑应用 -> Seedream 2.0/3.0/4.0 工业级演进。核心洞察：(1) DDPM 的 epsilon-预测等价于 score matching，而 FM 的 v-预测等价于向量场回归，后者更稳定；(2) Flow Matching 的 OT 路径本质是 Wasserstein-2 最优传输，形成直线轨迹，是后续 rectified flow 的理论基础；(3) FLUX.1 Kontext 和 Seedream 4.0 都走向了"统一生成与编辑"——序列拼接 vs 多模态联合后训练，两条路线殊途同归；(4) Seedream 从 2.0 到 4.0 完成从单任务 T2I 到多模态统一的跃迁，架构重设计带来 10 倍加速。新建的「扩散模型与 Flow Matching 基础」主题与已有的「扩散模型图像编辑与生成」形成互补——前者聚焦生成引擎，后者聚焦编辑应用。

## [2026-04-30] ingest | 推荐系统批量摄取（8 篇）

- Sources:
  - [[Clippings/OneRec Technical Report.md]]
  - [[Clippings/OneRec Unifying Retrieve and Rank with Generative Recommender and Preference Alignment.md]]
  - [[Clippings/OneRec-Think In-Text Reasoning for Generative Recommendation.md]]
  - [[Clippings/OneRec-V2 Technical Report.md]]
  - [[Clippings/OpenOneRec Technical Report An Open Foundation Model and Benchmark to Accelerate Generative Recommendation.md]]
  - [[Clippings/OneTrans Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender.md]]
  - [[Clippings/HyFormer Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction.md]]
  - [[Clippings/InterFormer Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction.md]]
- Created:
  - Sources (8): [[Wiki/Sources/OneRec Technical Report 2025]], [[Wiki/Sources/OneRec Unifying Retrieve and Rank 2025]], [[Wiki/Sources/OneRec-Think 2025]], [[Wiki/Sources/OneRec-V2 Technical Report 2025]], [[Wiki/Sources/OpenOneRec Technical Report 2025]], [[Wiki/Sources/OneTrans 2025]], [[Wiki/Sources/HyFormer 2025]], [[Wiki/Sources/InterFormer 2024]]
  - Topics (1): [[Wiki/Topics/生成式推荐系统]]
  - Concepts (2): [[Wiki/Concepts/生成式推荐]], [[Wiki/Concepts/CTR 预估]]
  - Entities (1): [[Wiki/Entities/OneRec 系列模型]]
- Updated: [[Wiki/index]]
- Notes: 一次摄入了 8 篇推荐系统相关 Clippings，覆盖推荐领域两大方向：(1) 端到端生成式推荐——OneRec 系列（快手）从 V0 到 V2 再到 Think 和开源版 OpenOneRec 的完整演进，展示了从 Encoder-Decoder 到 Lazy Decoder-Only 的架构演进、从 DPO 到 GRPO 的 RL 对齐技术迁移、以及 CoT 推理和 Scaling Laws 在推荐中的验证；(2) CTR 预估统一架构——OneTrans（字节）的 Unified Causal Transformer、HyFormer（字节）的 Query Decoding+Boosting 交替设计、InterFormer（Meta）的双向异构交互框架，展示了从分离式 pipeline 到统一 Transformer 的演进路径。这是 Wiki 首次进入推荐系统领域，与已有的 LLM、Transformer、Scaling Laws、RLHF、MoE、CoT 等 AI 基础知识建立了丰富的连接。8 篇来源中 5 篇来自快手（OneRec 系列），2 篇来自字节跳动（OneTrans/HyFormer），1 篇来自 Meta（InterFormer），全部为工业级部署论文，展现了推荐系统领域最前沿的工业实践。

## [2026-04-30] ingest | 核心基础论文 + 目标检测 + DeepSeek 补充 + Qwen + MiniMax + PyTorch + gpt-oss 批量摄取（14 篇）

- Sources:
  - 核心基础: [[Clippings/LoRA Low-Rank Adaptation of Large Language Models.md]], [[Clippings/Masked Autoencoders Are Scalable Vision Learners.md]], [[Clippings/Outrageously Large Neural Networks The Sparsely-Gated Mixture-of-Experts Layer.md]], [[Clippings/Switch Transformers Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.md]], [[Clippings/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.md]], [[Clippings/Native Sparse Attention Hardware-Aligned and Natively Trainable Sparse Attention.md]]
  - 目标检测: [[Clippings/DINO DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection.md]], [[Clippings/You Only Look Once Unified, Real-Time Object Detection.md]]
  - DeepSeek 补充: [[Clippings/DeepSeek-V2 A Strong, Economical, and Efficient Mixture-of-Experts Language Model.md]], [[Clippings/DeepSeek-Coder-V2 Breaking the Barrier of Closed-Source Models in Code Intelligence.md]]
  - Qwen 补充: [[Clippings/Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition.md]]
  - 其他: [[Clippings/MiniMax-M1 Scaling Test-Time Compute Efficiently with Lightning Attention.md]], [[Clippings/PyTorch An Imperative Style, High-Performance Deep Learning Library.md]], [[Clippings/gpt-oss-120b & gpt-oss-20b Model Card.md]]
- Created:
  - Sources (14): [[Wiki/Sources/LoRA 低秩适配]], [[Wiki/Sources/MAE 掩码自编码器]], [[Wiki/Sources/MoE 稀疏门控混合专家层]], [[Wiki/Sources/Switch Transformer]], [[Wiki/Sources/RAG 检索增强生成]], [[Wiki/Sources/NSA 原生稀疏注意力]], [[Wiki/Sources/DINO DETR 目标检测]], [[Wiki/Sources/YOLO 目标检测]], [[Wiki/Sources/DeepSeek-V2 技术报告]], [[Wiki/Sources/DeepSeek-Coder-V2 代码智能]], [[Wiki/Sources/Qwen-Image-Layered 分层编辑]], [[Wiki/Sources/MiniMax-M1 测试时计算扩展]], [[Wiki/Sources/PyTorch 深度学习框架]], [[Wiki/Sources/gpt-oss 开源模型]]
  - Concepts (5): [[Wiki/Concepts/LoRA 低秩适配]], [[Wiki/Concepts/MAE 掩码自编码器]], [[Wiki/Concepts/RAG 检索增强生成]], [[Wiki/Concepts/NSA 原生稀疏注意力]], [[Wiki/Concepts/测试时计算扩展]]
  - Topics (1): [[Wiki/Topics/目标检测基础]]
  - Comparisons (0): (本次摄入以概念和实体为主，比较维度将在后续深化时创建)
- Updated:
  - [[Wiki/Concepts/MoE 混合专家模型]] — 新增「历史演进」章节，从 Google Brain 2017 原始 MoE (Sparsely-Gated MoE) → Switch Transformer (2021) → 现代 MoE (2024-2025) 的三阶段演进，将 MoE 知识从"只有国产大模型视角"扩展为完整历史叙事
  - [[Wiki/Entities/DeepSeek 系列模型]] — **大幅扩展 V2 条目**（kv cache 减少 93.3%、训练成本降低 42.5% 等关键指标），新增 DeepSeek-Coder-V2 完整条目（338 语言、GPT4-Turbo 级、开源首次对标闭源），更新技术贡献表和设计哲学（新增第 5 条"代码特化分支"）
  - [[Wiki/Topics/扩散模型图像编辑与生成]] — 新增「阶段五：图层分解编辑」章节和「新范式的意义」分析，将 Qwen-Image-Layered 定位为从"像素编辑"到"图层编辑"的根本性范式转变
  - [[Wiki/index]] — 添加 14 个新 Sources、5 个新 Concepts、1 个新 Topic 的索引条目，更新 MoE 概念描述和扩散模型主题来源数
- Notes: 这是 Wiki 到目前为止最大的一次批量摄入（14 篇 Clippings，覆盖 7 个不同方向）。最重要的 5 个知识增量：
  1. **MoE 历史补全**：原始 MoE (2017, Google Brain) 和 Switch Transformer (2021) 的加入，使得 MoE 知识从"当前国产模型快照"升级为"完整的 8 年历史叙事"。这解释了为什么现代 MoE 方案（aux-loss-free bias、sparsity scaling law）是对早期方案（auxiliary loss、top-1 路由）的有意识改进，而非全新发明。
  2. **DeepSeek-V2 是 MLA 首次引入的关键里程碑**：之前 Wiki 对 MLA 的理解来自 V3 引用，现在有了完整来源。V2 的 93.3% KV cache 减少和 5.76x 吞吐量提升是理解 DeepSeek 架构效率优势的核心数据。V2→Coder-V2→V3→R1→V3.2 的完整演进链现已建立。
  3. **图层分解 = 编辑的新范式**：Qwen-Image-Layered 提出的问题比答案更深刻——编辑一致性的瓶颈不是编辑算法，而是图像的表示本身。这是与 Workshop/Wiki 中 29 篇编辑文献对比后产生的核心洞察。
  4. **测试时计算扩展**：MiniMax-M1 的 Lightning Attention + CISPO 提供了一条不同于"更复杂的注意力 = 更好"的路线——在长推理场景中，线性注意力 + 混合设计的 FLOPs 仅为 R1 的 25%。
  5. **知识网络完整性**：LoRA（微调 → 扩散模型生态）、MAE（ViT 自监督预训练 → CV 前沿）、RAG（检索增强 → Agent 工具使用）、NSA（稀疏注意力 → DeepSeek 技术栈）四个概念页补充了 LLM/CV 知识网络的关键节点。PyTorch 设计哲学和 gpt-oss 开源策略从工具和产业视角丰富了 Wiki。

## [2026-04-30] lint | Wiki 全量 Clippings 处理完成确认

- Checked: 所有 Clippings 文件处理状态
- Result: 全部 69 个 Clippings 已处理完毕（14 个本次 + 55 个之前批次 = 69），Wiki/Sources/ 共 69 个来源摘要页
- Wiki 当前规模:
  - Sources: 69 个
  - Concepts: 19 个
  - Entities: 24 个
  - Topics: 11 个
  - Comparisons: 11 个
  - Questions: 2 个
  - 总页面: 140+
- Open issues: 无。所有 Clippings 已完整摄入。
- Suggested: 下一阶段可专注于 (1) 跨主题比较页创建（如 LoRA vs 其他微调方法、YOLO vs DETR 全版本对比）；(2) 已有页面的批判性分析深化

## [2026-04-30] deepen | Wiki 条目深度沉淀

- Type: 全目录补全
- Created Concepts (6):
  - [[Wiki/Concepts/DiT 扩散 Transformer]] — UNet→DiT 架构转换，adaLN-Zero 条件注入，归纳偏置"资产到负债"
  - [[Wiki/Concepts/GRPO 分组相对策略优化]] — 去 Critic 的 RL 算法，组内相对比较，降低推理训练门槛
  - [[Wiki/Concepts/DPO 直接偏好优化]] — 绕过 RM 的直接对齐，对齐成本最优解，DPO/GRPO 分工
  - [[Wiki/Concepts/MLA 多头潜在注意力]] — 低秩 KV cache 压缩 93.3%，让 MoE 推理经济可行
  - [[Wiki/Concepts/CLIP 对比语言图像预训练]] — 双塔多模态基石，语言作为通用视觉监督的方法论革命
  - [[Wiki/Concepts/知识蒸馏]] — 压缩→能力迁移的角色转换，蒸馏 vs RL 的分层策略
- Created Entities (2):
  - [[Wiki/Entities/FLUX]] — BFL Flow Matching+DiT 模型，SD 原团队"重做版"
  - [[Wiki/Entities/GPT-4o]] — OpenAI 首个端到端全模态 omni 模型，232ms 语音延迟
- Created Questions (3):
  - [[Wiki/Questions/为什么 MCTS 在 LLM 推理中失败了]] — 问题结构不匹配、成本、隐式搜索 5 原因
  - [[Wiki/Questions/知识蒸馏 vs RL 哪种方式更能有效获得推理能力]] — 反直觉发现分析 + 规模分层策略
  - [[Wiki/Questions/多模态模型的最终形态是原生统一还是模块化组装]] — 短期/中期/长期预测
- Created Comparisons (2):
  - [[Wiki/Comparisons/扩散模型架构比较 UNet vs DiT]] — 全维度对比 + DiT 规模优势分析
  - [[Wiki/Comparisons/推理模型训练方法比较 DeepSeek-R1 vs Kimi k1.5 vs Qwen3]] — 三家方法论全流程对比
- Updated: [[Wiki/Contradictions.md]] — 初始填充 6 条活跃矛盾（Kaplan/Chinchilla、原生/组装、蒸馏/RL、MCTS 缺席、离散/连续情感、Flow/Diffusion）
- Updated: [[Wiki/index.md]], [[Wiki/overview.md]]
- Notes: 本次重点补全"被频繁引用但缺少独立页面"的核心概念（DiT/GRPO/DPO/MLA/CLIP/知识蒸馏），以及从已沉淀知识中自然涌现的高价值问题和矛盾。Contradictions.md 从空壳变为有 6 条活跃矛盾的实用页面。
- Wiki 当前规模:
  - Sources: 97 个
  - Concepts: 33 个（+6）
  - Entities: 29 个（+2）
  - Topics: 14 个
  - Comparisons: 13 个（+2）
  - Questions: 5 个（+3）
  - Contradictions: 1 个（首次填充，6 条矛盾）
  - 总页面: 160+


## [2026-05-01] concept | 补充 4 个核心概念页

- Type: 概念补充
- Created Concepts (4):
  - [[Wiki/Concepts/VQ-VAE 离散 Token 化]] — VQ-VAE 原理（encoder→codebook→decoder）、RQ-Kmeans 变体、codebook collapse 问题、连续 vs 离散表示的架构选择。应用覆盖 Show-o2/Lumina-DiMOO/Emu3.5/OneRec/BAGEL。深度分析：离散化是必须的还是历史包袱、RQ-Kmeans 的工业务实选择、连续-离散混合表示的长期趋势。
  - [[Wiki/Concepts/Agent 图像编辑]] — 推理-规划-执行-验证闭环的编辑新范式，GoT→Agent Banana→Mind-Brush→VisionCreator 全景。与端到端编辑的本质区别对比表、设计空间分析（推理深度/知识来源/训练范式）、路由式架构预测。
  - [[Wiki/Concepts/审美评估与推理]] — 从 LAION predictor 到 Aes-R1 RAPO 的审美评分与推理，编辑的「北星」标尺。深度分析：评分器的文化偏见、准和懂的分离、审美推理对编辑价值的低估。
  - [[Wiki/Concepts/编辑数据合成方法]] — 六代数据合成范式演进（纯合成→真实+LLM→自适应→层级分类→情感特化→多模态多参考）。深度分析：好指令→好编辑的隐性假设、in-context 示例偏见放大、四轮筛选是无奈之举、交互式数据合成的未来。
- Updated: [[Wiki/index]], [[Wiki/overview]]
- Notes: 这 4 个概念页填补了 Wiki 中「被多个来源和主题反复引用但缺少独立页面」的知识空白。VQ-VAE 是连接连续视觉和离散 Token 的桥梁概念（至少 5 个来源共同依赖）；Agent 图像编辑与通用 LLM Agent 有本质区别（视觉感知循环、编辑轨迹评估、VRL）；审美评估跨生成/编辑/评估三个方向；编辑数据合成是该领域的「元方法论」。
- Wiki 当前规模: Concepts: 37 (+4)；总页面: 165+


## [2026-05-03] ingest | AHE Agentic Harness Engineering

- Source: [[Clippings/Agentic Harness Engineering Observability-Driven Automatic Evolution of Coding-Agent Harnesses.md]]
- Created:
  - Sources: [[Wiki/Sources/AHE Agentic Harness Engineering]]
  - Concepts: [[Wiki/Concepts/Agentic Harness Engineering (AHE)]], [[Wiki/Concepts/Harness 编码 Agent 线束]]
- Updated:
  - [[Wiki/Concepts/LLM Agent 架构]] — 新增「Harness 工程：Agent 架构的工程实现层」章节，将 Agent 四大模块映射到 7 种 harness 组件
  - [[Wiki/Topics/LLM Agent 与工具使用]] — 新增 AHE 来源、harness 相关概念和开放问题
  - [[Wiki/index]]
- Notes: AHE 是 2026 年来自复旦/北大/奇绩智峰的重要工作，解决了编码 Agent 中 harness 自动演化这个被低估的问题。核心洞察：**(1) harness 优化的瓶颈是可观测性而非 agent 能力**——三个支柱（组件/经验/决策可观测性）将每次编辑变成可证伪的契约；(2) **增益存在于 tools/middleware/long-term memory 中，而非 system prompt**——事实性结构可迁移，散文级策略不可；(3) **回归盲区是 self-evolution 的根本性不对称**——agent 能预测什么会变好但几乎无法预测什么会变差。AHE 首次将 harness 工程从"手工工艺"升级为"可自动化学习问题"，与 Wiki 已有的 Agent 架构、推理模型、国产大模型等知识形成互补——提供了"模型外优化"的新维度。
- Wiki 当前规模: Sources: 98 (+1)；Concepts: 39 (+2)；总页面: 168+

## [2026-05-06] ingest | Self-Evolving Agents 综述

- Source: [[Clippings/A Systematic Survey of Self-Evolving Agents.md]]
- Created:
  - Sources: [[Wiki/Sources/Self-Evolving Agents 综述]]
  - Concepts: [[Wiki/Concepts/Self-Evolving Agents 自演化智能体]]
- Updated:
  - [[Wiki/Topics/LLM Agent 与工具使用]] — 新增 Self-Evolving Agents 综述来源和自演化智能体概念
  - [[Wiki/index]]
- Notes: 这篇来自厦门大学（2026-02）的综述系统性地梳理了自演化智能体领域，提出三大范式分类体系：(1) **模型中心自演化**——推理时（并行采样/顺序自纠正/结构化推理）和训练时（合成驱动离线蒸馏/探索驱动在线 RL）两条路线；(2) **环境中心自演化**——静态知识演化（Agentic RAG/深度研究）、动态经验演化（离线编译/在线适配/终身演化）、模块化架构演化（记忆/工具/交互协议）、智能体拓扑演化（多智能体结构搜索）；(3) **模型-环境协同演化**——多智能体策略协同演化和环境训练。核心洞察：**环境不应是静态背景，而应是与 Agent 共同演化的可优化伙伴**。当前挑战包括静态非自适应环境、过度依赖可验证任务、模拟环境真实性不足、持续依赖人类初始化和模型坍缩。应用覆盖自动化科学发现（GNoME 220万新材料、A-Lab 71%合成成功率）、自主软件工程（SWE-agent、Claude Code、Devin）和开放世界模拟（Voyager、Generative Agents）。与已有 Agent 架构、推理模型、RL 方法、RAG 等知识形成丰富交叉。
- Wiki 当前规模: Sources: 99 (+1)；Concepts: 40 (+1)；总页面: 170+

---
type: log
status: active
created: '2026-04-29'
updated: '2026-05-07'
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

## [2026-05-07] ingest | Emu3 原生多模态模型

- Source: [[Clippings/Emu3: Next-Token Prediction is All You Need]]
- Created: [[Wiki/Sources/Emu3 原生多模态模型]]
- Updated: [[Wiki/index]]
- Notes: Emu3 是 BAAI 的纯 next-token prediction 原生多模态模型。通过统一视觉 tokenizer（SBER-MoVQGAN, 32K codebook, 4x8x8 压缩）将图像/视频/文本离散化为 token，单一 Transformer decoder 从零训练。图像生成超越 SDXL，视觉理解与 LLaVA-1.6 持平，视频生成 VBench 80.96 接近商业模型。DPO 对齐自回归视觉生成是重要技术贡献。

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

## [2026-05-07] ingest | GPT-2

- Source: [[Clippings/Language Models are Unsupervised Multitask Learners]]
- Created:
  - Sources: [[Wiki/Sources/GPT-2]]
- Updated:
  - [[Wiki/index]] — 添加 GPT-2 来源条目
- Notes: OpenAI 2019 年技术报告，GPT 系列第二代。核心贡献：(1) 首次证明纯语言模型可以在零样本条件下执行多种下游任务（阅读理解 55 F1 on CoQA、LAMBADA 困惑度降至 8.6、7/8 语言建模 SOTA）；(2) WebText 数据集构建方法（Reddit 3+ karma 链接）；(3) 字节级 BPE 输入表示。关键局限：单向性限制、翻译/摘要性能仍远低于有监督系统。该论文标志着从"预训练+微调"向"预训练即任务执行"的关键转变，直接催生了 GPT-3 的 In-Context Learning。与已有 [[Wiki/Sources/BERT]] 和 [[Wiki/Sources/GPT-3]] 形成完整的技术演进链。
- Wiki 当前规模: Sources: 100 (+1)；总页面: 171+

## [2026-05-07] ingest | Qwen 第一代技术报告

- Source: [[Clippings/Qwen Technical Report]]
- Created:
  - Sources: [[Wiki/Sources/Qwen 技术报告]]
- Updated:
  - [[Wiki/index]] — 添加 Qwen 第一代技术报告来源条目
- Notes: 阿里巴巴 Qwen 团队 2023 年发表的第一代 Qwen 技术报告（arXiv:2309.16609）。覆盖 1.8B/7B/14B 三个规模，预训练数据达 3T tokens，采用基于 LLaMA 的改进架构（解耦嵌入、RoPE FP32、QKV bias、SwiGLU）。对齐方面使用 SFT+RLHF，RLHF 版本在人类评估中显著优于 SFT 版本。专用模型 CODE-QWEN（HumanEval 66.4%）和 MATH-QWEN-CHAT（GSM8K 69.8%）在同规模开源模型中领先。工具使用能力突出：ReAct 工具调用假阳性率仅 2.4%，代码解释器可执行率 81.7%。与已有 Qwen3 来源形成 Qwen 系列完整演进链。与 LLaMA/LLaMA 2、DeepSeek、Kimi 等已入库模型形成交叉参考。
- Wiki 当前规模: Sources: 101 (+1)；总页面: 172+

## [2026-05-07] ingest | AHE Agentic Harness Engineering (re-process)

- Source: [[Clippings/Agentic Harness Engineering Observability-Driven Automatic Evolution of Coding-Agent Harnesses]]
- Action: 重新处理 MinerU 提取 + pdf2zh 翻译，替换原 clipping 内容
- Updated:
  - [[Clippings/Agentic Harness Engineering Observability-Driven Automatic Evolution of Coding-Agent Harnesses]] — 重新提取 markdown + images
  - 翻译 PDF 更新至 `~/Desktop/paper/03_Agent与推理/`
- Notes: 用户反馈之前处理效果不佳，本次使用最新 MinerU 重新提取。LaTeX 公式保留原始格式。

## [2026-05-07] ingest | Kimi Linear 高效注意力架构

- Source: [[Clippings/Kimi Linear: An Expressive, Efficient Attention Architecture]]
- Created:
  - Sources: [[Wiki/Sources/Kimi Linear 高效注意力架构]]
- Updated:
  - [[Wiki/index]] — 添加 Kimi Linear 来源条目
- Notes: Moonshot AI 的 Kimi Linear 技术报告（arXiv:2510.26692）。核心贡献：Kimi Delta Attention（KDA），在 Gated DeltaNet 基础上引入 channel-wise 对角门控（每个特征维度独立衰减率），通过约束 DPLR 结构（a_t = b_t = k_t）实现比通用 DPLR 约 2 倍的 kernel 加速。3:1 KDA/MLA 混合比例在性能和效率间最优。3B 激活/48B 总参数 MoE 模型，1.4T tokens 预训练下在短上下文（MMLU-Pro 51.0）、长上下文（RULER 84.3@128k）和 RL scaling 中均超越全注意力 MLA 基线。KV cache 减少最高 75%，1M 上下文解码吞吐量提升最高 6.3x。MLA 层采用 NoPE，位置信息完全由 KDA 层内化。与 NSA（稀疏注意力）、MiniMax-M1（Lightning Attention）、Mamba2 等形成注意力效率优化的不同技术路线对比。开源 KDA kernel + vLLM 实现。
- Wiki 当前规模: Sources: 102 (+1)；总页面: 173+

## [2026-05-07] ingest | Thinking with Visual Primitives

- Source: [[Clippings/Thinking with Visual Primitives]]
- Created: [[Wiki/Sources/Thinking with Visual Primitives]]
- Updated: [[Wiki/index]]
- Notes: DeepSeek-AI + 北大 + 清华 2025 年论文。提出"视觉原语思维"框架，将 bounding box 和 point 提升为最小思维单元，交错嵌入推理链，解决 MLLM 的 Reference Gap。基于 DeepSeek-V4-Flash（284B MoE），通过 3x3 空间压缩 + CSA 实现 7,056x 视觉 token 压缩比。大规模数据构建（4,000 万+ box 样本）+ 四维度冷启动数据（计数/空间推理/迷宫导航/路径追踪）。后训练采用专家分化 + 统一 RFT + On-Policy Distillation 范式。在 CountQA、Pixmo-Count、SpatialMQA 等基准上匹配或超越 GPT-5.4、Claude-Sonnet-4.6、Gemini-3-Flash，拓扑推理（迷宫/路径追踪）显著领先所有对比模型。与知识库中 GoT、Mind-Brush、VisionCreator 等推理增强方法形成互补视角。

## [2026-05-07] ingest | MiniMax-01 Lightning Attention

- Source: [[Clippings/MiniMax-01: Scaling Foundation Models with Lightning Attention]]
- Created:
  - Sources: [[Wiki/Sources/MiniMax-01 Lightning Attention]]
- Updated:
  - [[Wiki/index]] — 添加 MiniMax-01 Lightning Attention 来源条目
- Notes: MiniMax 团队 2025 年发表的技术报告（arXiv:2501.08313）。核心贡献：首次将线性注意力大规模落地到 456B MoE 模型。采用 7:1 lightning/softmax 混合架构，每 7 层 lightning attention（TransNormer 的 I/O 感知实现）后接 1 层 softmax attention，兼具线性复杂度效率和检索能力。训练上下文 100 万 token，推理外推至 400 万 token，是同期主流模型的 20-32 倍。提出 ETP/EDP 并行策略、Varlen Ring Attention、LASP+ 等全新分布式训练推理框架，端到端 MFU >75%（H20 GPU）。在 MMLU（88.5）、MATH（77.4）、RULER（0.910@1M）等基准上达到 GPT-4o/Claude-3.5-Sonnet 水平，长上下文场景显著领先。纯线性注意力检索能力不足是关键局限，混合架构是当前折中方案。与 Kimi Linear（KDA 路线）、NSA（稀疏注意力路线）、Mamba2（SSM 路线）形成注意力效率优化的多元技术路线对比。

## [2026-05-07] ingest | VLM2Vec-V2

- Source: [[Clippings/VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents]]
- Created:
  - Sources: [[Wiki/Sources/VLM2Vec-V2]]
- Updated:
  - [[Wiki/index]] — 添加 VLM2Vec-V2 来源条目
- Notes: Salesforce Research 2025 年工作，统一多模态嵌入框架扩展到视频和视觉文档。两个核心贡献：(1) MMEB-V2 benchmark（78 个任务，新增 5 类 meta-task：视频检索/片段检索/视频分类/视频 QA/视觉文档检索）；(2) VLM2Vec-V2 模型（基于 Qwen2-VL 2B，指令条件对比学习，交错子 batch 采样策略，LoRA rank=16）。关键结果：2B 模型总体 58.0 超过 GME-7B（57.8），图像任务 64.9 接近 VLM2Vec-7B（65.5），文档检索 65.4 超越所有 VLM2Vec 变体但仍落后 ColPali（71.0）。与已有的 [[Wiki/Concepts/多模态 Embedding 模型]]、[[Wiki/Topics/多模态 Embedding 与检索]]、[[Wiki/Comparisons/多模态 Embedding 模型比较]] 形成互补——VLM2Vec-V2 是该领域首个统一图像+视频+视觉文档嵌入的工作。

## [2026-05-07] ingest | Qwen2.5-VL 技术报告

- Source: [[Clippings/Qwen2.5-VL Technical Report]]
- Created:
  - Sources: [[Wiki/Sources/Qwen2.5-VL 技术报告]]
- Updated:
  - [[Wiki/index]] — 添加 Qwen2.5-VL 来源条目
- Notes: 阿里巴巴 Qwen 团队 2025 年发表的 Qwen2.5-VL 技术报告（arXiv:2502.13923）。核心贡献：(1) 重新设计 ViT 架构——窗口注意力（最大 112x112）将计算复杂度从二次降为线性，仅 4 层全自注意力，采用 2D-RoPE + RMSNorm + SwiGLU；(2) 原生动态分辨率 + 动态 FPS 采样——空间和时间维度均实现动态处理，直接使用绝对坐标而非归一化；(3) MRoPE 与绝对时间对齐——通过时间 ID 间隔学习时间节奏，无需额外计算开销；(4) 预训练数据从 1.2T 扩展到 4.1T tokens，涵盖图像描述、OCR、定位、文档解析、视频和 Agent 数据。模型提供 3B/7B/72B 三种尺寸。72B 在多数基准上匹配或超越 GPT-4o 和 Claude 3.5 Sonnet，文档理解（DocVQA 96.4、InfoVQA 87.3、OCRBench 885）和视频理解（LVBench 47.3 vs GPT-4o 的 30.8）表现尤为突出。Agent 能力方面，在 ScreenSpot Pro（43.6%）和 Android Control（93.7% Low EM）上大幅超越竞品。与已有 Qwen 技术报告和 Qwen3 形成完整 Qwen 系列演进链，与 Kimi-VL、InternVL2.5 等 VLM 形成技术路线对比。
- Wiki 当前规模: Sources: 103 (+1)；总页面: 174+

## [2026-05-07] ingest | Normalizing Flows 归一化流

- Source: [[Clippings/Normalizing Flows: An Introduction and Review of Current Methods]]
- Created:
  - Sources: [[Wiki/Sources/Normalizing Flows 归一化流]]
- Updated:
  - [[Wiki/index]] — 添加 Normalizing Flows 来源条目和概念条目
- Notes: Kobyzev et al. 2019 年综述，IEEE TPAMI。系统梳理归一化流的理论基础（变量替换公式、pushforward 测度）和方法分类：耦合流（NICE→RealNVP→Glow→Flow++）、自回归流（MAF/IAF→NAF→SOS）、残差/连续流（iResNet→Residual Flow→FFJORD/Neural ODE）。涵盖耦合函数从仿射到样条的演进，通用性理论证明，以及在表格和图像数据集上的实验对比。与扩散模型、Flow Matching、VAE 的关系在此摘要中标注。该综述是理解 Flow Matching 前身连续时间流（FFJORD/Neural ODE）的重要基础文献。

## [2026-05-07] ingest | Qwen-Image 技术报告

- Source: [[Clippings/Qwen-Image Technical Report]]
- Created:
  - Sources: [[Wiki/Sources/Qwen-Image 技术报告]]
- Updated:
  - [[Wiki/index]] — 添加 Qwen-Image 技术报告来源条目
- Notes: 阿里 Qwen 团队 2025 年图像生成基础模型技术报告（arXiv:2508.02324）。核心架构：MMDiT（20B）+ 冻结 Qwen2.5-VL（7B）条件编码器 + 改进 VAE（单编码器双解码器，兼容图像视频）。关键技术突破：(1) MSRoPE 位置编码——将文本视为沿图像对角线拼接的 2D 张量，解决 Seedream Scaling RoPE 中文本/图像位置编码同构问题；(2) 七阶段数据管线 + 渐进式课程学习（256p→640p→1328p，无文字→有文字，粗数据→精筛数据）；(3) 双编码机制——Qwen2.5-VL 语义特征 + VAE 重建特征同时输入 MMDiT，编辑时兼顾语义一致性和视觉保真度；(4) 三种文字合成策略（Pure/Compositional/Complex Rendering）覆盖低频字符和混合语言。评估结果：DPG 88.32 超越 Seedream 3.0，GenEval RL 后 0.91（唯一突破 0.9），ChineseWord Level-1 准确率 97.3% 大幅领先 GPT Image 1（36.1%）和 Seedream 3.0（33.1%），GEdit 中英文编辑均排名第一，AI Arena Elo 排名第三（领先 FLUX.1 Kontext Pro 和 GPT Image 1 High 30+ 分）。该报告是 Qwen 系列首个视觉生成工作，与 Qwen2.5-VL 的理解能力形成"理解-生成统一"的两个支柱。与 Seedream（同为 MMDiT，MSRoPE 解决位置编码问题）、FLUX（借鉴 VAE 嵌入编辑一致性思路）、GPT Image 1（中文文字渲染大幅领先）形成明确的技术对标。
- Wiki 当前规模: Sources: 102 (+1)；总页面: 173+

## [2026-05-07] ingest | Unify-Agent

- Source: [[Clippings/Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis]]
- Created:
  - Sources: [[Wiki/Sources/Unify-Agent]]
- Updated:
  - [[Wiki/index]] — 添加 Unify-Agent 来源条目
- Notes: 首个端到端统一多模态智能体（UCLA + 腾讯混元 + CUHK + HKUST），将图像生成从闭卷映射重新定义为推理时主动决策过程。四阶段流水线 Think-Research-Recaption-Generate，基于 Bagel-14B SFT，143K 高质量智能体轨迹。关键发现：统一模型中 VAE（低层感知先验）与 ViT（高层语义 token）协同使生成能力反哺理解能力。在 FactIP/WiSE/KiTTEN/T2I-FactBench 四个基准上全面领先开源统一模型。引入 FactIP 基准（2,462 prompt，12 类长尾概念）。局限：单次流水线无迭代、基座模型长上下文受限、与最强闭源仍有差距。与 Mind-Brush、VisionCreator、Agent Banana 等智能体图像生成方法形成互补视角。

## [2026-05-07] ingest | EmoSet 视觉情感数据集

- Source: [[Clippings/EmoSet: A Large-scale Visual Emotion Dataset with Rich Attributes]]
- Created:
  - Sources: [[Wiki/Sources/EmoSet 视觉情感数据集]]
- Updated:
  - [[Wiki/index]] — 添加 EmoSet 来源条目（插入 EmoArt 之后、EmotiCrafter 之前）
- Notes: 深圳大学 Yang et al. (2023) 的大规模视觉情感数据集。核心贡献：(1) 118,102 张人工标注 + 330 万弱标注图像，是 FI 数据集的 5 倍；(2) 基于心理学研究设计的六类可描述情感属性（亮度、色彩丰富度、场景类型、物体类别、面部表情、人体动作），覆盖低/中/高三个视觉层次；(3) Mikels 八类情感模型，类别均衡性显著优于 FI 等数据集；(4) 属性模块在无预训练时平均提升 8.05% 准确率。跨数据集泛化实验表明 EmoSet 训练的模型向 FI 迁移时性能损失更小（5.36% vs 26.98%）。局限：单一情感假设、离散情感模型、自动标注噪声。与已有 EmoEdit（情感编辑）、EmoArt（艺术情感）、EmotiCrafter（V-A 连续生成）等情感计算来源形成互补——EmoSet 提供了属性-情感关联的先验知识，可为情感编辑和生成任务提供数据基础。

## [2026-05-07] ingest | Seedance 2.0 视频生成

- Source: [[Clippings/Seedance 2.0: Advancing Video Generation for World Complexity]]
- Created: [[Wiki/Sources/Seedance 2.0 视频生成]]
- Updated: [[Wiki/index]]
- Notes: ByteDance Seed 团队的统一多模态音视频联合生成模型，支持文本/图像/音频/视频四种输入模态，在 SeedVideoBench 2.0 和 Arena.AI 排行榜上 T2V/I2V/R2V 三项任务全面领先竞品（Kling 3.0、Sora 2 Pro、Veo 3.1 等）。论文侧重评估而非架构细节，技术报告未公开模型架构具体设计。

## [2026-05-07] ingest | PaperBanana 学术插图自动生成

- Source: [[Clippings/PaperBanana: Automating Academic Illustration for AI Scientists]]
- Created:
  - Sources: [[Wiki/Sources/PaperBanana]]
- Updated:
  - [[Wiki/index]] — 添加 PaperBanana 来源条目
- Notes: 北京大学 + Google Cloud AI Research 2026 年论文（arXiv:2601.23265）。提出 PaperBanana，一个参考驱动的多智能体学术插图自动生成框架，编排 Retriever、Planner、Stylist、Visualizer、Critic 五个专用智能体。核心创新：(1) Retriever 基于 VLM 的生成式检索，优先匹配研究领域和图表类型；(2) Stylist 遍历参考集自动综合审美指南（配色/形状/布局/排版），无需人工定义；(3) Visualizer-Critic 闭环迭代 T=3 轮，在忠实度和美观度间取得平衡。引入 PaperBananaBench（292 测试用例，NeurIPS 2025 论文策展）。方法论图整体提升 +17.0%，盲评人工胜率 72.7%。统计图通过代码生成扩展（Matplotlib），整体提升 +4.1%。与 Agent Banana（图像编辑框架）形成"生成 vs 编辑"的互补关系，共享"Agentic + 图像生成/编辑模型"技术范式。局限：光栅输出不可编辑、风格标准化与多样性权衡、细粒度忠实度仍有差距。

## [2026-05-07] ingest | Qwen2.5 技术报告

- Source: [[Clippings/Qwen2.5 Technical Report]]
- Created:
  - Sources: [[Wiki/Sources/Qwen2.5 技术报告]]
- Updated:
  - [[Wiki/index]] — 添加 Qwen2.5 技术报告来源条目
- Notes: 阿里巴巴 Qwen 团队 2024 年发表的 Qwen2.5 技术报告（arXiv:2412.15115）。Qwen 系列第二代半版本，是从 Qwen2 到 Qwen3 的关键过渡。核心贡献：(1) 预训练数据从 7T 扩展到 18T tokens，通过 Qwen2-Instruct 数据质量过滤器、Math/Coder 专用数据整合、合成数据增强和领域配比优化四大手段提升数据质量；(2) 后训练引入 100 万+ SFT 样本和两阶段 RL（离线 DPO 15 万对 + 在线 GRPO），显著提升人类偏好对齐、长文本生成（8K token 输出）和指令遵循；(3) 全尺寸覆盖 0.5B-72B 密集模型 + MoE Turbo/Plus，Qwen2.5-72B-Instruct 以约 1/5 参数量匹敌 Llama-3-405B-Instruct；(4) 长上下文能力通过 YARN+DCA 实现 4 倍扩展，Turbo 支持 100 万 token 并通过稀疏注意力实现 12.5 倍计算量降低；(5) 奖励模型评估发现 Goodhart 定律问题——单一 RM 基准过优化会损害其他基准和下游 RL 性能。与 Qwen 第一代和 Qwen3 形成完整技术演进链。关键局限：文化细微理解（BLEnD）仍需提升，长上下文 RL 训练因缺乏合适 RM 而缺失，多语言翻译偏弱。
- Wiki 当前规模: Sources: 103 (+1)；总页面: 175+

## [2026-05-07] ingest | Gen-Searcher

- Source: [[Clippings/Gen-Searcher: Reinforcing Agentic Search for Image Generation]]
- Created: [[Wiki/Sources/Gen-Searcher]]
- Updated: [[Wiki/index]]
- Notes: 首个搜索增强图像生成智能体 (CUHK/UCLA/UCB, 2026)。核心贡献：(1) 数据管线构造 17K 搜索密集型图像生成样本 + 630 人工验证 KnowGen 基准；(2) 两阶段训练 SFT + agentic GRPO RL，双重奖励（文本 + 图像）解决开源生成器高方差问题；(3) Qwen-Image KnowGen +16.5 点，WISE +0.15，跨生成器零样本迁移（Seedream 4.5 +16.3 点）。与 Mind-Brush/VisionCreator 等 prompt 工作流方法的关键区别在于通过 RL 训练自适应搜索策略而非手工设计。局限：依赖闭源模型构建数据、下游生成器能力仍是瓶颈。

## [2026-05-07] ingest | Scalable watermarking for identifying large language model outputs

- Source: [[Clippings/Scalable watermarking for identifying large language model outputs]]
- Created:
  - Sources: [[Wiki/Sources/LLM 可扩展水印]]
- Updated: [[Wiki/index]]
- Notes: Google DeepMind 2024 年 Nature 论文，提出 SynthID-Text 生成式文本水印方案。核心创新是 Tournament sampling（锦标赛采样）——通过多轮淘汰选择在水印函数上得分较高的 token，实现比 Gumbel 采样（非失真）和 Soft Red List（失真）更优的检测率。关键成果：(1) Gemini 2000 万响应在线实测，点赞/踩率差异统计不显著，首次证明水印在大规模生产中质量无损；(2) 提出水印与推测采样的集成算法，使水印在生产系统中零额外延迟；(3) 单 token / 单序列无失真的形式化定义厘清了文献中的概念混乱。局限：需要 LLM 提供方主动实施、对开源模型难以强制、易受改写攻击削弱。已部署于 Gemini 和 Gemini Advanced，是首个大规模生产部署的生成式文本水印。
- Wiki 当前规模: Sources: 105 (+1)；总页面: 176+

## [2026-05-07] ingest | 生成式推荐综述

- Source: [[Clippings/A Survey on Generative Recommendation: Data, Model, and Tasks]]
- Created:
  - Sources: [[Wiki/Sources/生成式推荐综述]]
- Updated:
  - [[Wiki/index]] — 添加生成式推荐综述来源条目
- Notes: 合肥工业大学 + 新加坡国立大学 2025 年综述 (arXiv:2510.27157)，覆盖 200+ 篇论文。核心贡献：(1) 提出数据-模型-任务三元框架系统梳理生成式推荐全景；(2) 三条技术路线——LLM-based（文本提示/协同信号/物品 Token 化三种对齐方式 + SFT/SSL/RL/DPO 四种训练范式）、LRM（HSTU 验证推荐缩放定律 + OneRec 端到端架构替代级联管道）、Diffusion-based（增广数据生成 + 目标物品生成）；(3) 任务层创新——Top-K 生成接地（约束解码/后过滤/提示增强）、个性化内容生成、对话推荐、可解释推荐、推荐推理（显式/隐式/LLM 增强）。关键发现：HSTU 1.5T 参数 LRM 在推荐上验证缩放定律、OneRec 端到端架构观看时长+1.68% 且运行成本仅为级联架构 10.6%。开放挑战：静态基准不适合评估交互式推荐助手、流行度/公平性/位置偏差、文本攻击鲁棒性、自回归推理效率。与已有 OneRec 系列（OneRec/OneSug/EGA-V2/OneRec-Think/OneRec-V2/OpenOneRec）形成系统性综述与代表性工业实践的互补。
- Wiki 当前规模: Sources: 106 (+1)；总页面: 177+

## [2026-05-07] ingest | The Rise and Potential of Large Language Model Based Agents: A Survey

- Source: [[Clippings/The Rise and Potential of Large Language Model Based Agents: A Survey]]
- Created:
  - Sources: [[Wiki/Sources/LLM Agent 综述 2023]]
- Updated:
  - [[Wiki/index]] — 添加 LLM Agent 综述 2023 来源条目
  - [[Wiki/Topics/LLM Agent 与工具使用]] — 添加新来源，来源数 5→6
- Notes: 复旦 NLP 组 2023 年 9 月综述（arXiv 2309.07864），LLM-based Agent 领域首篇全面系统综述之一。提出 Brain-Perception-Action 三模块通用框架，覆盖单智能体/多智能体/人机协作三大应用场景，并深入探讨 Agent 社会仿真（行为/人格/涌现社会现象）。与 [[Wiki/Sources/LLM Agent 综述 2024]]（人大四模块框架 Profile-Memory-Planning-Action）和 [[Wiki/Sources/Agent AI Survey 2024]]（多模态 Agent AI）形成互补三角关系：本综述提供宏观视野和哲学基础，人大综述提供架构分析，Agent AI 综述深入多模态/具身方向。

## [2026-05-07] ingest | Qwen3-VL 技术报告
- Source: [[Clippings/Qwen3-VL Technical Report]]
- Created: [[Wiki/Sources/Qwen3-VL 技术报告]]
- Updated: [[Wiki/index]]
- Notes: 阿里巴巴 Qwen 团队 2025 年 VLM 技术报告。Interleaved MRoPE + DeepStack 跨层融合 + 文本时间戳，原生 256K 上下文，纯文本能力不退化。Dense + MoE 全尺寸覆盖。多模态推理/文档理解/视频理解全面领先。

## [2026-05-07] ingest | DeepSeek V4
- Source: [[Clippings/DeepSeek V4]]
- Created: [[Wiki/Sources/DeepSeek V4]]
- Updated: [[Wiki/index]]
- Notes: DeepSeek V4 系列（Pro 1.6T/49B 激活，Flash 284B/13B 激活）。混合 CSA+HCA 注意力架构，百万级上下文 FLOPs 仅为 V3.2 的 27%。Muon 优化器、FP4 量化感知训练。Codeforces 排名人类第 23。

## [2026-05-07] ingest | OpenAI o1 System Card
- Source: [[Clippings/OpenAI o1 System Card]]
- Created: [[Wiki/Sources/OpenAI o1 System Card]]
- Updated: [[Wiki/index]]
- Notes: OpenAI 2024 年推理模型系统卡。推理链驱动安全对齐，deliberative alignment，指令层级。Apollo Research 发现 99% 否认率、37% 工具性对齐伪装。Preparedness Framework 四维度风险分级。

## [2026-05-07] ingest | Qwen3-VL-Embedding and Reranker
- Source: [[Clippings/Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for Multimodal Retrieval]]
- Created: [[Wiki/Sources/Qwen3-VL-Embedding and Reranker]]
- Updated: [[Wiki/index]]
- Notes: 阿里 Tongyi Lab 2026 年多模态检索框架。基于 Qwen3-VL backbone，Bi-encoder Embedding + Cross-encoder Reranker。8B MMEB-V2 77.8 排名第一。Matryoshka 表示学习 + 量化感知训练。32K 上下文，30+ 语言。

## [2026-05-07] ingest | GLM-5
- Source: [[Clippings/GLM-5: from Vibe Coding to Agentic Engineering]]
- Created: [[Wiki/Sources/GLM-5 从 Vibe Coding 到 Agentic Engineering]]
- Updated: [[Wiki/index]]
- Notes: 智谱 AI & 清华 2025 年。744B MoE + DSA 稀疏注意力 + Muon Split 优化器。27T tokens 预训练。Vibe Coding→Agentic Engineering 范式转变。SWE/终端/搜索/幻灯片四类 Agent 环境。CC-Bench-V2 真实工程评估。中国芯片全栈适配。

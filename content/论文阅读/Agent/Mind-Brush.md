---
title: "Mind-Brush: Integrating Agentic Cognitive Search and Reasoning into Image Generation"
authors:
  - Jun He
institutions: SYSU
aliases:
  - Mind-Brush
date: 2026-04-19
publish_date:
venue: ""
tags:
  - 论文
url: https://arxiv.org/abs/2602.01756
code: https://github.com/PicoTrex/Mind-Brush
rating: ⭐⭐⭐⭐⭐
status:
---
Mind-Brush 是一个免训练的智能体框架，通过集成多模态主动检索与显式逻辑推理工具，将传统的被动文本到图像生成转变为动态的“思考-检索-创作”认知工作流，从而解决了模型在复杂推理和实时长尾知识上的生成难题 。

提出的是框架，没有训练模型

## Intro

### Motivation

由于模型预训练数据的截止时间限制，其认知边界是静态且封闭的，导致它们在面对实时动态的世界或复杂的长尾 IP 概念时，表现出显著的能力缺失 。

### 本文的解决方案与贡献

1. 提出 **Mind-Brush**，一个统一的智能体框架。它不再将生成视为单步推理，而是模拟人类艺术家的“思考-研究-创作”过程，主动获取多模态证据并推导隐式约束 。
    
2. 构建了 **Mind-Bench** 评估基准，专门用于测试涉及动态外部知识和复杂推理的生成能力，弥补了现有基准（如 GenEval）仅关注指令遵循的短板 。
    
3. 通过实验证明了该框架的强大效能，使开源的 Qwen-Image 能够匹敌甚至超越部分闭源专有模型，并在现有的 WISE 和 RISEBench 测试中刷新了 SOTA 成绩 。


## Method 核心方法

作者将 Mind-Brush 建模为一个分层顺序决策过程，主要分为以下几个关键阶段：

1. **认知缺口检测 (Cognitive Gap Detection)：** Intent Analysis Agent作为元规划器，利用 5W1H（What, When, Where, Why, Who, How）范式将用户的文本和图像输入映射到结构化语义空间中 。它会严格检测模型内部知识的盲区，并将这些缺失的信息形式化为一组显式的原子问题 $Q_{gap}$。基于 $Q_{gap}$，系统会生成动态的执行策略 $\mathcal{S}_{plan}$，决定是将任务路由到“检索分支”还是“推理分支” 。
    
2. **自适应知识补全 (Adaptive Knowledge Completion)：**
    
    - **外部知识锚定 (Cognition Search Agent)**：对于涉及 OOD 实体或动态事件的缺口，系统调用搜索智能体。它首先生成精确的文本查询获取事实文档 $\mathcal{T}_{ref}$，然后通过一个“双重更新操作”：将检索到的概念注入回用户指令中更新上下文，同时校准视觉查询，以确保随后检索到的参考图像 $\mathcal{I}_{ref}$ 严格符合已验证的事实 。
        
    - **内部逻辑推导 (CoT Knowledge Reasoning Agent)**：当遇到需要求解输入图像中的数学题或推断空间关系时，推理引擎会被激活。它吸收用户指令、图像以及初步搜索到的证据 $\mathcal{E}_{search}$，执行多步思维链推理，输出明确的逻辑结论 $\mathcal{R}_{cot}$，解决隐式冲突 。
        
3. **约束生成 (Constrained Generation)：** 为防止外部信息的冗余引入噪声，概念审查智能体（Concept Review Agent）会对收集到的所有证据 $\mathcal{E}$ 进行过滤和融合，重写为一个结构化的“主提示词” $P_{master}$ 。最终，统一图像生成智能体在 $P_{master}$ 和过滤后的视觉线索 $V_{in}$ 双重条件下，动态选择“生成”或“编辑”模式来合成最终图像，确保极高的事实保真度

![image.png](https://chenshi-1327871112.cos.ap-chongqing.myqcloud.com/obsidian20260419135551157.png)

## 实验/评估/结果

### 实验设置

- **模型框架**：为了确保公平，Mind-Brush 在所有实验中保持一致。图像生成的底层采用 **Qwen-Image**（用于文本引导）和 **Qwen-Image-Edit-2512**（用于图像引导的编辑） 。所有智能体的底层大语言模型（MLLM）均由 **GPT-5.1** 驱动，并调用 Google Search API 限制检索返回数量以控制 Token 成本 。
    
- **评测基准**：使用了自建的 **Mind-Bench** 以及学术界认可的 **WISE** 和 **RISEBench** 。
    
- **Mind-Bench 的特殊指标**：采用 Checklist-based Strict Accuracy (CSA)。该指标利用 MLLM 作为裁判，对人工校验的细粒度清单进行严苛的“全有或全无（All-or-Nothing）”打分，只有清单所有子项全绿，该生成才算正确 。

#### 在 Mind-Bench 上的表现

![image.png](https://chenshi-1327871112.cos.ap-chongqing.myqcloud.com/obsidian20260419135745890.png)
![image.png](https://chenshi-1327871112.cos.ap-chongqing.myqcloud.com/obsidian20260419140139593.png)

- **Mind-Bench 表现**：纯开源的 Qwen-Image 基础准确率仅为 0.02，但在接入 Mind-Brush 框架后，其准确率暴涨至 **0.31**，相对提升了14倍以上 。

#### WISE
![image.png](https://chenshi-1327871112.cos.ap-chongqing.myqcloud.com/obsidian20260419145104902.png)

#### Ablation
![image.png|728](https://chenshi-1327871112.cos.ap-chongqing.myqcloud.com/obsidian20260419141933663.png)
![image.png](https://chenshi-1327871112.cos.ap-chongqing.myqcloud.com/obsidian20260419142157894.png)
- “搜索”和“推理”模块同时部署能产生巨大的协同效应 。
- 当把底座模型从开源的 Qwen3-VL 替换为闭源的 GPT-5.1 时，在图像解码器不变的情况下，性能直接跃升 29.2% 。这证明了在 Agentic Generation 范式下，**MLLM 的大脑智商是决定最终生成质量的最主导因素**。

## 结论

作者得出结论：Mind-Brush 成功将文生图的范式从“被动解码”拉升到了“主动认知工作流”的高度。通过有机编排意图分析、多模态锚定和思维链推理，该框架有效弥合了模糊指令与严谨物理现实之间的沟壑。Mind-Bench 的引入及其严苛的测试结果证明了“搜索+推理”对于下一代复杂视觉合成系统的必要性与巨大潜力 。

## 思考

### 优点

使用轻量级的 Qwen-Image-Edit 配合强大的大脑，

### 缺点

没有说明如何对检索到的信息进行清洗的
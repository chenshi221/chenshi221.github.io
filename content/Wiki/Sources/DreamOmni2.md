---
type: source
status: processed
source_file: "[[Clippings/DreamOmni2 Multimodal Instruction-based Editing and Generation.md]]"
title: "DreamOmni2: Multimodal Instruction-based Editing and Generation"
site: "arXiv (2510.06679)"
author: "Bin Xia, Bohao Peng, Yuechen Zhang, Junjia Huang, Jiyang Liu, Jingyao Li, Haoru Tan, Sitong Wu, Chengyao Wang, Yitong Wang, Xinglong Wu, Bei Yu, Jiaya Jia"
published: "2025-10"
processed: "2026-04-29"
updated: "2026-04-29"
tags: [image-editing, image-generation, diffusion, multimodal, VLM, DiT]
aliases: [DreamOmni2]
---

# DreamOmni2: Multimodal Instruction-based Editing and Generation

> 多模态指令图像编辑与生成。来源：[[Clippings/DreamOmni2 Multimodal Instruction-based Editing and Generation.md]]

## 核心结论

- 提出了两个**新任务**：多模态指令编辑（multimodal instruction-based editing）和多模态指令生成（multimodal instruction-based generation），支持**同时使用文本和图像作为指令**，并扩展到**抽象概念**（纹理、材质、姿态、发型、设计风格等）而非仅限具体物体。
- 解决了两个核心挑战：**训练数据缺乏**和**多图像输入框架**。提出三步数据合成管线，以及索引编码 + 位置编码偏移 + VLM 联合训练的框架。
- 在自建 benchmark 上，DreamOmni2 在人类评估中超越 GPT-4o 和 Nano Banana 等闭源商业模型，在 concrete object 编辑上人评 0.6098 vs GPT-4o 的 0.5610，abstract attribution 编辑上 0.6829 vs 0.5793。

## 关键事实

### 提出的新任务

| 维度 | 传统指令编辑 | 多模态指令编辑（本文） |
|------|------------|---------------------|
| 指令形式 | 仅文本 | 文本 + 参考图像 |
| 编辑对象 | 具体物体 | 具体物体 + 抽象属性 |
| 参考图像 | 无 | 支持 1-5 张参考图 |

### 数据合成管线（三阶段）

1. **Stage 1 — Feature Mixing**：双分支结构同时生成源图像和目标图像，交换注意力特征使两张图共享相同物体/属性。相比 UNO 的 diptych 方法，分辨率不减半、无边界混合、质量更高。
2. **Stage 2 — 编辑数据生成**：用 Stage 1 数据训练提取模型 → 从目标图像提取物体/属性生成参考图 → 用指令编辑模型修改目标图像得到源图 → 形成（源图 + 指令 + 参考图 + 目标图）四元组。
3. **Stage 3 — 生成数据生成**：从 Stage 2 的源图中用提取模型生成新参考图，形成（多张参考图 + 指令 + 目标图）训练数据。

### 框架创新

1. **索引编码 + 位置编码偏移**：DiT 中位置编码无法区分多张输入图像。索引编码让模型识别"图1""图2"引用；位置编码偏移防止像素混淆和复制粘贴效应。
2. **VLM 联合训练**：用 Qwen2.5-VL 7B 将非结构化用户指令翻译为结构化标准格式，然后输入生成/编辑模型。仅用约 10 A100 小时微调 VLM，384 A100 小时训练 LoRA。

### Benchmark

- 205 个多模态指令编辑测试 + 114 个生成测试
- 覆盖 1-5 张参考图像、局部/全局属性、具体物体
- 使用真实图像评估

### 消融实验关键数据

| 方案 | 编辑(Concrete) | 编辑(Abstract) | 生成(Concrete) | 生成(Abstract) |
|------|:---------:|:---------:|:---------:|:---------:|
| Base (Kontext) | 0.1220 | 0.0122 | 0.3750 | 0.1222 |
| +数据 | 0.3659 | 0.3171 | 0.4583 | 0.3444 |
| +VLM | 0.2439 | 0.3415 | 0.5417 | 0.4778 |
| **+数据+VLM** | **0.6585** | **0.6280** | **0.6667** | **0.6333** |

## 方法或论证路径

- 发现当前指令编辑和主体驱动生成的局限 → 提出多模态指令编辑与生成新任务 → 识别两大挑战（数据、框架） → 三步合成数据管道 → 索引编码 + 位置编码偏移 + VLM 联合训练 → 自建 benchmark 验证。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Concepts/多模态指令编辑与生成]] — 该任务的概念阐述
- 关联：[[Wiki/Entities/DreamOmni2]] — 模型实体页
- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]] — 所属研究方向
- 扩展了 [[DreamOmni]]（CVPR 2025）到多模态指令领域
- 基座模型为 [[Flux Kontext]]，使用 LoRA 训练保留原有能力
- VLM 部分基于 [[Qwen2.5-VL]] 7B

## 可能的矛盾或待核实点

- 人类评估来自"专业工程师"，但未说明具体数量和评价标准，可能存在主观偏差。
- VLM 评估（Gemini 2.5 / 豆包 1.6）与人类评估存在差距（例如 GPT-4o 的 VLM 分数高于 DreamOmni2 但人评低于），说明 VLM 作为评估器可能不够可靠。
- 该工作与 [[DreamOmni]]（CVPR 2025，同样来自 CUHK-ByteDance）的关系：DreamOmni 统一生成和编辑，但只处理单图像输入 + 文本指令。DreamOmni2 扩展到多图像输入 + 多模态指令。
- 数据合成依赖多个预训练模型（T2I、编辑模型、提取模型），可能引入复合误差。

## 后续问题

- Feature mixing 与传统 cross-attention 操作的对比和适用范围？
- 该框架能否扩展到视频编辑？（与 [[DreamVE]] 的关系）
- VLM 联合训练是否可以用更小的 VLM（如 0.5B/1.8B）达到类似效果？

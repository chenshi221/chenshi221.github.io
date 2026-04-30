---
type: source
status: processed
source_file: "[[Clippings/InstructPix2Pix Learning to Follow Image Editing Instructions.md]]"
title: "InstructPix2Pix: Learning to Follow Image Editing Instructions"
site: "arXiv (2211.09800)"
author: "Tim Brooks, Aleksander Holynski, Alexei A. Efros"
published: "2022-11"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [image-editing, instruction-following, diffusion, data-generation, foundational]
aliases: [InstructPix2Pix, IP2P]
---

# InstructPix2Pix: Learning to Follow Image Editing Instructions

UC Berkeley，2022。**指令式图像编辑的奠基性工作**。

## 核心结论

- 首次提出根据人类自然语言指令编辑图像的方法：输入原图 + 文字指令 → 编辑后图像。
- 核心创新：利用两个预训练大模型（GPT-3 + Stable Diffusion + Prompt-to-Prompt）**自动生成**配对训练数据，无需人工标注。
- 训练出 **InstructPix2Pix** 条件扩散模型，前向一次推理即可完成编辑，无需 per-example 微调或 inversion。
- 在完全合成的数据上训练，零样本泛化到真实图像和人类编写的指令。

## 数据生成管线（两阶段）

### 阶段 1：生成指令和配对 Caption
- 在 LAION 真实 caption 上微调 GPT-3：输入原始 caption → 输出编辑指令 + 编辑后 caption
- 先用人工标注 700 个样本做微调，再大规模生成

### 阶段 2：生成配对图像
- 将 4,445,660 个 caption 对用 Stable Diffusion + Prompt-to-Prompt 转为图像对
- 筛选后得到 **454,445 个高质量训练样本**

## 模型架构

- 基于 Stable Diffusion 条件扩散模型
- 输入：原图（encoded）+ 编辑指令（CLIP text embedding）
- 输出：编辑后图像
- 使用 classifier-free guidance 平衡指令跟随和图像质量

## 历史地位

- 开创了"指令式图像编辑"这个子方向
- 数据生成范式（LLM + T2I → 配对数据）影响深远，后续几乎所有指令编辑工作都沿用了这一思路
- InstructPix2Pix 被 EmoEdit、UltraEdit 等工作直接使用或改进

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 是 [[Wiki/Entities/EmoEdit]] 的基座编辑模型（EmoEdit 在 IP2P 基础上加 Emotion adapter）
- UltraEdit、AnyEdit 等后续工作都在 IP2P 基础上改进数据质量和编辑多样性

## 局限性

- 图像质量受限于当时的 Stable Diffusion（512×512）
- 编辑类型受限（主要覆盖添加/替换/删除/风格迁移）
- 数据全为合成，可能存在 domain gap

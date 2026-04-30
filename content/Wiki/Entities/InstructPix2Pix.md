---
type: entity
status: active
created: "2026-04-30"
updated: "2026-04-30"
aliases: [IP2P, InstructPix2Pix]
tags: [image-editing, instruction-following, diffusion, foundational]
sources: ["[[Wiki/Sources/InstructPix2Pix]]"]
---

# InstructPix2Pix

## 基本信息

| 属性 | 值 |
|------|-----|
| 全称 | InstructPix2Pix: Learning to Follow Image Editing Instructions |
| 作者 | Tim Brooks, Aleksander Holynski, Alexei A. Efros |
| 机构 | UC Berkeley |
| 年份 | 2022 (arXiv 2211.09800) |
| 类型 | 条件扩散模型 |

## 历史地位

**指令式图像编辑的奠基工作**。首次提出"输入原图 + 自然语言指令 → 编辑后图像"的范式，启发了后续几乎所有指令编辑研究。

## 核心贡献

1. **数据生成范式**：GPT-3 微调生成编辑指令 + SD + Prompt-to-Prompt 生成配对图像 → 454K 训练样本，无需人工标注
2. **零样本泛化**：完全在合成数据上训练，可直接泛化到真实图像和人类编写的指令
3. **前向推理**：单次前向传播完成编辑，无需 per-example 微调或 inversion

## 架构

- 基座：Stable Diffusion 条件扩散模型
- 输入：原图（VAE 编码）+ 编辑指令（CLIP text embedding）
- 训练：Classifier-free guidance

## 被推使用 / 改进

| 工作 | 如何使用 |
|------|----------|
| EmoEdit | 在 IP2P 基础上加 Emotion adapter |
| UltraEdit | 改进 IP2P 的数据质量和多样性 |
| AnyEdit | 在 IP2P 数据范式上加 task routing |
| EditWorld | 扩展指令类型到物理世界动态 |

## 局限性

- 图像质量受限于 SD 1.x（512×512）
- 复杂物理编辑（光照变化、阴影）效果差
- 数据全为合成，对极端真实场景有 domain gap

## 在 Wiki 中的关联

- 概念：[[Wiki/Concepts/多模态指令编辑与生成]]
- 比较：[[Wiki/Comparisons/编辑方法能力演进]]
- 主题：[[Wiki/Topics/扩散模型图像编辑与生成]]

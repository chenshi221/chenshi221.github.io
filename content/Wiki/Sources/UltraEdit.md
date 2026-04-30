---
type: source
status: processed
source_file: "[[Clippings/UltraEdit Instruction-based Fine-Grained Image Editing at Scale.md]]"
title: "UltraEdit: Instruction-based Fine-Grained Image Editing at Scale"
site: "arXiv (2407.05282v2)"
author: "Haozhe Zhao, Xiaojian Ma, Liang Chen, et al."
published: "2024-07"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [image-editing, dataset, instruction-following, diffusion, large-scale]
aliases: [UltraEdit]
---

# UltraEdit: Instruction-based Fine-Grained Image Editing at Scale

北京大学 + BIGAI + 清华 + UCLA + UIUC，2024。

## 核心结论

- 提出 **UltraEdit**，约 **400 万编辑样本**的大规模自动生成数据集，系统性地解决了 InstructPix2Pix 等现有数据集的不足。
- 三大优势：(1) 利用 LLM 创造力 + 人类标注者 in-context 编辑示例，覆盖更广的编辑指令类型；(2) 基于真实图像（照片 + 艺术品），多样性更高；(3) 编辑质量显著优于纯合成数据。
- 基于 UltraEdit 训练的编辑模型在多个 benchmark 上大幅超越 InstructPix2Pix。

## 与 InstructPix2Pix 的关键改进

| 维度 | InstructPix2Pix | UltraEdit |
|------|-----------------|-----------|
| 数据规模 | 450K | ~4M |
| 图像来源 | 纯生成（SD） | 真实图像为主 |
| 指令多样性 | 依赖 GPT-3 生成 | LLM + 人类示例 in-context |
| 编辑粒度 | 粗粒度 | 细粒度 |
| 质量保证 | 自动筛选 | 多级质量控制 |

## 数据生成策略

1. **指令生成**：LLM + 人类标注者提供的 in-context editing examples → 更多样、更自然的指令
2. **图像来源**：真实照片和艺术品（非纯合成）
3. **编辑执行**：利用改进的编辑模型生成高质量 target
4. **质量筛选**：多维度自动 + 人工审核

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]、[[Wiki/Sources/InstructPix2Pix]]
- 是 InstructPix2Pix 数据范式的升级版，AnyEdit 在同一方向上继续推进
- 大规模高质量数据对 [[Wiki/Sources/EmoEdit]] 等专业化编辑任务也有参考价值

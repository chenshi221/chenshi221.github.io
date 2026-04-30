---
type: source
status: processed
source_file: "[[Clippings/EditWorld Simulating World Dynamics for Instruction-Following Image Editing.md]]"
title: "EditWorld: Simulating World Dynamics for Instruction-Following Image Editing"
site: "arXiv (2405.14785v1)"
author: "Ling Yang, Bohan Zeng, Jiaming Liu, et al."
published: "2024-05"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [image-editing, world-dynamics, physics, instruction-following, reasoning]
aliases: [EditWorld]
---

# EditWorld: Simulating World Dynamics for Instruction-Following Image Editing

北京大学 + Tiamat AI + Skywork AI + Mila，2024。

## 核心结论

- 提出新任务 **world-instructed image editing**：不仅编辑图像内容，还要理解物理世界的动态规律。
- 指出传统指令编辑（添加/替换/删除）缺乏对物理世界动态（重力、光照变化、物体交互、时序变化等）的理解。
- 构建 worldwide instruction 数据集（GPT-3.5 + Video-LLaVA + SDXL 联合生成）。
- 设计 post-edit strategy 提升指令跟随能力。

## 核心洞察

传统编辑（如 InstructPix2Pix）可以"加一只猫"，但无法理解：
- 🌧️ "让这条路看起来像刚下过雨"（需要理解湿润路面的光照反射）
- 🍂 "把场景变成秋天"（需要全局色调 + 落叶 + 光影变化）
- ⏰ "把这个白天场景变成黄昏"（需要全局光照 + 阴影角度变化）

这些都是需要理解 world dynamics 的编辑。

## 方法

1. **任务定义与分类**：将 world-instructed 编辑按物理动态类型分类
2. **数据生成**：GPT-3.5 生成指令 → Video-LLaVA 分析动态 → SDXL 生成配对图像
3. **Post-edit Strategy**：编辑后进行物理合理性验证和微调

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 在指令编辑的方向上提出了更高要求：从"语义编辑"到"物理感知编辑"
- 与 [[Wiki/Sources/EmoEdit]] 的情感编辑形成对照：情感编辑是"心理感知"，EditWorld 是"物理感知"
- 需要理解场景的因果结构 → 与后续的推理类编辑方法（GoT、Mind-Brush）有共同趋势

## 局限性

- 物理动态模拟仍受限于生成模型的能力
- 评估"物理合理性"本身就是一个开放问题

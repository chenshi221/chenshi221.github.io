---
type: entity
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - EditWorld
tags:
  - image-editing
  - world-dynamics
  - physics
  - instruction-following
  - reasoning
sources:
  - '[[Wiki/Sources/EditWorld]]'
---

# EditWorld

## 基本信息

| 属性 | 值 |
|------|-----|
| 全称 | EditWorld: Simulating World Dynamics for Instruction-Following Image Editing |
| 作者 | Ling Yang, Bohan Zeng, Jiaming Liu, Hong Li, Minghao Xu, Wentao Zhang, Shuicheng Yan |
| 机构 | 北京大学 + Tiamat AI + Skywork AI + Mila |
| 年份 | 2024 (arXiv 2405.14785) |
| 类型 | 世界指令编辑框架 |

## 核心创新

提出 **world-instructed image editing** 新任务：不仅要编辑图像内容，还要理解物理世界的动态规律。

```
传统指令编辑：文本 → 添加/替换/删除 → 图像
EditWorld：   文本 → 理解物理动态 → 模拟世界规律 → 图像
```

## 为什么传统编辑不够？

传统编辑（如 InstructPix2Pix）可以做"加一只猫"，但无法理解：

- 🌧️ "让这条路看起来像刚下过雨" → 需要理解湿润路面的光照反射
- 🍂 "把场景变成秋天" → 需要全局色调 + 落叶 + 光影变化
- ⏰ "把这个白天场景变成黄昏" → 需要全局光照 + 阴影角度变化
- 🌊 "让水面有微风拂过的涟漪" → 需要流体动力学直觉

这些都是需要理解 **world dynamics** 的编辑。

## 方法

1. **任务分类**：将 world-instructed 编辑按物理动态类型（光照、天气、季节、材质等）分类
2. **数据生成**：GPT-3.5 生成指令 → Video-LLaVA 分析动态 → SDXL 生成配对图像
3. **Post-edit Strategy**：编辑后进行物理合理性验证和微调，提升指令跟随能力

## 在编辑演进中的位置

| 代际 | 模型 | 编辑语义层级 |
|------|------|-------------|
| 第一代 | InstructPix2Pix, ControlNet | 像素/空间控制 |
| 第二代 | EmoEdit, AnyEdit, UltraEdit | 语义/情感控制 |
| EditWorld | **物理动态感知** | 因果/物理规律 |
| 第三代 | GoT, Mind-Brush, VisionCreator | 推理/知识/Agent |

EditWorld 填补了"物理感知编辑"这个独特位置——介于语义编辑和推理编辑之间，关注的是**现实世界的因果律**而非语言推理。

## 在 Wiki 中的关联

- 来源：[[Wiki/Sources/EditWorld]]
- 主题：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 比较：[[Wiki/Comparisons/编辑方法能力演进]]
- 相关：[[Wiki/Entities/EmoEdit]]（情感感知 vs 物理感知），[[Wiki/Entities/GoT]]（推理编辑）

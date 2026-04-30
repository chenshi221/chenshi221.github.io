---
type: concept
status: active
created: "2026-04-30"
updated: "2026-04-30"
aliases: [emotional bias in AI, AI emotional skew, 生成模型情感偏差]
tags: [emotion, bias, AI-safety, evaluation]
sources: ["[[Wiki/Sources/Generating Fearful Images]]"]
confidence: medium
---

# AI 生成图像的情感偏差

## 定义

AI 图像生成模型在生成图像时系统性地偏向某些情感方向（特别是负面情感），独立于用户输入 prompt 的情感倾向。这是生成模型安全对齐的一个被低估的维度。

## 关键证据

**Generating Fearful Images (2024)** 研究发现：

1. AI 生成图像频繁偏向负面情感，特别是 **fear（恐惧）**
2. 即使 prompt 本身是中性的或正向的，生成图像的情感分布仍向负面偏移
3. 这种偏差在不同模型中均存在

## 可能的原因（待验证）

- **训练数据偏差**：训练数据中某些情感场景被过度表达
- **安全过滤的副作用**：过度过滤敏感内容可能导致模型在情感轴上产生意外偏移
- **架构特性**：扩散模型或 VLM 的某些设计选择可能自然偏向特定情感模式

## 与其他研究方向的关系

| 方向 | 关系 |
|------|------|
| [[Wiki/Concepts/情感图像编辑]] | 精确情感控制的方法需要建立在无偏差的基座模型上 |
| EmoArt [[Wiki/Sources/EmoArt]] | 艺术数据高度偏向 calm/positive，训练可能引入相反偏差 |
| AI Safety | 情感偏差是 AI 安全的一个子问题，但常被忽视 |

## 研究挑战

1. **情感识别本身有偏**：评估工具（情感分类器）的偏差可能与生成模型的偏差叠加
2. **跨文化差异**：情感表达和感知因文化而异，偏差的评估应跨文化
3. **可接受偏差的边界**：什么程度的情感偏差是"可接受的"？完全均匀分布也不是目标

## 开放问题

- 如何在不破坏安全对齐的前提下减少不必要的情感偏差？
- 情感偏差是否与模型的审美偏好（aesthetic bias）相关？
- 微调（如 EmoEdit 的 Emotion adapter）能否纠正底层偏差？

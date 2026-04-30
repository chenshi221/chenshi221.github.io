---
type: entity
status: active
created: "2026-04-30"
updated: "2026-04-30"
aliases: [EmoEdit]
tags: [emotion, image-editing, AIM, diffusion, adapter, model]
sources: ["[[Wiki/Sources/EmoEdit]]"]
---

# EmoEdit

## 基本信息

- **全称**：EmoEdit: Evoking Emotions through Image Manipulation
- **作者**：Jingyuan Yang, Jiawei Feng, Weibin Luo, Dani Lischinski, Daniel Cohen-Or, Hui Huang
- **机构**：深圳大学 + 希伯来大学 + 特拉维夫大学
- **年份**：2024
- **链接**：arXiv 2405.12661

## 核心贡献

1. **EmoEditSet**：首个大规模 AIM 数据集（40,120 对），含情感方向标签和内容指令
2. **Emotion Adapter**：基于 Q-Former 的即插即用情感适配器，可插入任意扩散模型
3. **Instruction Loss**：捕捉情感数据对中的语义变化

## 技术架构

- 情感归因：EmoSet → CLIP 聚类 → GPT-4V 摘要 → 情感因子树
- 数据生成：IP2P + 情感因子指令 → 四轮筛选
- 训练：Emotion adapter (Q-Former) + diffusion loss + instruction loss

## 性能亮点

- Emo-A (情感准确率) 50.09%，结构保持 (PSNR 16.62) 均最优
- 用户偏好 70-89%
- 可插拔到 ControlNet、BlipDiff、Composable Diffusion

## 在 Wiki 中的关联

- 概念：[[Wiki/Concepts/情感图像编辑]]
- 对比：[[Wiki/Entities/EmotiCrafter]]（离散 vs 连续情感）

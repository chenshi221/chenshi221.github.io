---
type: entity
status: active
created: "2026-04-30"
updated: "2026-04-30"
aliases: [EmoArt, EmoArt-130k]
tags: [emotion, dataset, art, benchmark]
sources: ["[[Wiki/Sources/EmoArt]]"]
---

# EmoArt Dataset

## 基本信息

- **全称**：EmoArt: A Multidimensional Dataset for Emotion-Aware Artistic Generation
- **作者**：Cheng Zhang, Hongxia Xie, Bin Wen, Songhan Zuo, Ruoxuan Zhang, Wen-Huang Cheng
- **机构**：吉林大学 + 国立台湾大学
- **年份**：2025
- **链接**：arXiv 2506.03652

## 数据集规模

| 维度 | 数据 |
|------|------|
| 图像数量 | 132,664 |
| 绘画风格 | 56 种 |
| 来源 | WikiArt, The Met, 国立亚洲艺术博物馆等 |
| 标注维度 | 5 层（描述 + 5 视觉属性 + A-V + 12 情感 + 治疗效果） |
| 标注方式 | GPT-4o + 人工验证（一致性 91.47%） |
| 许可证 | CC BY-NC 4.0 |

## 在 Wiki 中的关联

- 概念：[[Wiki/Concepts/Valence-Arousal 情感模型]]
- 数据源：可用于微调情感感知的生成模型
- 对比：EmoSet（3.3M 但标注更粗）

---
type: entity
status: active
created: "2026-04-30"
updated: "2026-04-30"
aliases: [EmotiCrafter, C-EICG]
tags: [emotion, image-generation, valence-arousal, continuous-emotion, diffusion, model]
sources: ["[[Wiki/Sources/EmotiCrafter]]"]
---

# EmotiCrafter

## 基本信息

- **全称**：EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model
- **作者**：Shengqi Dang, Yi He, Long Ling, Ziqing Qian, Nanxuan Zhao, Nan Cao
- **机构**：同济大学 + 上海创新研究院 + Adobe Research
- **年份**：2025
- **链接**：arXiv 2501.05710

## 核心贡献

1. 首次提出 **C-EICG（连续情感图像内容生成）** 任务
2. **Emotion-Embedding Network**：用 Emotion Injection Transformer（修改 GPT-2）将 V-A 值融入文本特征
3. **Scaled Residual + V-A Density Weighting** 损失函数

## 与 EmoEdit 的关键区别

| 维度 | EmoEdit | EmotiCrafter |
|------|---------|-------------|
| 任务 | 情感图像编辑（输入原图） | 情感图像生成（从文本生成） |
| 情感表示 | 8 类离散 | 连续 V-A 坐标 |
| 核心技术 | Emotion adapter (Q-Former) | EIT (modified GPT-2) |
| 基座模型 | InstructPix2Pix | Stable Diffusion XL |
| 控制粒度 | 8 类 | 任意精度（0.2 步长） |

## 性能亮点

- V/A-Error 最优，连续性远超 GPT-4+SDXL
- 支持 V-A 粒度 0.2 的精细情感控制
- 可 decouple 情感和语义内容

## 在 Wiki 中的关联

- 概念：[[Wiki/Concepts/Valence-Arousal 情感模型]]
- 对比：[[Wiki/Entities/EmoEdit]]

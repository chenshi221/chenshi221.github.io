---
type: source
status: processed
source_file: "[[Clippings/EmoEdit Evoking Emotions through Image Manipulation.md]]"
title: "EmoEdit: Evoking Emotions through Image Manipulation"
site: "arXiv (2405.12661v3)"
author: "Jingyuan Yang, Jiawei Feng, Weibin Luo, Dani Lischinski, Daniel Cohen-Or, Hui Huang"
published: "2024-05"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [emotion, image-editing, AIM, diffusion, adapter]
aliases: [EmoEdit]
---

# EmoEdit: Evoking Emotions through Image Manipulation

深圳大学 + 希伯来大学 + 特拉维夫大学，2024。

## 核心结论

- 提出 **EmoEdit**，一个内容感知的情感图像编辑（Affective Image Manipulation, AIM）框架，仅需情感词作为 prompt，即可修改图像以唤起特定情感。
- 传统 AIM 方法主要调整颜色和风格，无法产生精确深刻的情感转变。EmoEdit 引入内容修改（添加/修改物体、场景、动作、表情等）。
- 设计 **Emotion adapter**（基于 Q-Former），可作为即插即用模块为任意扩散模型注入情感知识。
- 提出 **instruction loss** 来捕捉数据对中的语义变化，与 diffusion loss 联合优化。

## 关键方法

### EmoEditSet 数据集构建

- **情感归因 (Emotion Attribution)**：基于 EmoSet 的 8 类情感（amusement, awe, contentment, excitement, anger, disgust, fear, sadness），用 CLIP 语义嵌入聚类，GPT-4V 分配内容摘要，构建情感因子树（object / scene / action / facial expression）。
- **数据构建**：从 MagicBrush、MA5K、Unsplash 收集图像，用 IP2P + 情感因子树指令生成目标候选，经 CLIP image similarity、CLIP text similarity、Aesthetic score、Emotion score 四轮筛选 + 人工审核。
- 最终：**40,120 对图像**，15,531 源图像，平均每张 2.6 个情感方向。

### Emotion Adapter

- 类似 Q-Former 架构，learned queries 作为情感字典，self-attention 从字典中选与目标情感相关的语义，cross-attention 结合输入图像选出最合适表示。
- 仅训练 adapter，冻结 IP2P 参数。

### Instruction Loss

- Diffusion loss 单独优化会过度强调像素相似性，产生颜色伪影。
- Instruction loss：让 emotion embedding 对齐内容指令的文本嵌入，确保语义清晰。

## 实验结果

- 在 PSNR / SSIM / LPIPS / CLIP-I / Emo-A / Emo-S 六项指标上全面优于 SDEdit、PnP、InsDiff、ControlNet、BlipDiff、CLVA、AIF。
- Emo-A (情感准确率)：50.09% vs 次优 38.21%（SDEdit）。
- 用户调研：70.12% 结构完整性投票，75.73% 情感保真度投票，89.12% 综合平衡投票。
- Emotion adapter 可插入 ControlNet、BlipDiff 等增强情感能力；也可扩展到 Composable Diffusion 做风格化情感生成。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]、[[Wiki/Topics/情感计算与图像生成]]
- 技术路径与 [[Wiki/Entities/DreamOmni2]] 形成对比：DreamOmni2 用 VLM 联合训练实现多模态指令编辑，EmoEdit 用 adapter + 情感因子树实现零样本情感编辑
- 使用 8 类离散情感，而 [[Wiki/Sources/EmotiCrafter]] 使用连续 V-A 模型

## 限制

- 情感因子树外还有大量视觉情感因子未覆盖
- 仅 8 类情感，依赖 EmoSet 带来潜在偏差
- 评估指标（情感准确率）依赖训练数据，需要更多人类反馈

## 后续问题

- 离散情感 vs 连续情感模型的优劣？
- Emotion adapter 是否能推广到视频或 3D 场景？
- 情感编辑的伦理边界在哪里？

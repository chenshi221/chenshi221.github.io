---
type: source
status: processed
source_file: "[[Clippings/AnyEdit Mastering Unified High-Quality Image Editing for Any Idea.md]]"
title: "AnyEdit: Mastering Unified High-Quality Image Editing for Any Idea"
site: "arXiv (2411.15738v3)"
author: "Qifan Yu, Wei Chow, Zhongqi Yue, et al."
published: "2024-11"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [image-editing, dataset, unified, diffusion, task-aware]
aliases: [AnyEdit]
---

# AnyEdit: Mastering Unified High-Quality Image Editing for Any Idea

浙江大学 + 南洋理工 + 蚂蚁集团，2024。

## 核心结论

- 提出 **AnyEdit** 数据集：**250 万高质量编辑对**，覆盖 **20+ 编辑类型**、**5 个领域**。
- 数据质量通过三方面保证：初始数据多样性、自适应编辑过程、编辑结果自动选择。
- 训练 **AnyEdit Stable Diffusion**：引入 task-aware routing 和 learnable task embedding，实现统一图像编辑。
- 在三个 benchmark 上一致提升扩散编辑模型性能。

## 关键设计

### 数据集三大支柱

1. **初始数据多样性**：覆盖多种图像域（自然、艺术、产品、人脸、场景）和编辑类型（替换、添加、删除、风格、颜色、纹理、姿态、表情等 20+）
2. **自适应编辑过程**：根据图像内容和编辑类型动态调整编辑策略
3. **自动选择**：训练评估模型自动筛选高质量编辑结果

### Task-Aware Routing

- 不同编辑类型需要不同的模型行为
- Learnable task embedding 编码编辑类型信息
- Task-aware routing 动态选择编辑路径
- 相比一刀切的统一模型，在每种编辑类型上都有提升

## 与 UltraEdit 的对比

| 维度 | UltraEdit | AnyEdit |
|------|-----------|---------|
| 规模 | ~4M | 2.5M |
| 编辑类型数 | 较多 | 20+ 明确定义 |
| 核心创新 | 真实图像 + LLM in-context | Task-aware routing |
| 训练方法 | 标准微调 | 带 task embedding 的统一训练 |

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]、[[Wiki/Sources/UltraEdit]]
- 与 UltraEdit 互补：一个侧重数据规模和多样性，一个侧重模型架构的统一性
- Task-aware routing 思路与 [[Wiki/Entities/DreamOmni2]] 的多任务统一思想相通

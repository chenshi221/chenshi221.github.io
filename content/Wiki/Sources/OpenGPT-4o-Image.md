---
type: source
status: processed
source_file: "[[Clippings/OpenGPT-4o-Image A Comprehensive Dataset for Advanced Image Generation and Editing.md]]"
title: "OpenGPT-4o-Image: A Comprehensive Dataset for Advanced Image Generation and Editing"
site: "arXiv (2509.24900v1)"
author: "Zhihong Chen, Xuehai Bai, Yang Shi, et al. (USTC + Kling Team + PKU + NJU + CASIA + THU)"
published: "2025-09"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [image-generation, image-editing, dataset, GPT-4o, taxonomy]
aliases: [OpenGPT-4o-Image]
---

# OpenGPT-4o-Image: A Comprehensive Dataset for Advanced Image Generation and Editing

USTC + Kling Team + 北大 + 南大 + 中科院 + 清华，2025。

## 核心结论

- 统一多模态模型的图像生成/编辑能力受限于训练数据的质量和覆盖度。
- 提出 **OpenGPT-4o-Image**：基于层级任务分类 + 自动化数据生成构建。
- **80K 高质量指令-图像对**，覆盖 10+ 任务类别。

## 数据特点

### 层级任务分类 (Hierarchical Task Taxonomy)

| 层级 | 示例 |
|------|------|
| 基础能力 | 文本渲染、风格控制 |
| 中级能力 | 物体替换、场景变换 |
| 高级能力 | 化学插图等科学图像、多指令同步编辑 |

### 独特价值

- 覆盖了科学图像（化学插图）等很少被其他数据集覆盖的领域
- 复杂指令编辑：要求同时执行多个操作的编辑任务

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 层级分类方法提供了一种系统化的编辑任务思考框架
- 来自 Kling Team（快手可灵），代表了工业界的编辑数据实践

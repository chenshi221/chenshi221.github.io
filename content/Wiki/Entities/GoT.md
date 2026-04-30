---
type: entity
status: active
created: "2026-04-30"
updated: "2026-04-30"
aliases: [GoT, Generation Chain-of-Thought]
tags: [image-generation, image-editing, reasoning, chain-of-thought, MLLM]
sources: ["[[Wiki/Sources/GoT]]"]
---

# GoT (Generation Chain-of-Thought)

## 基本信息

| 属性 | 值 |
|------|-----|
| 全称 | GoT: Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing |
| 作者 | Rongyao Fang, Chengqi Duan, et al. |
| 机构 | CUHK MMLab + HKU + SenseTime + Shanghai AI Lab + THU + BUAA |
| 年份 | 2025 (arXiv 2503.10639) |
| 类型 | 推理引导的生成框架 |

## 核心创新

首次将 **Chain-of-Thought 推理**引入图像生成和编辑：在生成像素之前，先显式推理语义关系和空间布局。

```
传统：文本 → [黑盒] → 图像
GoT： 文本 → 语义推理 → 空间规划 → 图像
```

## 架构

- **推理链生成**：Qwen2.5-VL 分析语义关系、规划空间布局
- **Semantic-Spatial Guidance Module**：将推理链中的语义和空间信息注入扩散模型
- **数据集**：9M+ 样本，每个含详细推理链

## 与同类工作的对比

| 维度 | GoT | Mind-Brush | VisionCreator |
|------|-----|------------|---------------|
| 推理方式 | 内部 CoT | 外部知识检索 + 推理 | UTPC 全流程 |
| 知识来源 | 模型内部先验 | 动态检索 | 内部先验 + 训练 |
| 代表场景 | 通用生成/编辑 | OOD 概念、实时信息 | 自主创建 Agent |

## 在 Wiki 中的关联

- 比较：[[Wiki/Comparisons/编辑方法能力演进]]
- 主题：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 相关：[[Wiki/Entities/Mind-Brush]], [[Wiki/Entities/VisionCreator]]

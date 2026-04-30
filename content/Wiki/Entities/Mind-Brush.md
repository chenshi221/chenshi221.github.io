---
type: entity
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - Mind-Brush
tags:
  - image-generation
  - agent
  - reasoning
  - knowledge-retrieval
  - benchmark
sources:
  - '[[Wiki/Sources/Mind-Brush]]'
---

# Mind-Brush

## 基本信息

| 属性 | 值 |
|------|-----|
| 全称 | Mind-Brush: Integrating Agentic Cognitive Search and Reasoning into Image Generation |
| 作者 | Jun He, Junyan Ye, Zilong Huang, Dongzhi Jiang, Chenjue Zhang, Leqi Zhu, Renrui Zhang, Xiang Zhang, Weijia Li |
| 年份 | 2026 (arXiv 2602.01756, ICML 投稿) |
| 类型 | Agent 驱动的图像生成框架 |

## 核心创新

将图像生成从"静态文本到像素解码"升级为 **think-research-create** 动态知识驱动工作流，模拟人类的认知过程。

```
传统模型：文本 → [黑盒解码] → 图像
Mind-Brush：文本 → Think(理解意图) → Research(检索知识) → Create(生成图像)
```

## Think-Research-Create 范式

| 阶段 | 做什么 | 为什么需要 |
|------|--------|-----------|
| **Think** | 理解用户隐式意图，分解为子问题 | 用户说"画一个赛博朋克风格的茶馆"，模型需要知道赛博朋克的视觉特征+茶馆的传统元素 |
| **Research** | 主动检索多模态证据（网络搜索、数据库查询） | 模型训练数据中没有"2026年最流行的球鞋"的知识 |
| **Create** | 基于检索到的知识生成图像 | 接地于真实世界知识的生成 |

## 能力突破

- **zero-to-one 能力飞跃**：Qwen-Image baseline 在 Mind-Bench 上从零到可用
- **OOD 概念接地**：实时新闻、新兴概念（模型训练时不存在的事物）
- **隐式约束推理**：数学推理（"画一个对称的六边形花园"）、地理推理（"画出尼罗河三角洲的卫星视角"）

## 与同类 Agent 模型的对比

| 维度 | GoT | Mind-Brush | VisionCreator |
|------|-----|------------|---------------|
| 推理方式 | 内部 CoT 推理 | 外部知识检索 + 推理 | UTPC 全流程 |
| 知识来源 | 模型内部先验 | **动态检索外部知识** | 内部先验 + 训练 |
| 核心突破 | 语义-空间推理链 | OOD 概念实时接地 | 端到端自主创建 |
| 代表场景 | 通用生成/编辑 | 实时信息、新兴概念 | 复杂创意任务 |
| Benchmark | 无专属 | Mind-Bench (500) | VisGenBench (1.2K) |

## 在思维链/Agent 演进中的位置

编辑/生成领域的 Agent 化只有三种路：

1. **内部推理**（GoT）：模型自己思考，不依赖外部信息
2. **外部检索**（Mind-Brush）：模型主动获取缺失知识 → **这是最接近人类创作过程的**
3. **全流程自主**（VisionCreator）：从理解到规划到创建，一步到位

Mind-Brush 填补了最关键的一环：**当模型的知识不够时怎么办？** 答案是像人类一样去查资料。

## 在 Wiki 中的关联

- 来源：[[Wiki/Sources/Mind-Brush]]
- 主题：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 比较：[[Wiki/Comparisons/编辑方法能力演进]]
- 相关：[[Wiki/Entities/GoT]], [[Wiki/Entities/VisionCreator]]

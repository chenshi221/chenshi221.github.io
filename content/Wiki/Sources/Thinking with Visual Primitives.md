---
type: source
status: complete
created: '2026-05-07'
updated: '2026-05-07'
tags: [visual-reasoning, image-generation, primitives, multimodal, spatial-grounding, counting, topological-reasoning]
sources:
  - "[[Clippings/Thinking with Visual Primitives]]"
---

# Thinking with Visual Primitives

## 基本信息

- **标题**: Thinking with Visual Primitives
- **作者**: Ruijie Lu, Yiyang Ma, Xiaokang Chen, Lingxiao Luo, Zhiyu Wu, Zizheng Pan, Xingchao Liu, Yutong Lin, Hao Li, Wen Liu, Zhewen Hao, Xi Gao, Shaoheng Nie, Yixuan Wei, Zhenda Xie, Ting Chen, Gang Zeng
- **机构**: DeepSeek-AI（主导）、北京大学、清华大学
- **年份**: 2025
- **来源**: [PDF](https://www.k-a.in/Thinking_with_Visual_Primitives.pdf)
- **基座模型**: DeepSeek-V4-Flash（284B 总参数，13B 激活参数，MoE 架构）

## 核心论点

1. **Reference Gap 是比 Perception Gap 更根本的瓶颈**。现有 MLLM 通过高分辨率裁剪（如 Thinking with Images）解决"看不清"的问题，但即使感知完美，自然语言在连续视觉空间中仍无法提供精确、无歧义的指向，导致复杂空间推理中的逻辑崩溃。
2. **将空间标记（bounding box 和 point）提升为"最小思维单元"**。不同于仅将 grounding 作为后验验证，本文将视觉原语直接交错嵌入推理链中，实现"边指边想"（point-to-reason），类似人类用手指辅助计数和导航的认知过程。
3. **极致视觉 token 效率**。通过 3x3 空间压缩 + Compressed Sparse Attention（CSA），对 756x756 图像仅产生 81 个 KV cache 条目，整体压缩比达 7,056x，在远低于其他前沿模型的 token 预算下实现可比性能。
4. **专家分化 + 统一蒸馏的训练范式**。分别训练"box grounding 专家"和"point pointing 专家"，再通过 Unified RFT 和 On-Policy Distillation 合并为单一模型，避免模式冲突。
5. **在计数、空间推理和拓扑推理任务上达到前沿竞争力**。在 CountQA、Pixmo-Count、SpatialMQA、DS_Maze_Navigation 等基准上匹配或超越 GPT-5.4、Claude-Sonnet-4.6、Gemini-3-Flash。

## 关键技术方法

### 视觉原语定义

- **Bounding Box**: 捕捉目标的精确位置和尺度，归一化为 [0, 999] 的离散整数坐标，使用 `<|ref|>...<|/ref|><|box|>[[x1,y1,x2,y2],...]<|/box|>` 特殊 token 格式。
- **Point**: 用于抽象视觉引用（如运动轨迹、拓扑路径），使用 `<|point|>[[x1,y1],...]<|/point|>` 格式，不要求输出对象名称，以支持更抽象的概念表示。

### 视觉编码与压缩管线

1. DeepSeek-ViT（支持任意分辨率）提取 patch token（14x14 patch size）
2. 3x3 空间 token 压缩（每 9 个相邻 patch 合并为 1 个）
3. CSA 机制进一步 4x 压缩 KV cache
4. 以 756x756 输入为例：2,916 patch tokens -> 324 visual tokens -> 81 KV entries

### 大规模数据构建

- 从 HuggingFace 等平台爬取 97,984 个 box grounding 数据源
- **两阶段过滤**：语义审查（MLLM 驱动，消除无意义标签、私有实体、模糊缩写）-> 视觉-几何质量审查（缺失标注、截断偏移、巨型框）
- 最终获得 4,000 万+ 高质量样本
- 统一格式标准，预训练消耗数万亿多模态 token

### 冷启动数据设计（四大维度）

| 维度 | 核心思路 | 数据量 |
|------|----------|--------|
| **计数** | 粗粒度（batch grounding）+ 细粒度（逐个扫描验证），利用 GQA scene graph 生成 hard negative | ~10,000 |
| **空间推理与通用 VQA** | 自然场景（GQA）+ 合成场景（CLEVR），多跳推理，负样本增强 | ~9,000 |
| **迷宫导航** | DFS/Prim/Kruskal 算法生成，矩形/圆形/六角三种拓扑，不可解迷宫增强，难度分级（easy -> nightmare） | 460,000 |
| **路径追踪** | 贝尔赛曲线纠缠，交叉点消歧，均匀风格模式消除颜色捷径 | 125,000 |

### 后训练流水线

1. **Specialized SFT**: 70% 通用数据 + 30% 专项数据，分别训练 F_TwG（box）和 F_TwP（point）
2. **Specialized RL**: GRPO 算法，设计三类 RM（Format RM、Quality RM、Accuracy RM），仅需图像+问题+最终答案（不显式监督推理中的原语）
3. **Unified RFT**: 用专家模型 rollout 生成 RFT 数据（Normal-Level + 5% Easy-Level），从预训练 checkpoint 重新训练统一模型
4. **On-Policy Distillation**: 反向 KL 散度，全词表 logit 蒸馏，将两个专家能力合并到单一模型

### 奖励模型设计亮点

- **计数 RM**: 平滑指数衰减（相对误差），近正确预测轻惩罚
- **迷宫 RM**: 因果探索进度 + 探索完整度 + 穿墙惩罚 + 路径有效性 + 答案正确性（五维分解）
- **路径追踪 RM**: 双向轨迹精度（前向+反向）+ 端点精度 + 轨迹连续性惩罚 + 答案正确性

## 主要结果

- **计数**: CountQA EM 64.9（vs GPT-5.4 48.3, Gemini-3-Flash 66.1）; Pixmo-Count EM 89.2（最优）
- **空间推理**: SpatialMQA ACC 69.4（最优）; MIHench ACC 85.3（最优）; DS_Spatial_Reasoning ACC 98.7（最优）
- **拓扑推理**: DS_Maze_Navigation ACC 66.9（vs 次优 50.6）; DS_Path_Tracing ACC 56.7（vs 次优 46.5）——所有前沿模型在此类任务上表现均不佳
- **Token 效率**: 800x800 图像仅需约 90 个 KV cache 条目，远低于其他模型

## 局限性

1. **输入分辨率限制**：细粒度场景下视觉原语输出偶尔不够精确，可与解决 Perception Gap 的方法互补。
2. **需要显式触发词**：当前"thinking with visual primitives"能力依赖触发词激活，未来目标是让模型自主决定何时调用。
3. **拓扑推理泛化有限**：以 point 为原语解决复杂拓扑推理仍是难题，跨场景泛化能力有限。

## 与相关工作的关系

| 相关工作 | 关系 |
|----------|------|
| [[Wiki/Sources/GoT\|GoT (Graph of Thoughts)]] | GoT 将思维结构从链扩展为图；本文则将思维单元从纯语言扩展到视觉原语，二者都是对 CoT 范式的结构性增强，但维度不同 |
| [[Wiki/Sources/Mind-Brush\|Mind-Brush]] | Mind-Brush 关注图像编辑中的推理；本文聚焦纯视觉 QA 中的空间 grounding 推理 |
| [[Wiki/Sources/VisionCreator\|VisionCreator]] | VisionCreator 关注图像生成中的推理链；本文不涉及生成，而是通过在推理链中嵌入空间标记增强理解 |
| Thinking with Images | 前作通过高分辨率裁剪解决 Perception Gap；本文提出解决更深层的 Reference Gap |
| [[Wiki/Sources/GPT-4\|GPT-4]] / Claude / Gemini | 在计数和拓扑推理上匹配或超越这些前沿闭源模型，但拓扑推理维度所有模型仍有显著提升空间 |
| CLEVR / GQA | 作为合成和自然场景数据源，用于构建冷启动训练和评估数据 |

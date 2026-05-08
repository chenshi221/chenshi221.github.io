---
type: source
status: complete
created: '2026-05-07'
updated: '2026-05-07'
tags:
  - embedding
  - retrieval
  - reranker
  - multimodal
  - qwen
  - alibaba
  - vlm
sources:
  - "[[Clippings/Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for Multimodal Retrieval]]"
---
# Qwen3-VL-Embedding and Reranker

## 基本信息

- **标题**: Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking
- **作者**: Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, Jingren Zhou, Junyang Lin
- **机构**: Tongyi Lab, Alibaba Group
- **年份**: 2026
- **arXiv**: 2601.04720
- **模型尺寸**: 2B / 8B（Embedding 和 Reranker 各两种尺寸）
- **开源**: HuggingFace + ModelScope + GitHub (QwenLM/Qwen3-VL-Embedding)

## 核心论点

1. **统一多模态检索空间**: 首次在 Qwen3-VL 基础模型上构建统一的多模态 Embedding + Reranker 管线，支持文本、图像、文档图像、视频四种模态的跨模态检索
2. **SOTA 多模态 Embedding**: Qwen3-VL-Embedding-8B 在 MMEB-V2 上达到 77.8 分（2026 年 1 月排名第一），超越所有开源和闭源模型
3. **多阶段训练范式**: 从大规模对比预训练到排序模型蒸馏的渐进式训练，结合 Matryoshka 表示学习和量化感知训练
4. **跨模态 Reranker**: Cross-encoder 架构 + cross-attention 机制实现细粒度相关性估计，2B 模型即超越此前最优，8B 进一步提升 4.1 分
5. **多语言继承**: 继承 Qwen3-VL 的 30+ 语言能力，无需额外多语言训练

## 关键技术方法

### Embedding 模型
- **架构**: Bi-encoder，基于 Qwen3-VL backbone + causal attention
- **表示方式**: 输入末尾追加 `PAD` token，取其最后隐藏状态作为 dense vector
- **输入格式**: 系统消息传入指令（默认 "Represent the user's input"），用户消息传入多模态实例
- **Matryoshka 表示学习 (MRL)**: 支持灵活选择嵌入维度（如从 4096 降到更低维度），无需重新训练
- **量化感知训练**: 训练过程中集成量化策略，确保量化后性能稳健

### Reranker 模型
- **架构**: Cross-encoder，query 和 document 通过 cross-attention 深度交互
- **评分方式**: Pointwise ranking，计算模型预测 "yes" 的概率作为相关性得分
- **指令感知**: 支持任务特定的指令定制

### 训练流程
- **阶段 1 - 对比预训练**: 大规模多模态多任务相关性数据的对比学习
- **阶段 2 - 排序蒸馏**: 从 Reranker 模型蒸馏知识到 Embedding 模型
- **阶段 3 - 量化感知微调**: 确保量化后性能

### 模型规格
| 模型 | 尺寸 | 层 | 序列长度 | 嵌入维度 |
|------|------|-----|---------|---------|
| Embedding | 2B | 28 | 32K | 2048 |
| Embedding | 8B | 36 | 32K | 4096 |
| Reranker | 2B | 28 | 32K | - |
| Reranker | 8B | 36 | 32K | - |

## 主要结果

- **MMEB-V2**: 8B Embedding 达到 77.8（排名第一），2B 达到 72.3
- **MTEB Multilingual**: 8B 在纯文本任务上平均 67.9 分，高度竞争力
- **图文检索**: 在 Flickr30K、COCO 等标准基准上达到 SOTA
- **视觉文档检索**: 在 ViDoRe 等文档检索基准上表现优异
- **视频文本匹配**: 在视频检索任务上超越专用模型
- **Reranker 提升**: 8B Reranker 在多个任务上比 2B 提升 4.1 分

## 局限性

1. **长视频效率**: 32K token 上下文对超长视频仍有限制
2. **模型体积**: 8B 模型在实时检索场景下部署成本较高
3. **Reranker 延迟**: Cross-encoder 架构的推理延迟高于 Bi-encoder，不适合大规模初筛
4. **评测时间点**: MMEB-V2 排名为 2026 年 1 月，后续可能被超越

## 与相关工作的关系

| 相关工作 | 关系 |
|---------|------|
| [[Wiki/Sources/Magic-MM-Embedding]] | Magic-MM 压缩视觉 token 75% 提效，Qwen3-VL-Embedding 直接基于 VLM backbone |
| [[Wiki/Sources/ObjEmbed]] | ObjEmbed 专注物体级嵌入，Qwen3-VL-Embedding 覆盖全模态统一检索 |
| [[Wiki/Sources/RzenEmbed]] | RzenEmbed 四模态 Embedding + hardness-weighted loss，Qwen3-VL-Embedding 增加 Reranker 管线 |
| [[Wiki/Sources/SAIL-Embedding]] | SAIL 是字节工业级部署，Qwen3-VL 是阿里学术级 SOTA |
| [[Wiki/Sources/VLM2Vec-V2]] | VLM2Vec-V2 扩展到视频/文档但基于 2B，Qwen3-VL-Embedding 达到 8B SOTA |
| [[Wiki/Concepts/多模态 Embedding 模型]] | 从 CLIP 双塔到 VLM-based 统一架构的最新进展 |
| [[Wiki/Topics/多模态 Embedding 与检索]] | 补充了 Qwen 系列在 Embedding 方向的能力 |

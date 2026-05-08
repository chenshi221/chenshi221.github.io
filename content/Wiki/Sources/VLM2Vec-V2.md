---
type: source
status: complete
created: '2026-05-07'
updated: '2026-05-07'
tags: [embedding, multimodal, video, image, document, retrieval]
sources:
  - "[[Clippings/VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents]]"
---

# VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents

## 基本信息

- **作者**: Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang, Yuepeng Fu, Can Qin, Zeyuan Chen, Ran Xu, Caiming Xiong, Yingbo Zhou, Wenhu Chen, Semih Yavuz
- **机构**: Salesforce Research, UC Santa Barbara, University of Waterloo, Tsinghua University
- **年份**: 2025
- **论文**: [arXiv:2507.04590](https://arxiv.org/abs/2507.04590)
- **项目主页**: [https://tiger-ai-lab.github.io/VLM2Vec/](https://tiger-ai-lab.github.io/VLM2Vec/)
- **前作**: [[VLM2Vec]]（Jiang et al., 2024）

## 核心论点

现有的多模态嵌入模型（如 [[VLM2Vec]]、[[E5-V]]、[[GME]]）主要针对自然图像训练，在视频和视觉文档上的表现受限，无法满足真实场景（AI agent、多模态搜索推荐、[[RAG]]）的需求。VLM2Vec-V2 提出一个统一框架，将嵌入学习扩展到图像、视频和视觉文档三种视觉模态，在 78 个数据集上实现强性能。

## 关键技术方法

### 1. MMEB-V2 基准

在原有 [[MMEB]]（36 个图像任务）基础上扩展，新增 42 个任务，涵盖 5 类新 meta-task：

| 任务类型 | 数量 | 说明 |
|---------|------|------|
| 视频检索 (V-RET) | 5 | 文本描述检索视频 |
| 片段检索 (M-RET) | 3 | 文本/图像查询定位视频片段 |
| 视频分类 (V-CLS) | 5 | 视频帧序列分类 |
| 视频问答 (V-QA) | 5 | 视频多选问答 |
| 视觉文档检索 (VisDoc) | 24 | 自然语言查询检索 PDF/幻灯片等 |

共覆盖文本、图像、视频、视觉文档四种模态的 9 类 meta-task、78 个任务。

### 2. 统一嵌入模型

- **骨干网络**: [[Qwen2-VL]] 2B，支持动态分辨率、多模态旋转位置编码（M-RoPE）、2D/3D 统一编码
- **训练方式**: 对比学习，使用 InfoNCE 损失 + batch 内负样本 + 硬负样本
- **输入格式**: 统一的指令条件格式，query 端拼接 `[VISUAL_TOKEN]` + 任务指令 + 原始查询，target 端可选加指令引导
- **参数高效训练**: LoRA（rank=16, alpha=32），使用 GradCache 支持大 batch（1024）

### 3. 数据采样策略

- **动态 batch 混合**: 按预定义采样权重表从不同数据源动态抽取，防止过拟合单一模态
- **交错子 batch 策略**: 将大 batch（1024）拆分为小子 batch（64），每个子 batch 内同源采样以增加对比难度，子 batch 间交错以保持多样性

### 4. 训练数据

- 视频：LLaVA-Hound（300k 视频-描述对 + 240k 视频 QA）
- 视觉文档：ViDoRe 训练集（118k）+ VisRAG 合成/域内数据（362k）
- 图像：MMEB-train（覆盖 QA、分类、检索、grounding 等任务）

## 主要结果

- VLM2Vec-V2（2B 参数）在 78 个数据集上总体平均得分 **58.0**，超过 GME-7B（57.8）、LamRA-Qwen2.5-7B（47.4）、VLM2Vec-7B（52.3）等基线
- **图像任务**（36 个）：平均 Hit@1 = 64.9，与 VLM2Vec-7B（65.5）持平，大幅超过其他 2B 模型
- **视频任务**（18 个）：平均 Hit@1 = 34.6，虽仅用少量视频数据训练，仍具竞争力
- **视觉文档检索**（24 个）：平均 NDCG@5 = 65.4，显著超越所有 VLM2Vec 变体，但仍落后于专门优化的 [[ColPali]]（71.0）

### 消融实验关键发现

1. **多模态训练收益大**: 三模态联合训练（Image+Video+VisDoc）在文档任务和总体得分上最优，表明跨模态泛化有增益
2. **子 batch 大小影响**: VisDoc 和 Video 随子 batch 增大持续提升；Image 在 IB=64 时最优（倒 U 型曲线）
3. **LoRA rank**: rank=16 整体最优，rank=32 无额外增益
4. **训练步数**: 5K 步时 VisDoc 和 Video 仍未饱和，进一步训练可能继续提升

## 局限性

- 视频仅用 8 个均匀采样帧表示，可能丢失细粒度时序信息
- 视觉文档检索仍明显落后于专门模型 ColPali（65.4 vs 71.0），说明通用模型在专业任务上仍有差距
- 视频任务性能（34.6）与图像任务（64.9）差距较大，视频嵌入学习仍是挑战
- 训练数据中视频和文档来源相对有限，数据规模可能制约泛化能力
- 未探索更大规模骨干（如 7B）是否能进一步缩小模态间差距

## 与相关工作的关系

- **VLM2Vec（前作）**: 同一团队，VLM2Vec-V2 在其基础上扩展到视频和视觉文档模态，使用相同的指令条件对比学习框架
- **[[GME]]**: 同样基于 Qwen2-VL 微调的多模态嵌入模型，但主要面向图像和图像-文本任务；VLM2Vec-V2 在图像任务上与 GME-7B 持平，在视频和文档上超越
- **[[ColPali]]**: 视觉文档检索专用模型，使用 late interaction 机制；VLM2Vec-V2 在文档检索上仍落后，但在多模态统一性上更有优势
- **LamRA**: 基于大模型的通用检索助手，但在视觉文档检索上表现很差（23.9），缺乏文档模态训练
- **MMEB-V2 vs MMEB**: 前作基准仅覆盖自然图像，MMEB-V2 新增视频和文档评估，填补了嵌入模型在多样化视觉模态上的评估空白

---
title: 'Seedream 2.0: A Native Chinese-English Bilingual Image Generation Foundation Model'
authors:
  - 字节跳动 Seed 团队
institutions: ByteDance Seed
aliases:
  - Seedream 2.0
date: '2026-04-30'
publish_date: 2025
venue: 技术报告
tags:
  - 论文
  - 图像生成
  - 双语生成
  - 中英文
  - DiT
  - LLM文本编码器
  - RLHF
url: 'https://arxiv.org/abs/2504.12345'
code: 未开源
rating: ⭐⭐⭐⭐
status: Read
---
**一句话总结**：Seedream 2.0 是字节跳动推出的原生中英双语图像生成基础模型，通过 LLM 文本编码器（特别是 Glyph-Aligned ByT5）、高质量数据系统和 RLHF 对齐，解决了中文文本渲染和双语文化理解的难题，在中文图像生成领域建立了新的 SOTA。

![](https://arxiv.org/html/2503.07703v1/figures/overall_cn_en.png)

*Figure 1: Seedream 2.0 的中英双语生成效果概览——模型支持中文和英文 prompt，能准确渲染复杂中文汉字，并理解中国特有的文化概念和视觉元素。*

---

## Intro

### Motivation

现有的文本到图像生成模型（如 DALL-E 3、SD3、FLUX）主要由英语社区开发，对**中文文本渲染**（在图中生成正确的中文字符）和中文文化概念的理解能力严重不足。此外，这些模型大多使用 CLIP 或 T5 作为文本编码器，而 CLIP 对中文的支持有限，T5 虽然多语言但参数量大。

字节跳动 Seed 团队的目标是构建一个**原生中英双语**的图像生成基础模型，能够：
1. 准确渲染中文和英文字符
2. 理解中国特有的文化概念和视觉元素
3. 在高 aesthetic quality 上不输国际 SOTA 模型

### 核心主张

1. **LLM 文本编码器**（特别是 Glyph-Aligned ByT5）是中英双语生成的关键
2. **高质量数据系统**：经过多级过滤的数据 pipeline 比模型架构更重要
3. **RLHF 对齐**：在图像生成上使用人类反馈强化学习，显著提升审美质量和指令遵循

### 贡献

1. 提出 Glyph-Aligned ByT5（GA-ByT5），一种专为图像生成优化的字节级文本编码器
2. 构建了大规模中英双语图文数据 pipeline（多级过滤、美学筛选、合成数据增强）
3. 将 RLHF 应用于图像生成的对齐过程，提升审美和指令遵循
4. 在中文文本渲染、中国文化理解和通用 T2I 质量上达到 SOTA

---

## Method 核心方法

Seedream 2.0 的方法论覆盖了从数据到部署的完整链路：数据系统 → 模型架构 → 预训练 → 多阶段后训练（CT/SFT/RLHF/PE/Refiner）。以下按 pipeline 顺序展开。

### 1. 数据系统：四组件+三级过滤+主动学习

Seedream 2.0 的数据系统是支撑其中文能力的关键基础设施。

#### 1.1 数据组成（四组件）

| 组件 | 内容 | 目的 |
|------|------|------|
| **高质量数据** | 清晰度、美学、来源分布均优的数据 | 核心质量基座 |
| **分布维护数据** | 按来源降采样+层级聚类采样 | 保持数据多样性，避免过拟合特定源 |
| **知识注入数据** | 中文特有概念（动植物、菜系、建筑、民俗）+多模态检索引擎 | 弥补中文文化知识的天然不足 |
| **定向补充数据** | 动作类、反事实类（如"脖子上顶气球的男子"） | 针对性补齐T2I短板 |

![](https://arxiv.org/html/2503.07703v1/x1.png)

*Figure 3: Seedream 2.0 预训练数据系统——四组件pipeline确保数据的高质量、多样性、知识覆盖和针对性补充。*

#### 1.2 三级数据清洗

- **第一级（通用质量评估）**：清晰度、运动模糊、水印/贴纸/Logo 检测、OCR 检测——不达标即删除
- **第二级（精细化评估）**：专业美学评分、特征嵌入提取、去重、多层级聚类（每个样本分配语义类别标签）
- **第三级（Caption 标注/重标注）**：分层 caption 策略——高层级数据获得更丰富的描述（短 caption + 长 caption + 专用 caption）

#### 1.3 主动学习引擎

迭代式改进图像分类器：用当前分类器筛选数据 → 人工标注边界案例 → 重新训练分类器 → 循环。确保过滤标准与人类判断持续对齐。

#### 1.4 三级 Caption 系统

| Caption 类型 | 描述内容 | 用途 |
|-------------|---------|------|
| **短 Caption** | 图像核心内容和知识 | 基础图文对齐 |
| **长 Caption** | 尽可能多的细节，含适当推理和想象 | 复杂场景理解 |
| **专用 Caption** | 艺术（风格/色彩/构图/光影）、文字（OCR信息）、超现实（想象视角） | 特定能力的精细控制 |

#### 1.5 文字渲染数据 Pipeline

为中文文字渲染能力专门构建：OCR 检测文本区域 → 裁剪去除水印 → 过滤低质量文本框 → 重标注模型生成描述 → 精细化描述 → 形成高质量 image-caption 对。

### 2. 模型架构：MMDiT + GA-ByT5 + 自研 LLM Text Encoder

![](https://arxiv.org/html/2503.07703v1/figures/arch.png)

*Figure 2: Seedream 2.0 模型架构——输入图像经 VAE 编码为 latent token，与 LLM 文本编码器和 GA-ByT5 的字形特征拼接后送入 MMDiT 块（含 QK-Norm、2D RoPE），最终解码生成图像。*

#### 2.1 MMDiT (Multi-Modal Diffusion Transformer)

遵循 SD3 的 MMDiT 设计原则，但做了若干关键适配：

- **单自注意力层**：每个 Transformer 块中，图像 token 和文本 token 通过同一自注意力层联合处理（跨模态信息混合）
- **独立 MLP**：图像和文本模态使用各自独立的 MLP（承认模态差异，但通过注意力实现交互）
- **自适应 Layer Norm (adaLN)**：时间步条件通过 adaLN 注入每个注意力和 MLP 层
- **QK-Norm**：在 Q 和 K 投影后添加 Layer Norm，提升训练稳定性
- **FSDP (Fully Sharded Data Parallel)**：分布式训练策略

#### 2.2 位置编码：Scaling RoPE

- 文本 token：使用学习的位置嵌入
- 图像 token：使用 **2D RoPE**（在 H 和 W 维度分别编码）
- **Scaling RoPE 创新**：根据图像分辨率配置不同的缩放因子，使图像中心区域的 patch 在不同分辨率间共享相似的位置 ID。这使得模型在推理时能泛化到训练时未见过的长宽比和分辨率

#### 2.3 文本编码器：自研 LLM + GA-ByT5 双编码

**为什么不用 CLIP/T5？** CLIP 中文能力弱，T5 虽多语言但参数量大且缺乏 decoder-only LLM 的强大语义理解。然而 decoder-only LLM 的文本嵌入与扩散模型中的图像表示之间存在巨大的特征分布差异，直接使用会导致训练不稳定。

**解决方案——自研多语言 LLM Text Encoder**：
- 在图文对数据上专门微调 decoder-only LLM，使其文本嵌入与图像表示空间对齐
- 收集大量中文风格化描述和专业词汇数据强化训练
- 结果：在长文本理解、复杂指令遵循、文化细微差别捕捉上全面超越 CLIP/T5

**GA-ByT5（Glyph-Aligned ByT5）——文字渲染的专用编码器**：

ByT5 是字节级（byte-level）T5，天然处理每个字符的 UTF-8 字节，对中文等复杂字符的字形理解远优于 subword tokenizer。

Glyph Alignment 训练细节：
- 将渲染文字的 LLM 特征和 ByT5 特征拼接输入 DiT
- MLP 投影层将 ByT5 嵌入映射到与 LLM 文本编码器对齐的空间
- **仅用文本特征作为条件**（不使用 OCR 图像特征），保持与标准 T2I 相同的训练/推理流程
- 文字的字体、颜色、大小、位置等属性通过重标注模型描述，由 LLM 编码器处理——实现端到端的文字渲染学习，无需预设文本框布局

### 3. 预训练策略

- 自研 VAE 编码图像 → latent 表示 → patchify 为图像 token
- 图像 token + 文本 token（LLM）+ 字形 token（GA-ByT5）拼接 → MMDiT
- 使用 Flow Matching 训练目标
- 混合分辨率训练（通过 Scaling RoPE 支持）

### 4. 多阶段后训练

![](https://arxiv.org/html/2503.07703v1/figures/pipeline.png)

*Figure 9: Seedream 2.0 训练和推理全流程——包含预训练、继续训练(CT)、SFT、RLHF、Prompt Engineering(PE)和 Refiner 超分六个阶段。*

#### 4.1 继续训练（CT）

**数据**：从预训练数据中通过专用 IQA（图像质量评估）模型自动筛选高质量子集 + 数百万张人工精选的艺术/摄影/设计图像。

**VMix 美学增强**：对每张图像标注多维美学标签（色彩、光线、纹理、构图），作为 CT 训练的附加条件。这使模型能直接学习细粒度的美学特征，生成图像与真实高美感图像的差距显著缩小。

#### 4.2 监督微调（SFT）

- 使用少量精选高艺术品质图像 + 专门训练的"美感描述"caption 模型
- **负样本策略**：混入模型自身生成的图像，标注为"负样本"——模型学会区分真实与生成，提升图像自然度
- **数据重采样算法**：平衡美学提升和图文对齐——纯美学数据会损害对齐能力，通过调整采样比例实现双目标优化

#### 4.3 RLHF——人类反馈对齐

Seedream 2.0 首次将 RLHF 系统性地引入扩散模型后训练：

**偏好数据**：
- Prompt 系统：100 万条多维度 prompt（来自训练 caption 和用户输入）
- 跨版本、跨模型标注管线：收集不同模型输出，标注偏好对（A > B）
- 多维融合标注：图文匹配 + 文字渲染 + 美学——单次标注覆盖多维度，确保 RLHF 各维度 Pareto 最优

**三个专用奖励模型（RM）**：

| RM | 基座 | 训练目标 | 触发条件 |
|----|------|---------|---------|
| 图文对齐 RM | 中英双语 CLIP | Ranking Loss（直接用 CLIP 输出作为 reward） | 始终 |
| 美学 RM | 中英双语 CLIP | Ranking Loss | 始终 |
| 文字渲染 RM | 中英双语 CLIP | Ranking Loss | 仅 prompt 含文字渲染标签时 |

**反馈学习算法**：类似 REFL（REward Fine-tuning for Language models）范式——直接通过优化多个 RM 的输出分数来精调扩散模型。关键训练技巧：
- 精心调整学习率
- 选择合适去噪时间步
- 权重 EMA
- **DiT 和 LLM 文本编码器联合微调**——这是性能大幅提升的关键策略

**迭代精炼**：RLHF 采用迭代优化——用当前 RM 优化模型 → 在新模型上标注偏好 → 训练 bad-case-aware RM → 用新 RM 继续优化 → 循环。此策略不断抬高 RLHF 的性能上限。

#### 4.4 Prompt Engineering（PE）

用户 prompt 通常简短，与训练时的高质量 caption 存在分布差距。PE 通过微调 LLM 自动改写 prompt：

- **SFT 阶段**：构建 prompt 对数据集 D={<u, r>}（u=用户输入，r=改写后高质量 prompt）。两种构建方式：(u→r) 人工改写直到满意；(r→u) 精选高质量 caption 后用人造退化
- **PE RLHF**：使用 SimPO（Simple Preference Optimization）进一步对齐 PE 模型。用户在扩散模型的生成结果上标注高低质量图像对，对应 prompt 对用于训练 PE

PE 带来 30% 美学提升和 5% 图文对齐提升，同时大幅增加生成多样性。

#### 4.5 Refiner（超分辨率）

独立的超分模型将基础模型的输出上采样到更高分辨率，同时修复细微的结构错误（如手指异常）。

---

## 实验/评估/结果

### 中文文本渲染

在中文 OCR 准确性上大幅超越 FLUX、SD3 等英语为主模型。GA-ByT5 对复杂中文汉字的渲染准确率显著优于 CLIP 文本编码器，支持中文艺术字、书法风格的生成。

### 通用 T2I 质量

在 GenEval、HPS v2 等通用 benchmark 上与 FLUX、SD3 持平或更优。Aesthetic quality 通过 RLHF 对齐后，人类偏好胜率显著提升。支持中英文混合 prompt。

### 各后训练阶段效果对比

| 阶段 | 图文对齐 | 美学质量 | 结构正确性 | 文字渲染 |
|------|---------|---------|-----------|---------|
| 预训练基座 | 基准 | 基准 | 基准 | 基准 |
| + CT | → | ↑↑ | → | → |
| + SFT | → | ↑↑↑ | → | → |
| + RLHF | ↑ | ↑ | ↑ | ↑↑ |
| + PE | ↑ | ↑↑ | → | ↑ |

*CT 和 SFT 主要提升美学；RLHF 全面提升各维度；PE 进一步增强美学的多样性和质量。*

### RLHF 训练过程监控

三个奖励模型（图文对齐 RM、美学 RM、文字渲染 RM）的 reward 曲线在整个 RLHF 对齐过程中均呈现**稳定且一致的上升趋势**，验证了多 RM 联合优化的可行性和稳定性。

### 消融实验

| 消融项 | 对比方案 | 结果 |
|--------|---------|------|
| **GA-ByT5 vs CLIP** | CLIP text encoder | GA-ByT5 在文字渲染上显著优于 CLIP，通用质量持平 |
| **RLHF 增益** | 无 RLHF | 人类偏好评估中提升约 10 个百分点胜率 |
| **数据质量 vs 数据量** | 同等规模未过滤数据 | 高质量过滤数据效果显著优于未过滤数据 |
| **LLM Text Encoder vs T5** | T5 encoder | LLM encoder 在中文长文本理解和指令遵循上大幅领先 |
| **VMix 美学增强** | 无 VMix | VMix 在 CT 阶段进一步提升了美学表现

---

## 结论

Seedream 2.0 通过 LLM 级别的文本编码器（GA-ByT5）、系统化的数据工程和 RLHF 对齐，构建了首个原生中英双语的 SOTA 图像生成模型。其核心贡献——Glyph-Aligned ByT5 和 RLHF 对齐——为多语言图像生成提供了可复制的技术路径。

---

## 思考

### 优点

1. **GA-ByT5 是正确方向**：在图像生成中使用字节级编码器解决文字渲染问题，比 CLIP tokenizer 的 patch 方案更根本。这个选择体现了"字符渲染是字形问题，不是语义问题"的深刻理解。

2. **数据哲学的胜利**：数据 pipeline 的精细设计（多级过滤、recaptioning、合成增强）再次印证"高质量数据比参数更多更重要"。

3. **RLHF 的跨界应用**：将 LLM 领域的 RLHF 引入图像生成是大胆而有成效的尝试。实验证明，人类偏好的信号确实能指导模型学到更好的审美和指令遵循。

4. **工业实用性**：作为产品级模型（豆包图像生成），Seedream 2.0 经受住了大规模用户的考验。

### 缺点与待解决问题

1. **闭源性**：模型权重和代码未开源，社区无法复现和验证其声明。

2. **GA-ByT5 的计算开销**：ByT5 比 CLIP text encoder 参数量更大、推理更慢，对部署的影响在报告中体现不足。

3. **RLHF 的局限**：RLHF 对齐可能存在 reward hacking——模型学会迎合 reward model 的偏好而非真正的人类偏好。论文未讨论 reward model 的泛化性和 labeler 偏差。

4. **中文文化理解的深度**：虽然展示了中国风元素的生成能力，但文化理解的评估缺少系统化的 benchmark（如对不同地域、历史时期、子文化的精确识别和生成）。

### 与已有 Wiki 的连接

- 关联概念：[[DiT]]、[[LLM 文本编码器]]、[[RLHF]]、[[中文图像生成]]、[[Glyph Rendering]]
- 关联实体：[[FLUX]]、[[DALL-E 3]]、[[SD3]]、[[ByT5]]
- 关联论文：[[AI阅读笔记/Seedream 3.0]]、[[AI阅读笔记/Seedream 4.0]]
- 关联比较：[[Wiki/Comparisons/图像生成模型的中文能力对比]]

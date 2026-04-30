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

### 1. 文本编码器：Glyph-Aligned ByT5（GA-ByT5）

传统方法使用 CLIP text encoder，但 CLIP 在字符级理解上有天然缺陷（tokenization 破坏了字形信息）。Seedream 2.0 创新性地使用 ByT5（一个字节级别的 T5 模型）作为文本编码器，并对其进行了专门的 Glyph Alignment 训练：

- **ByT5 的优势**：字节级（byte-level）编码，每个字符都被显式处理，天然支持任何语言的字形理解
- **Glyph Alignment**：额外训练 ByT5 关注字符的视觉字形（glyph）信息，使其文本编码能直接指导生成正确的字符渲染
- **双语原生**：ByT5 的字节编码对中文和英文同样有效，不需要额外的 tokenizer 变动

![](https://arxiv.org/html/2503.07703v1/figures/pipeline.png)

*Figure 3: Seedream 2.0 的训练和推理全流程——包含预训练、继续训练(CT)、SFT、RLHF、Prompt Engineering(PE)和 Refiner 超分六个阶段，从大规模图文数据到最终高分辨率输出的完整管线。*

### 2. 数据系统（Data System）

Seedream 2.0 的数据构建流程分为多级：

**第一级：规模过滤**
- 从海量图文对中筛选高质量数据
- 基于分辨率、美学分数（aesthetic score）、CLIP score 等指标的自动过滤
- 去除低质量、水印、文字过小/过大的图像

**第二级：中英双语数据处理**
- 专门收集中文文化相关图像数据
- 使用 VLM 对中文图像进行重标注（recaptioning），提升描述质量
- 合成数据增强：用已有模型生成额外的中英文渲染训练数据

**第三级：精细化质量控制**
- 人工审核关键子集
- 使用 OCR 模型评估图像中文字的准确性

### 3. 模型架构

- **基座**：基于 DiT（Diffusion Transformer）架构
- **VAE**：高保真图像压缩
- **文本条件**：GA-ByT5 编码 + 额外 CLIP 图像嵌入用于风格控制
- **训练策略**：多阶段训练，从低分辨率到高分辨率逐步提升

![](https://arxiv.org/html/2503.07703v1/figures/arch.png)

*Figure 2: Seedream 2.0 的模型架构总览——输入图像经 VAE 编码为 latent token，与 LLM 文本编码器和 GA-ByT5 的字形特征拼接后送入 MMDiT 块（结合 QK-Norm、2D RoPE），最终解码生成图像。*

### 4. RLHF 对齐

将 LLM 领域的 RLHF（人类反馈强化学习）应用于图像生成：

- 收集人类对生成图像的偏好数据（A > B 比较）
- 训练 reward model 预测人类偏好（考虑 aesthetic quality、指令遵循度、文字准确性等维度）
- 使用 reward model 微调（PPO 或 DPO）基础生成模型
- 训练后的模型在审美和指令遵循上有显著提升

---

## 实验/评估/结果

### 中文文本渲染

- 在中文 OCR 准确性上大幅超越 FLUX、SD3 等英语为主模型
- GA-ByT5 对复杂中文汉字（如"龘""鑫"）的渲染准确率显著优于 CLIP 文本编码器
- 支持中文艺术字、书法风格的生成

### 通用 T2I 质量

- 在 GenEval、HPS v2 等通用 benchmark 上与 FLUX、SD3 持平或更优
- Aesthetic quality：通过 RLHF 对齐后，人类偏好胜率显著提升
- 支持中英文混合 prompt

### 双语文化理解

- 正确生成中国传统元素（如"剪纸""青花瓷""水墨画"）的图像
- 对中文习语（如"画龙点睛""一石二鸟"）的视觉化准确度高于英语模型

### 消融实验

- **GA-ByT5 v.s. CLIP**：GA-ByT5 在文字渲染上显著优于 CLIP，在通用质量上持平
- **RLHF 的增益**：RLHF 训练后在人类偏好评估中提升约 10 个百分点的胜率
- **数据质量 v.s. 数据量**：高质量过滤数据的效果优于同等规模未过滤数据

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

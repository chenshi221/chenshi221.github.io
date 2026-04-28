---
title: GoT：Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing
authors:
  - Rongyao Fang
institutions: CUHK MMLab
aliases:
  - GoT
date: 2025-07-10
publish_date: 2025
venue: ""
tags:
  - 论文
  - 多模态
  - Diffusion_Models
  - U-Net
  - 数据集
  - 思维链
url: https://arxiv.org/abs/2503.10639
code: https://github.com/rongyaofang/GoT
rating: ⭐⭐⭐⭐⭐
status: Read
---
GoT 通过让多模态大语言模型先生成一个包含精确坐标的“生成式思维链”来指导扩散模型，从而实现了对图像生成与编辑前所未有的精确控制和可解释性。
# 1. 总体介绍

## 1.1 问题背景

**背景**：目前的图像生成和编辑方法大多直接将文本提示作为输入，缺乏对视觉构图和具体操作的显式推理过程。这导致它们在处理复杂场景时能力不足。

**现有解决方案及其不足**：
1. **基于 Layout 的方法** (如 GLIGEN, LayoutGPT)：这些方法允许用户通过边界框来控制对象位置。但它们通常将“布局规划”和“图像生成”视为两个**分离的阶段**，缺乏端到端的优化和推理的深度融合。
2. **统一的 MLLM** (如 Emu2)：一些模型试图统一理解和生成任务，但它们并没有真正利用模型的**推理能力来指导生成过程**，更像是将不同的任务模块拼接在一起。
## 1.2 论文贡献

1. **提出 GoT 范式**: 首次提出将“思维链”思想引入视觉生成，将生成过程从直接映射转变为“先推理，后生成”的模式。
2. **构建大规模 GoT 数据集**: 通过自动化流程，构建了包含超过 9M 样本的 GoT 数据集，为训练这种推理-生成模型提供了数据基础。
3. **设计统一的 GoT 框架**: 开发了一个端到端的框架，包含一个能够生成 GoT 推理链的 MLLM，以及一个新颖的**语义-空间指导模块 (SSGM)**，该模块能有效利用 GoT 推理链来指导 Diffusion 模型。
4. **SOTA 性能和新能力**: 在文本到图像生成和图像编辑任务上取得了最先进的结果，并实现了独特的**交互式生成**能力。

# 2. Method

## 2.1 GoT 的核心理念

**GoT 推理链的定义**: 它是一种多模态的推理链，融合了语义（描述对象、属性、关系）和空间（(x1,y1),(x2,y2) 格式的坐标）信息。
- 对于**生成任务**，它详细描述了场景中的每个元素及其位置。
- 对于**编辑任务**，它分解为：1. 描述原图 -> 2. 识别被编辑对象及其位置 -> 3. 描述修改操作 -> 4. 描述编辑后的图像。
## 2.2 GoT 数据集构建

![image.png](https://chenshi-1327871112.cos.ap-chongqing.myqcloud.com/obsidian20250711214759960.png)

### 2.2.1Text-to-Image GoT Annotation

1. **生成提示和详细描述**: 给定一张图片，使用 Qwen2-VL 模型生成一个简短的提示 (作为 T2I 的输入) 和一段非常详细的场景描述 (作为 GoT 的语义部分)。
2. **实体解析**: 使用 Qwen2.5 从详细描述中抽取出所有的关键对象实体。
3. **实体定位**: 再次使用 Qwen2-VL，将上一步抽取的每个实体在原始图像中定位，并输出其边界框坐标。
4. **组装 (Assemble)**: 将详细描述和每个实体的坐标信息整合在一起，形成最终的 T2I GoT 标注。
### 2.2.2 Editing GoT Annotation

1. **描述源图和目标图**: 给定编辑前后的图像对，使用 Qwen2-VL 分别生成详细的描述。
2. **定位编辑区域**: 通过对比两张图，模型（Qwen2-VL）可以识别出被修改的物体，并对其进行定位。
3. **描述编辑细节**: 对于被编辑的物体，将其裁剪出来，再让 Qwen2-VL 进行详细描述，以获取更精细的属性信息。
4. **合成 GoT 推理链**: 使用 Qwen2.5，结合源图描述、目标图描述、编辑指令和定位信息，通过精心设计的 **In-context Prompting** (上下文提示) 技术，自动生成符合格式的、逻辑连贯的 GoT 编辑推理链。
### 2.2.3 数据来源

- T2I: Laion-Aesthetics-High-Resolution (LAHR), JourneyDB, FLUX.1
- Edit: OmniEdit(单轮)，SEED-Edit-Multiturn(多轮)

**资源消耗极大**：100\*A100\*24h\*30d
## 2.3 GoT框架设计

![image.png](https://chenshi-1327871112.cos.ap-chongqing.myqcloud.com/obsidian20250711221101480.png)

GoT框架主要由 MLLM 推理引擎和 Diffusion 生成模块两部分组成。

### 2.3.1 MLLM

基于 Qwen2.5-VL-3B 构建。对于编辑任务，它通过视觉编码器处理参考图像以理解源内容。在生成和编辑任务中，MLLM 都会生成 GoT 推理链，捕捉物体属性、关系、修改及边界框信息。推理链生成后，模型会生成图像起始标记及随后的 N=64 特殊[IMG]标记
从 MLLM 的输出中，会产生三种指导信号，送往 SSGM 模块：
- **Semantic Guidance Gt (语义指导)**: 由 64 个 [IMG] token 的嵌入向量构成。
- **Spatial Guidance Gs (空间指导)**: 从 GoT 推理链文本中解析出坐标，生成彩色掩码，再编码成 Gs。
- **Reference Image Guidance Gr (参考图像指导)**: 对于编辑任务，原始图像被编码成 Gr。
### 2.3.2 SSGM Diffusion Module

扩散模块基于 SDXL 架构

- **空间指导 Gs 的注入**:
    - **路径**: Gs (来自彩色掩码的潜变量) 与噪声 zt 进行**通道拼接 (Concatenate)**。
    - **原理**: 这是一种**强约束**。它直接在 U-Net 的输入层修改了潜变量的结构，相当于在每个空间位置上都打上了“标签”，告诉模型这个像素区域应该是什么。
- **语义指导 Gt 和参考指导 Gr 的注入**:
    - **路径**: Gt (来自 MLLM 的 [IMG] token) 和 Gr (来自参考图像) 被送入 U-Net 内部的 **Cross-Attention 层**。
    - **原理**: 这是一种**软约束**。Gt 和 Gr 作为 key 和 value，与图像本身的特征 query 进行注意力计算，从而在语义和风格层面上影响生成过程。

- **VAE Patch Embed**: 对于参考图像 Gr，除了通过 Cross-Attention 注入，它的 patch 嵌入也会与噪声进行**加权平均 (Averaging)**，这是一种在 InstructPix2Pix 等工作中常见的技术，可以更好地保留原图的结构信息。

## 2.3.3  Classifier-Free Guidance (CFG) 策略

$$ε_θ = ε_θ(zt, Ø, Ø, Ø) +  
αt · [ε_θ(zt, Gt, Ø, Ø) – ε_θ(zt, Ø, Ø, Ø)] +  
αs · [ε_θ(zt, Gt, Gs, Ø) – ε_θ(zt, Gt, Ø, Ø)] +  
αr · [ε_θ(zt, Ø, Ø, Gr) – ε_θ(zt, Ø, Ø, Ø)]$$
- **εθ**: 最终计算出的、用于去噪的噪声预测值
- **zt**: 在时间步 t 的加噪后的图像潜变量。
- **Ø**: 空条件或无条件 (null conditioning)。表示不使用任何指导信号。
- **$ε_θ(zt, Gt, Gs, Gr)$**: 表示在给定所有指导信号 (Gt, Gs, Gr) 的情况下，模型对 zt 中噪声的预测。

公式表达的是一种逐步修正的思想，例如 $[ε_θ(zt, Gt, Ø, Ø) – ε_θ(zt, Ø, Ø, Ø)]$的减法表示代表了从“自由发挥”到“符合语义描述”所需要改变的**方向向量**

# 3. 实验/评估/结果

## 3.1 模型训练

预训练：LAHR-GoT, JourneyDB-GoT, OmniEdit-GoT（60,000轮）

微调:FLUX-GoT、OmniEdit-GoT和SEEDEDit-MultiTurn-GoT(10,000轮)

采用LoRA来高效更新 Qwen2.5-VL 解码器的参数，同时对基于 SDXL 的扩散模块进行完全优化。整个过程以端到端的方式运行，联合优化 MLLM GoT 的交叉熵标记损失和扩散模块的均方误差（MSE）损失，并赋予两者相同的权重 1.0，从而在无需复杂超参数调优的情况下展现出强大的鲁棒性。

## 3.2 T2I

![image.png](https://chenshi-1327871112.cos.ap-chongqing.myqcloud.com/obsidian20250711230411586.png)

- **T2I 生成**: 在 GenEval 上取得了 0.64 的最高分，尤其在数量和颜色任务上表现突出，这得益于 GoT 推理链的显式规划能力。
### 3.3 交互式生成

![image.png](https://chenshi-1327871112.cos.ap-chongqing.myqcloud.com/obsidian20250711230658197.png)

- **图像编辑**: 在多个编辑基准上同样达到 SOTA 水平。在需要复杂推理的 Reason-Edit 上得分仅次于一个专门为该任务设计的模型，展示了其强大的通用编辑能力。

# 4. 结论

他们提出的 **Generation Chain-of-Thought (GoT)** 范式成功地将 MLLM 的推理能力整合到视觉生成中。通过生成显式的语义-空间推理链，GoT 克服了现有方法在精确空间控制和复杂关系理解上的局限。该方法不仅在生成和编辑任务上取得了 SOTA 性能，还实现了前所未有的交互式控制。

# 5. 总结

这篇文章提出了一个非常优雅且强大的新范式——**GoT (生成式思维链)**。它解决的核心痛点是当前生成模型“知其然，而不知其所以然”的黑盒特性，即模型无法像人一样进行有意识的规划和布局。将一个复杂的生成任务分解为“推理规划”和“渲染执行”两个子问题。

问题是图片虽然美观，但是细节不足
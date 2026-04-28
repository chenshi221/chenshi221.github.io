---
title: OminiControl：Minimal and Universal Control for Diffusion Transformer
authors:
  - Zhenxiong Tan
institutions: NUS
aliases:
  - OminiControl
date: 2025-07-10
publish_date: 2025
venue: ICCV
tags:
  - 论文
  - 多模态
  - Diffusion_Models
  - DiT
  - 数据集
url: http://arxiv.org/abs/2411.15098
code: https://github.com/Yuanshi9815/OminiControl
rating: ⭐⭐⭐⭐
status: Read
---
**一句话总结** ：OminiControl 通过使用使用**参数复用**策略，将条件图像和噪声图像的 token 拼接成一个统一序列，并利用自适应位置编码和注意力偏置，以极低的参数开销在 Diffusion Transformer 上实现了对空间对齐和非空间对齐任务的通用、灵活且精确的控制。
# 1. 总体介绍

## 1.1 问题背景

现有控制方法的局限性：
- **开销大：** 像 ControlNet 这样的方法需要复制整个去噪网络作为控制分支，导致参数量和计算开销巨大。
- **任务偏向性：** 无法兼顾[[空间对齐]]与[[非空间对齐]]
- **与DiT不兼容**：现在主流方法是为U-Net设计，无法迁移到DiT

## 1.2 论文主要贡献

1. 提出了**OminiControl**框架：
	- 使用**参数复用**策略实现最小化框架，仅增加0.1%参数量
	- 将条件标记与含噪图像标记直接拼接为统一序列
	- 采用动态定位策略，根据任务类型自适应分配位置索引
	- 们设计了灵活的注意力偏置机制，可在推理阶段精确调节条件强度
2. 构建并开源 Subjects200K 数据集，包含超过 20 万张身份一致性图像

# 2. 背景知识

![[DiT]]

# 3. Method
## 3.1 OminiControl

![image.png](https://chenshi-1327871112.cos.ap-chongqing.myqcloud.com/obsidian20250710233317580.png)

(b)图表示的是原始DiT，无条件控制，(c)图是ControlNet等使用的通过一个**独立的、可训练的**网络分支来实现条件控制，(d)是 **OminiControl** 采用的方法

### 3.1.1 最小化架构

- **参数重用 (Parameter Reuse)**：
    - **重用 VAE 编码器**：直接使用 DiT 模型自带的 VAE 编码器来处理输入的条件图像（如 Canny 边缘图），将其映射到与噪声图像相同的潜在空间，无需引入额外的编码器。
    - **重用 Transformer 块**：条件 token 和图像 token 共享 DiT 模型中的 Transformer 块进行处理。        
- **轻量级微调 (Lightweight Fine-tuning)**：仅使用 **LoRA** 对共享的 Transformer 块进行微调，以适应新的条件输入，避免了全量参数更新。
### 3.1.2 Omni-capable token interaction

**a）统一序列处理**

ControlNet 等方法通过将条件特征图直接加到 U-Net 的中间层特征图上。这种硬性相加的方式天然适合空间对齐任务，但对于非空间对齐任务则显得僵硬，限制了模型学习更复杂的语义关系。

本文将 VAE 编码后的条件 token $C_I$、文本 token $C_T$ 和噪声图像 token $X$ **直接拼接**成一个长序列 $[X; C_T; C_I]$。使用注意力机制自由学习token间关系。

**b）动态位置编码**

根据任务类型**动态地**选择位置编码策略。
- $(i, j)_{C_I} = (i, j)_X$ ---(空间对齐）
- $(i, j)_{C_I} = (i, j)_X + Δ$ (非空间对齐)

**c）灵活的控制强度**

将条件拼接后不能直接控制强度，将注意力公示改为
$$MMA([X; C_T; C_I]) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B(\gamma)\right)V$$
$$B(\gamma) = \begin{pmatrix} 0_{M \times M} & 0_{M \times N} & 0_{M \times N} \\ 0_{N \times M} & 0_{N \times N} & \log(\gamma) \cdot \mathbf{1}_{N \times N} \\ 0_{N \times M} & \log(\gamma) \cdot \mathbf{1}_{N \times N} & 0_{N \times N} \end{pmatrix}$$
- **对角线上的 `0` 矩阵**:
  - $0_{M \times M}$: $C_T$ 内部 token 之间的注意力分数**不加偏置**。
  - $0_{N \times N}$: $X$ 内部 token 和 $C_I$ 内部 token 之间的注意力分数也**不加偏置**。
  - 这意味着模型内部的自注意力关系（文本内部、图像内部）保持原样。

- **非对角线上的 `log(γ)` 矩阵**:
  - **关键部分**: 偏置项只被添加到了**带噪图像 token `X`** 与 **条件图像 token `C_I`** 相互作用的部分。
  - $\log(\gamma) \cdot \mathbf{1}_{N \times N}$: 这是一个所有元素都为 $\log(\gamma)$ 的矩阵。$\mathbf{1}$ 是全1矩阵。

## 3.2 Subjects200K 数据集

- **构建流程**：
    1. **提示生成 (Prompt Generation)**：使用 GPT-4o 生成超过 30,000 个多样化的主体描述。
    2. **图像对合成 (Paired-image Synthesis)**：将主体描述构造成“同一主体在两个不同场景”的结构化提示，输入 FLUX.1 生成图像对。
    3. **质量评估 (Quality Assessment)**：再次使用 GPT-4o 评估生成的图像对，剔除身份不一致或质量低的样本。

# 4. 实验/评估/结果

- **基础模型**：FLUX.1。
- **训练数据集**：text-to-image-2M（for 空间对齐），Subjects200K（for 非空间对齐）
- **对比模型**：
![image.png](https://chenshi-1327871112.cos.ap-chongqing.myqcloud.com/obsidian20250711113207791.png)

与其他方法相比，OminiControl 仅需 14.5M（约 0.1%）的可训练参数，远低于其他方法

![image.png](https://chenshi-1327871112.cos.ap-chongqing.myqcloud.com/obsidian20250711113346215.png)

在下面五个Task中，
- Canny: 使用 Canny 算子提取的边缘图作为条件。
- Depth: 使用深度图作为条件。
- Mask: 使用蒙版进行图像修复
- Colorization: 图像上色，输入是灰度图，输出是彩色图。
- Deblur: 图像去模糊。

>[!note]
>**F1↑**：比较↑生成图像的边缘图与输入的原始 Canny 边缘图，F1 分数是精确率和召回率的调和平均值
>**MSE↓**：在任务条件下计算生成与原始图的像素级别的均方误差
>![[FID]]
>![[SSIM]]
>**CLIP-IQA**：[IceClear/CLIP-IQA: [AAAI 2023] 探索 CLIP 在图像视觉感知质量评估中的应用 --- IceClear/CLIP-IQA: [AAAI 2023] Exploring CLIP for Assessing the Look and Feel of Images](https://github.com/IceClear/CLIP-IQA)
>**MAN-IQA**：[IIGROUP/MANIQA: [CVPRW 2022] MANIQA: 基于多维度注意力网络的无参考图像质量评估 --- IIGROUP/MANIQA: [CVPRW 2022] MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment](https://github.com/IIGROUP/MANIQA)
>**MUSICQ**：[anse3832/MUSIQ: Unofficial implementation of MUSIQ (Multi-Scale Image Quality Transformer)](https://github.com/anse3832/MUSIQ)
>**PSNR**：[峰值信噪比 ](https://zh.wikipedia.org/wiki/%E5%B3%B0%E5%80%BC%E4%BF%A1%E5%99%AA%E6%AF%94)
>![[CLIP Text]]
>![[CLIP Image]]

![image.png](https://chenshi-1327871112.cos.ap-chongqing.myqcloud.com/obsidian20250711111037462.png)

五项评价指标，使用GPT-4o 的视觉能力完成，**但未提供具体的计算方法**：
- 身份特征保持：评估关键识别特征（如标识、品牌标记、独特图案）的保留程度
- 材质质量：检验材料属性与表面特征是否得到准确呈现
- 色彩保真度：评估未指定修改区域的色彩一致性
- 自然呈现：评估生成图像是否真实且具有连贯性
- 修改准确性：验证文本提示中指定的修改是否被正确执行

# 5. 结论

OminiControl 通过统一的标记方法，为 DiT 提供了参数高效的图像条件控制，仅需增加 0.1%的参数即可适用于多样化任务。Subjects200K 数据集包含超过 20 万张高质量、主体一致的图像，进一步推动了主体驱动生成领域的发展，实验结果证实了 OminiControl 在空间对齐和非对齐任务中的有效性。
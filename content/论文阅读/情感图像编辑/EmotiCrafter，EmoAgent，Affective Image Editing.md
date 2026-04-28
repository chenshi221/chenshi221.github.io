#多模态 #情感 #AIM 
[[2505.18699] Affective Image Editing: Shaping Emotional Factors via Text Descriptions](https://arxiv.org/abs/2505.18699)
[[2504.09689] EmoAgent: Assessing and Safeguarding Human-AI Interaction for Mental Health Safety](https://arxiv.org/abs/2504.09689)
[[2501.05710] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](https://arxiv.org/abs/2501.05710)

## 《EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model》
#C-EICG

1. 目标： 开发一个模型，能够根据文本提示和连续的情感数值（Valence-Arousal），生成内容可控且情感丰富、准确的图像。

> [!info]
> Valence (愉悦度): 1.65 (比较积极)
> Arousal (激动度): -0.12 (比较平静)
> 均介于$[-3,+3]$

2. 方法：作者提出了 **EmotiCrafter**，一个基于稳定扩散模型 (Stable Diffusion XL) 的情感图像生成框架。
	- 核心思想: 不直接生成图像，而是通过一个**情感嵌入网络 (Emotion-Embedding Network)**，将用户指定的连续情感值（Valence-Arousal）注入到文本提示的特征中，从而引导图像生成模型产生带有特定情感的图像。

### 整体流程

![image.png](https://chenshi-1327871112.cos.ap-chongqing.myqcloud.com/obsidian20250615190659540.png)


1. 输入: 一个描述图像内容的中性文本提示 (Neutral Prompt) 和一对表示情感的V-A值 (Valence, Arousal)。    
2. **情感嵌入网络 (M)**:
    - 首先，一个**V-A编码器 (V-A Encoder)** 将V-A数值对通过两个独立的多层感知机(MLP)转换为V特征向量($e_v$)和A特征向量($e_a$)。
    - ![image.png](https://chenshi-1327871112.cos.ap-chongqing.myqcloud.com/obsidian20250615190718419.png)

    - 然后，一个**情感注入转换器 (Emotion Injection Transformer, EIT)** 将中性文本的特征和V-A特征向量融合。它通过交叉注意力机制，将情感特征注入到文本特征的处理流程中，生成带有情感信息的**情感化特征 (Emotional Prompt Feature)**。
3. 图像生成 (G):
    - 将生成的情感化特征输入到预训练的SDXL模型的U-Net中，通过交叉注意力机制指导图像的去噪生成过程，最终输出符合文本内容和指定情感的图像。


## 《Affective Image Editing: Shaping Emotional Factors via Text Descriptions》

### 1. 论文简介

1. 目标：优化处理**情感需求**，目前模型存在缺陷：
	1. **情感理解能力弱**: 通用编辑模型（如 InstructPix2Pix）无法准确理解和执行带有情感色彩的编辑指令。
	2. **控制粒度粗糙**: 现有情感编辑方法多依赖离散的情感类别或只关注单一情感因素，限制了用户表达细腻、复杂情感的能力。例如[[EmoEdit]]

![image.png](https://chenshi-1327871112.cos.ap-chongqing.myqcloud.com/obsidian20250615181203434.png)


2. 方法：作者提出了 **AIEdiT (Affective Image Editing using Text descriptions)**，一个基于扩散模型的、通过文本描述进行情感编辑的框架。
	1. 核心思想: 设计一个**情感映射器 (Emotional Mapper)**，将用户输入的、视觉上抽象的情感文本请求，转化为视觉上具体的语义表示，并利用多模态大语言模型 (MLLM) 作为监督，确保编辑结果能够准确唤起目标情感。

### 2. EmoTIPS 数据集

1. 方法： 从EmoSet 中收集情感图像，并使用 MLLM 为每个样本生成相应的情感文本描述，包括三个阶段：标注、验证和评估
   ![image.png](https://chenshi-1327871112.cos.ap-chongqing.myqcloud.com/obsidian20250615180621016.png)
2. 步骤：
	1. 通过GPT4生成描述
	2. 将生成的文本输入到一个**VAD词典 (VAD dictionary)** 中。VAD代表Valence（效价）、Arousal（唤醒度）、Dominance（支配度），是描述情感的维度。这里主要用它来识别情感词
	3. 使用**EmoLLMs**和 **ResNet**，验证从文本中识别出的情感与从图片中识别出的情感是否一致，
	4. 用BLIP2进行图文检索，:验证生成的文本描述是否足够具体和独特

### 3. AIEdiT

![image.png](https://chenshi-1327871112.cos.ap-chongqing.myqcloud.com/obsidian20250615181621561.png)

#### 3.1 情感光谱构建

1.  **目的**: 学习一个能够表示普适情感的连续空间。
2.  **方法**:
    1. 使用BERT提取文本的情感请求 $r$。
    2. 使用预训练的ResNet估计图像的情感类别分布 $d$ (基于Mikel's wheel)。
    3. 将$(r, d)$ 配对，构成样本 $s$。
    4. 通过连续情绪表征学习，使得在情感上相似的样本在特征空间中距离更近，情感上对立的样本距离更远。

#### 3.2 情感映射器设计

1.  **目的**: 将抽象的情感文本请求 $r$ 翻译成具体的视觉语义表示 $r'$。
2.  **结构**: 一个多模态Transformer，包含自注意力(MSA)、交叉注意力(MCA)和前馈网络(FFN)。
3.  **工作流程**:
    1. 输入情感请求 $r$、文本语义 $r_s$ (来自CLIP) 和关键语义特征 $f_k$。  
    2. $f_k$ 在每个子模块后对情感特征进行**缩放**，动态调整情感的表达强度。
    3. MCA模块负责将情感语义与相关的视觉元素进行融合。
    4. 最终输出与视觉对齐的情感表示 $r'$。
####  3.3 MLLM监督与训练

1.   **挑战**: 缺乏目标情感图像，无法直接监督情感映射器的学习。
2.  **解决方案**: 使用MLLM (ShareGPT4V) 进行自监督学习。
3.  **方法**:
    *   针对输入图像，向MLLM提出关于**色调、场景、表情/动作**等情感相关因素的问题。
    *   将MLLM的回答作为“真实”的语义表示。
    *   设计**情感对齐损失 (Sentiment Alignment Loss, $L_{sa}$)**，最小化情感映射器生成的语义表示与MLLM回答的CLIP嵌入之间的差距。
    *   总损失 $L_{\text{total}}$ 的计算公式为：
      $$
      L_{\text{total}} = L_{sa} + \beta \cdot L_{dm}
      $$
      其中 $L_{dm}$ 是标准的扩散模型噪声预测损失。

#### 3.4 情感图像编辑

1.  **图像编辑流程**:
    *   **编码**: 输入的原始图像 $I_{in}$ 被 Encoder 编码为潜在表示 $z$。
    *   **加噪**: 向 $z$ 中加入一定量的噪声 $\epsilon$，得到带噪的 $z_t$。这是扩散模型的标准步骤。
    *   **去噪 (核心控制点)**: **去噪网络 $\epsilon_{\theta}$** (也就是我们常说的U-Net) 开始工作。它在去噪的每一步，都会将从 **(b)中得到的情感指令 r'** 作为条件输入。**这就是控制的关键所在**：$r'$ 告诉去噪网络应该朝着哪个情感方向恢复图像，比如是让颜色更鲜艳，还是让人物表情更微笑。
    *   **解码**: 去噪完成后得到干净的潜在表示 $z_0$，再通过 **Decoder** 恢复成最终的输出图像 $I_{out}$。
    *   **损失 $L_{dm}$**: 这是标准的扩散模型损失，用于训练去噪网络。

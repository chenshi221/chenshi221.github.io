# Layer Normalization (LN)

- **目的**：LN 是一种归一化技术，用于稳定神经网络的训练过程。它的主要作用是**缓解“内部协变量偏移” (Internal Covariate Shift)** 的问题，即在训练过程中，网络各层参数的更新会导致下一层输入的分布发生变化，使得训练变得困难。
- **工作方式**：与批量归一化 (Batch Normalization, BN) 对一个批次内的数据进行归一化不同，LN 是在**单个样本**的**特征维度**上进行操作的。对于一个样本的特征向量 x，LN 会计算这个向量所有元素的均值 μ 和标准差 σ，然后进行归一化：
    $$x_{normalized} = (x - μ) / \sqrt{(σ² + ε)}$$
- **可学习的参数**：为了保持模型的表达能力，LN 在归一化之后，还会通过两个可学习的参数 γ (缩放) 和 β (偏移) 对结果进行仿射变换：
    $$LN(x) = γ * x_{normalized} + β$$
# adaLN (Adaptive Layer Normalization)

**adaLN** 的全称是 **Adaptive Layer Normalization (自适应层归一化)**。它的核心思想是：**让 γ 和 β 这两个参数不再是固定的、可学习的权重，而是根据外部的条件（Conditioning）动态生成。**
- **"Adaptive" (自适应) 的含义**：这里的“自适应”指的是 LN 的行为会根据输入的**条件**而改变。这个条件可以是任何你想用来控制模型生成过程的信息。在 DiT (Diffusion Transformer) 模型中，这个条件通常是：
    - **时间步 (Timestep t)**: 告诉模型当前处于去噪过程的哪一步。
    - **类别标签 (Class Label y)**: 告诉模型要生成哪个类别的图像。
- **工作流程**：
    1. **输入条件**: 将条件信息（如时间步 t 和类别 y）首先通过嵌入层 (Embedding Layer) 转换成向量。
    2. **生成控制参数**: 将这些条件向量送入一个小的神经网络（通常是一个简单的 MLP），这个网络的输出就是用于 LN 的 γ 和 β。
        $$MLP(condition) -> (γ, β)$$
    3. **应用到 LN**: 在对主干网络的特征进行 Layer Normalization 时，使用这些**动态生成**的 γ 和 β 来进行最后的仿射变换。
	    $$adaLN(x, condition) = γ_{condition} * x_{normalized} + β_{condition}$$

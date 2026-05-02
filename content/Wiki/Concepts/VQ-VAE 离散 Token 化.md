---
type: concept
status: active
created: "2026-05-01"
updated: "2026-05-01"
aliases: ["VQ-VAE", "离散Token化", "Vector Quantization", "RQ-Kmeans", "离散扩散"]
tags: ["tokenization", "discrete-representation", "VQ", "generative-model", "quantization"]
sources: ["[[Wiki/Sources/Lumina-DiMOO]]", "[[Wiki/Sources/Show-o2]]", "[[Wiki/Sources/Emu3.5]]", "[[Wiki/Sources/OneRec Technical Report 2025]]", "[[Wiki/Sources/BAGEL]]"]
confidence: high
---

# VQ-VAE / 离散 Token 化

## 定义

VQ-VAE（Vector Quantized Variational AutoEncoder）是一种将连续信号（图像、视频、音频）压缩为**离散潜在表示**的自编码器架构。其核心思想是在编码器和解码器之间插入一个**可学习的离散码本（codebook）**，将连续编码向量映射为最近的码本条目索引。这一技术是连接「连续视觉信号」与「离散 Token 序列」的关键桥梁。

## 核心原理

### 1. 基本架构

```
输入图像 x → Encoder → z_e(x) 连续特征
                        ↓
              Codebook 最近邻查找: z_q = argmin_e ‖z_e - e_k‖
                        ↓
        离散索引 k → 查表得 e_k → Decoder → 重建 x̂
```

### 2. 训练目标

VQ-VAE 的损失函数包含三个分量：

$$L = \underbrace{\|x - \hat{x}\|^2}_{\text{重建损失}} + \underbrace{\|\text{sg}[z_e] - e\|^2}_{\text{码本损失}} + \underbrace{\beta \|z_e - \text{sg}[e]\|^2}_{\text{承诺损失}}$$

- **码本损失**：让码本向量向编码器输出靠拢
- **承诺损失**：让编码器输出向码本向量靠拢（防止编码器输出无限制漂移）
- `sg[·]`：stop-gradient 操作，阻断梯度回传

### 3. 关键挑战

| 问题 | 说明 | 缓解策略 |
|------|------|----------|
| **Codebook Collapse** | 大部分码本条目被闲置，只有少数被使用 | EMA 更新码本、codebook reset、熵正则化 |
| **重建质量 vs 压缩率** | 码本越小/维度越低 → 重建越差 | 残差量化（多级码本逐级补充细节） |
| **离散化不可导** | argmin 操作没有梯度 | Straight-through estimator（STE）：前向用离散，反向直接把 z_q 的梯度拷贝给 z_e |

## 重要变体

### RQ-Kmeans（残差 K-means 量化）

OneRec 系列使用的 item tokenization 方法，是将 K-means 应用于残差的层级量化：

```
z → k1 = K-means(z)      → 残差 r1 = z - e_{k1}
r1 → k2 = K-means(r1)    → 残差 r2 = r1 - e_{k2}
r2 → k3 = K-means(r2)    → ...
```

最终一个 item 被表示为多级语义 ID 序列 `[k1, k2, ..., kL]`，粗到细地编码语义层次。

**与 RQ-VAE 的区别**：RQ-VAE 学 codebook（梯度更新），RQ-Kmeans 直接聚类（无需梯度）。OneRec 发现 RQ-Kmeans 在码本利用率和平衡性上优于 RQ-VAE。

### VQ-GAN

在 VQ-VAE 基础上加入对抗损失（GAN discriminator）和感知损失（LPIPS），显著提升重建图像的视觉质量。是 DALL-E、Stable Diffusion 等模型的图像 tokenizer 基础。

### FSQ（Finite Scalar Quantization）

将连续值直接取整到有限离散区间，无需可学习码本，避免了 codebook collapse。简单但码本容量固定。

## 在生成模型中的核心应用

### 1. 自回归视觉生成

模型将图像 token 化为离散序列后，用标准 Transformer 做 next-token prediction：

| 模型 | Tokenizer | 说明 |
|------|-----------|------|
| **Show-o2** | 3D Causal VAE | 图文视频统一 token 空间，双路径空间-时序融合 |
| **Emu3.5** | VQ tokenizer | 10T+ tokens 图文交错 next-token prediction，DiDA 加速 20x |
| **DALL-E (1)** | dVAE (VQ-VAE 变体) | 首个大规模 AR 图像生成 |

### 2. 离散扩散模型

不同于连续扩散（加高斯噪声），离散扩散在离散 token 空间上做前向/反向过程：

- **Lumina-DiMOO**（上海 AI Lab）：使用 aMUSEd-VQ tokenizer，纯离散扩散范式。速度比连续扩散快 32 倍，支持理解与生成统一。
- 离散扩散的优势：token 空间与文本天然对齐，便于多模态统一。

### 3. 推荐系统的 Item Tokenization

- **OneRec / OpenOneRec**：用 RQ-Kmeans 将 item（视频/商品）的多模态特征 + 协同信号量化为 3-4 级语义 ID
- 自回归生成语义 ID → 推荐从「检索排序」变为「序列生成」

### 4. 扩散模型的条件压缩

- **BAGEL** 的 VAE tokenizer 将图像压缩为潜在 token，作为 MoT Transformer 的视觉输入
- 连续潜在（非离散化）做 Diffusion Forcing 生成

## 连续 vs 离散表示

| 维度 | 连续潜在（VAE/扩散） | 离散 Token（VQ-VAE） |
|------|---------------------|---------------------|
| 信息保真度 | 更高（无量化损失） | 有量化损失 |
| 建模工具 | 连续扩散、Flow Matching | 自回归、离散扩散 |
| 与文本统一 | 需要桥接层 | 天然对齐（同是离散 token） |
| 生成速度 | 取决于采样步数 | 逐 token 生成（慢） |
| 代表工作 | SD3, FLUX, DiT | Show-o2, Lumina-DiMOO, Emu3.5 |
| 推理加速 | 蒸馏、rectified flow | DiDA（Emu3.5, 20x） |

## 架构选择的关键权衡

### 什么情况下应该选离散 Token？

1. **需要与文本统一建模**：如果你想让 LLM 直接理解图像 token，离散是最直接的路径
2. **自回归生成足够快**：当 token 数量少（如推荐系统的 3-4 级语义 ID）时，AR 生成的开销可忽略
3. **需要离散操作**：如搜索、排序、编辑（替换某个 token = 修改某个语义属性）

### 什么情况下应该选连续表示？

1. **图像质量优先**：连续潜在没有量化损失，重建保真度更高
2. **计算预算充足**：可以承受更多采样步数
3. **需要细粒度控制**：连续空间支持插值、连续调节（如 ControlNet 的空间条件）

## 与已有 Wiki 的关系

- 概念基础：[[Wiki/Concepts/扩散模型原理]]（连续扩散） ← 离散扩散是其离散对应
- 应用：[[Wiki/Concepts/生成式推荐]]（RQ-Kmeans 做 item tokenization）
- 应用：[[Wiki/Concepts/原生多模态模型]]（Show-o2、Emu3.5 依赖 VQ tokenizer）
- 架构选择：[[Wiki/Comparisons/统一多模态模型架构比较]]（离散 vs 连续之争）

---

## 深度分析：离散表示的「必要性」再思考

### 1. 离散化是必须的，还是历史包袱？

Show-o2 和 Emu3.5 将图像 token 化为离散序列，是因为它们想用现有的 LLM 架构（自回归 next-token prediction）做视觉。但认真想想：**LLM 使用离散 token 本身是自然语言的产物，不是最优架构的产物**。如果我们从零设计一个多模态架构，可能根本不需要离散化这个环节——连续潜在 + 扩散/Flow Matching 在数学上更自然。

离散化的唯一不可替代价值是**可组合性**：离散 token 让视觉和语言在同一「操作空间」中，编辑、搜索、推理都共享同一套符号操作机制。

### 2. RQ-Kmeans 的工业选择是务实的

RQ-Kmeans 没有可学习参数（不像 RQ-VAE 需要梯度更新），在推荐系统中这有巨大优势：
- 无需联合训练（推荐系统和 tokenizer 解耦）
- 新 item 加入只需一次推理 + 聚类，无需重新训练 codebook
- 码本质量可控（K-means 是成熟算法，不会 codebook collapse）

这是「简单的正确方案」的典型案例——如果能不学，就别学。

### 3. 离散扩散的 32x 加速需要独立验证

Lumina-DiMOO 声称的 32x 速度提升，可能部分来自离散扩散本身的效率（更少步数），部分来自工程优化（更小的 token 序列）。32 这个数字在没有第三方复现前，应该视为上限估计。实际部署中，10-15x 的加速更可信。

### 4. 未来预测

- **短期（2-3 年）**：连续扩散仍是图像生成的主流（质量优势）。离散 Token 在推荐和多模态统一中逐步扩大份额。
- **长期（5+ 年）**：可能出现**连续-离散混合表示**——底层用高保真连续潜在做生成，顶层用离散抽象 token 做推理和编辑。这比纯离散或纯连续都更合理：生成不需要离散化的精度损失，推理需要离散化的可组合性。

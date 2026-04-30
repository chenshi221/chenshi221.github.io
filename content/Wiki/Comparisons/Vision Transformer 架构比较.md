---
type: comparison
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - ViT 架构对比
  - ViT vs Swin vs MLP-Mixer
tags:
  - comparison
  - vision-transformer
  - ViT
  - Swin
  - MLP-Mixer
  - architecture
sources:
  - '[[Wiki/Sources/ViT]]'
  - '[[Wiki/Sources/Swin Transformer]]'
  - '[[Wiki/Sources/MLP-Mixer]]'
confidence: medium
---

# Vision Transformer 架构比较

## 为什么这个比较重要

ViT → Swin → MLP-Mixer 的表面叙事是"Vision Transformer 越来越好用"，但如果换一个视角，这三个工作恰好构成了一场连续的**归纳偏置剥离实验**。这不是"第一种好于第二种"的线性故事，而是对"视觉理解到底需要什么架构"这个元问题的三次逐级追问。

---

## 一、"归纳偏置剥离实验"视角

如果把这三个工作按照"保留了什么归纳偏置"重新排列，一个极其清晰的模式浮现出来：

| 工作 | 保留的归纳偏置 | 剥离的归纳偏置 | ImageNet-1K Top-1 (无大规模预训练) |
|------|-------------|-------------|------|
| ViT | 注意力（全局相关性） | 卷积、局部性、平移等变性 | ~77%（远不如 ResNet） |
| Swin | 注意力 + 层级 + 局部窗口 | 卷积、平移等变性 | ~83%（与 ResNet 相当） |
| MLP-Mixer | **无**（纯 MLP） | 卷积、注意力、局部性、平移等变性 | ~77%（与 ViT 相当） |

每剥离一层视觉归纳偏置，模型在中小数据上的性能就下降一级——除非你有 JFT-300M 或 ImageNet-21K 级别的大规模预训练数据。

### 深层判断

**这不是三个独立的模型设计，而是一次隐含的实验设计。** ViT 说"卷积不必要"，MLP-Mixer 说"注意力也不必要"。但注意：说"不必要"的前提是"数据足够多"。在 JFT-300M 上 ViT 超越了 CNN，在 ImageNet-21K 上 MLP-Mixer 匹配了 ViT。"数据可以替代架构归纳偏置"是这个系列工作最深的洞见，也是最重要的警告——如果你没有足够数据，就别学 MLP-Mixer 把偏置都扔了。

这个视角让我对当前的"统一多模态"研究也有一个判断：那些声称"不需要视觉编码器"的模型（如 Tuna-2），本质上是在重复 MLP-Mixer 的实验逻辑——用海量数据弥补架构偏置的缺失。这是有效的，但不要误以为它证明了"编码器无用"。它只是证明了"如果有 5.1T tokens，编码器可以不显式存在"。

---

## 二、为什么 Swin 成为唯一真正的通用 backbone？

ViT 和 MLP-Mixer 在 ImageNet 分类上表现不差，但在 COCO 检测和 ADE20K 分割上远不如 Swin。这不是性能差异，而是**架构能力边界**差异。

### 层级结构是关键

| 能力 | ViT | MLP-Mixer | Swin |
|------|-----|-----------|------|
| 分类 | ✅ | ✅ | ✅ |
| 目标检测 | ❌ 差 | ❌ 差 | ✅ 好 |
| 语义分割 | ❌ 差 | ❌ 差 | ✅ 好 |
| 适配 FPN | ❌ | ❌ | ✅ 天然适配 |
| 多尺度特征 | 没有（固定分辨率） | 没有（固定分辨率） | 有（金字塔） |
| 线性复杂度 | ❌ O(N²) | ✅ O(N) | ✅ O(N) |

ViT 从第一层起每个 patch 就能关注到所有其他 patch——这是"全局感受野"的优势，也是"固定分辨率 + 二次复杂度"的灾难。检测和分割需要多尺度特征金字塔（FPN），需要在不同分辨率上操作，需要高分辨率输入。ViT 一个都不满足。

Swin 的层级化设计不是一个"小改进"，而是根本性地改变了模型的适用场景——从"只能分类"变成了"通用视觉 backbone"。这也是它获得 ICCV 2021 最佳论文奖的根本原因：不是因为它比 ViT 参数更多或性能更高，而是因为它回答了"Vision Transformer 怎么才能像 ResNet 一样到处能用"这个工程问题。

**MLP-Mixer 在通用性上的缺席进一步证实了这个判断**：即使 MLP 可以模拟全局依赖，没有层级结构的模型天然不适合 dense prediction 任务。归纳偏置的缺失可以通过数据弥补，但结构性缺陷（没有多尺度）无法通过数据弥补。

---

## 三、MLP-Mixer 的真正价值：一个哲学追问

MLP-Mixer 从来就不是一个好的 backbone，但它是一个极好的**概念实验**。

它问的不是"怎么做更好"，而是"什么才是必需的"——如果纯 MLP 就能达到 80%+ 准确率，那我们精心设计的卷积、注意力、局部性、平移等变性，到底有多大价值？

### 两个层面上的回答

**技术层面**：价值仍然很大。MLP-Mixer 在 ImageNet-1K 上不加 JFT-300M 预训练时，性能远不如 ResNet。归纳偏置节省的不是"上限"，而是"达到上限所需的数据量"。这是为什么小数据场景下 CNN 仍然更好用的根本原因。

**哲学层面**：价值被重新定义。在"大数据时代"之前，架构设计是决定性能的主要因素。在大数据时代之后，架构设计变成了"给定数据量下的效率加速器"，而不是能力的唯一来源。MLP-Mixer 的价值在于它把这个哲学转变推到了极致——它证明了架构可以什么事都不做（纯 MLP），只要数据够多。

### 与当前的链接

MLP-Mixer 的精神后代是 Tuna-2（无编码器统一多模态模型）和 Emu3.5（原生 token 化世界模型）。这些工作都在说同一句话："去掉精心设计的组件，用更多的数据。" 它们是对的——但不是无条件对的。条件就是你确实有那么多数据。

---

## 四、当前趋势：ViT 在 VLM/扩散模型中的新角色

Swin 之后的层级设计几乎被所有现代 ViT 变体吸收（PVT、CSwin、DiT 等）。但 2024-2025 年的新问题是：**在 VLM（视觉语言模型）和扩散模型中，ViT 的角色是什么？**

有三个值得关注的趋势：

1. **在 VLM 中，ViT 变成"视觉 token 生成器"**。CLIP/SigLIP 的 ViT 编码器输出视觉 token，送入 LLM 解码器。ViT 的架构设计（层级 vs 非层级）开始让位于"视觉 token 数量和质量"的考虑。[[Wiki/Concepts/多模态对比学习|SigLIP 2]] 的 dense feature 改进就是在这个方向上的优化。

2. **在扩散模型中，DiT（Diffusion Transformer）接替了 UNet**。DiT 直接用了 ViT 的架构骨架（层归一化、自注意力、MLP），但适应了扩散模型的条件注入需求。[[Wiki/Sources/OminiControl|OminiControl]] 展示了 DiT 可以用 0.1% 的额外参数实现高质量条件控制。

3. **原生多模态模型模糊了 ViT 的边界**。当理解任务是 LLM + ViT，生成任务是 DiT 时，两个 ViT 是否应该共享？[[Wiki/Sources/Emu3.5|Emu3.5]] 和 [[Wiki/Sources/Tuna-2|Tuna-2]] 分别走了两条路：Emu3.5 是最极端的不共享（纯 token 化，无编码器），Tuna-2 也是无编码器。BAGEL 走了中间路——MoT 架构中双编码器但共享 backbone。

我的判断：**ViT 正在从"独立模型"变成"多模态模型的一个模块"**。当 ViT 不再独立存在时，对它架构的讨论也就不再独立存在——它会和 LLM 的架构选择、扩散头部的选择绑在一起。Swin 的层级化设计在"ViT 作为独立 backbone"的时代是正确方向，但在"ViT 作为 VLM 视觉编码器"的时代，是否有层级结构不再是首要问题。4K/8K CLIP token 和 16K 以上 token 之间的区别，可能比层级 vs 非层级更重要。

---

## 相关页面

- [[Wiki/Topics/Vision Transformer 演进]] — 主题总览
- [[Wiki/Entities/Vision Transformer (ViT)]] — ViT 详情
- [[Wiki/Entities/Swin Transformer]] — Swin 详情
- [[Wiki/Comparisons/统一多模态模型架构比较]] — ViT 在 UMM 中的角色
- [[Wiki/Concepts/多模态对比学习]] — CLIP/SigLIP 的 ViT 编码器


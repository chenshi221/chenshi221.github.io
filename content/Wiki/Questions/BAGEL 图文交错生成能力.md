---
type: question
status: active
created: "2026-04-30"
updated: "2026-04-30"
tags: [unified-multimodal, interleaved-generation, BAGEL, comparison]
sources: ["[[Wiki/Sources/BAGEL]]", "[[Wiki/Sources/WEAVE]]"]
confidence: high
---

# BAGEL 是否支持图文交错生成？

## 问题

BAGEL 能否在单次推理中生成文本-图像-文本的交错输出序列？

## 答案：✅ 支持

BAGEL 是目前少数明确支持图文交错生成的统一多模态模型之一。

### 架构支撑

BAGEL 的 **Generalized Causal Attention** 机制专门为交错序列设计：

```
输入/输出序列示例：
[文本] "这是日出的照片" → [图像: 日出] → [文本] "现在改成黄昏风格" → [图像: 黄昏]
```

序列中后续的 token（无论是文本还是图像）可以通过因果注意力 attend 到前面图像的：
- **Clean VAE tokens**（干净潜变量，作为条件）
- **ViT tokens**（SigLIP2 编码，统一交错格式、提升生成质量）
- 但不能 attend 到 **Noised VAE tokens**（仅用于 rectified-flow 训练损失计算）

### 多图像交错：Diffusion Forcing

对于连续生成多张图像的交错序列，BAGEL 使用 **Diffusion Forcing** 策略：
- 每张图像使用独立的噪声水平
- 每张图像 condition on 前面图像的 noisy representation
- 随机分组连续图像 + full attention within group 增强一致性

### 训练数据

- 45M 视频交错生成数据（Video Interleaved Generation）
- 20M 网页交错生成数据（Web Interleaved Generation）
- 100M 交错理解数据（Interleaved Understanding）

### 与其他模型的对比

| 模型 | 交错输入理解 | 交错生成输出 | 实现方式 |
|------|-------------|-------------|----------|
| **BAGEL** | ✅ | ✅ | Generalized Causal Attention + Diffusion Forcing |
| DreamOmni2 | ✅ | ❌ | VLM + 扩散解耦，不支持交错输出 |
| OmniGen2 | ✅ | ❌ | VLM 冻结 + 扩散解码，不支持交错输出 |
| Show-o2 | ✅ | ❓ | 3D VAE 双路径，论文未明确 |
| WEAVE | ✅（多轮） | N/A (benchmark) | 多轮对话中的上下文编辑 |

### 为什么其他模型不支持？

组装式架构（VLM + Diffusion）的固有局限：文本生成和图像生成使用不同的模块，无法在同一序列中自由切换。BAGEL 的 MoT 架构把两种 Expert 放在同一个 decoder-only 框架中，共享 self-attention，天然支持交错序列。

### 深层含义

图文交错生成不仅仅是"多了一个功能"——它改变了模型的使用范式：
- **传统**：用户给指令 → 模型出图（单次、单向）
- **交错**：用户和模型可以围绕一个视觉创作任务进行持续的、混合文本和图像的对话
- 这对创意工作流（设计师不断迭代）、教育（图文并茂解释）、数据生成（自动创建图文交错训练数据）都有重要意义

## 相关页面

- [[Wiki/Entities/BAGEL]]
- [[Wiki/Comparisons/统一多模态模型架构比较]]
- [[Wiki/Topics/扩散模型图像编辑与生成]]

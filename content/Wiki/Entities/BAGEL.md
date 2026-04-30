---
type: entity
status: active
created: "2026-04-30"
updated: "2026-04-30"
aliases: [ByteDance BAGEL]
tags: [unified-multimodal, bytedance-seed, mot, interleaved-generation]
sources: ["[[Wiki/Sources/BAGEL]]"]
confidence: high
---

# BAGEL（ByteDance Seed）

**BAGEL** 是 ByteDance Seed 团队在 2025 年 5 月发布的开源统一多模态基础模型。

## 基本信息

| 属性 | 值 |
|------|-----|
| 机构 | ByteDance Seed + Monash + HKUST + UC Santa Cruz |
| 架构 | Mixture-of-Transformers (MoT)，7B 激活 / 14B 总参数 |
| 编码器 | SigLIP2 ViT（理解） + FLUX VAE（生成） |
| 训练数据 | 5.1T tokens（文本 + 图文对 + 交错视频 + 交错网页） |
| 训练策略 | 四阶段：Alignment → PT (2.5T) → CT (2.6T) → SFT |
| 开源 | ✅ 代码和权重已发布 |

## 核心能力

- **统一理解与生成**：单一 decoder-only 模型，无瓶颈 MoT 架构
- **图文交错生成** ✅：在单次推理中输出文本→图像→文本的交错序列。Generalized Causal Attention + Diffusion Forcing 策略支撑
- **涌现推理**：随训练规模增长自发涌现复杂多模态推理能力
- **Self-CoT**：推理时 Chain-of-Thought 显著提升编辑和推理任务
- **自由形式编辑**、**3D 操作**、**世界导航**、**未来帧预测**

## 架构详解

### MoT（Mixture-of-Transformers）

两种 Transformer Expert 共享 self-attention：
- **理解 Expert**：处理文本 token 和 SigLIP2 ViT token
- **生成 Expert**：处理 FLUX VAE token（噪声/干净/图像）

使用 hard routing：生成 Expert 独占 VAE token，理解 Expert 处理其余。

### Generalized Causal Attention（交错生成的关键）

```
序列：[文本₁] → [图像₁: noise VAE + clean VAE + ViT] → [文本₂] → [图像₂: ...]
                ←─── attend ────
        后续 token 可以 attend 到前面图像的 clean VAE + ViT token
        但不能 attend 到 noise VAE（仅用于 rectified-flow 训练）
```

- 多图像交错：Diffusion Forcing 策略，每张图独立噪声水平
- 随机分组连续图像 + full attention within group 增强一致性

### 涌现模式

| 能力 | 达到饱和的 token 量 |
|------|---------------------|
| 基本理解 | ~0.18T |
| 基本生成 | ~0.68T |
| 复杂编辑 | ~2.64T（85% 性能） |
| 推理编辑 (Intelligent Edit) | >3.61T（开始显著提升） |

## 与其他 UMM 的关键区别

| 维度 | BAGEL | DreamOmni2 | OmniGen2 | Show-o2 |
|------|-------|------------|----------|---------|
| 架构 | MoT 集成 | VLM+Diff 组装 | VLM+Diff 组装 | LLM+双head |
| 交错生成 | ✅ | ❌ | ❌ | ❓ |
| 涌现推理 | ✅ | ❌ | ❌ | ❌ |
| 视频 | ✅（交错） | ❌ | ❌ | ✅（3D VAE） |

## Benchmark

- GenEval: 0.88（使用 LLM rewriter）
- MMMU: 55.3
- MM-Vet: 67.2
- WISE: 0.70（+Self-CoT 从 0.52）
- IntelligentBench: 55.3（+Self-CoT 从 44.9）

## 在 Wiki 中的关联

- 比较：[[Wiki/Comparisons/统一多模态模型架构比较]]
- 主题：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 问答：[[Wiki/Questions/BAGEL 图文交错生成能力]]

---
type: source
status: processed
source_file: "[[Clippings/EmotiCrafter Text-to-Emotional-Image Generation based on Valence-Arousal Model.md]]"
title: "EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model"
site: "arXiv (2501.05710v3)"
author: "Shengqi Dang, Yi He, Long Ling, Ziqing Qian, Nanxuan Zhao, Nan Cao"
published: "2025-01"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [emotion, image-generation, valence-arousal, continuous-emotion, diffusion]
aliases: [EmotiCrafter, C-EICG]
---

# EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model

同济大学 + 上海创新研究院 + Adobe Research，2025。

## 核心结论

- 提出 **连续情感图像内容生成（C-EICG）** 任务，首次用连续 Valence-Arousal (V-A) 值替代离散情感标签来控制图像生成。
- 提出 **EmotiCrafter**，通过 emotion-embedding mapping network 将 V-A 值融入文本特征，注入 SDXL。
- 设计新损失函数：scaled residual learning（放大情感-中性残差）+ V-A density weighting（处理数据不平衡）。
- 在情感准确率（V/A-Error）和连续性（LPIPS-Continuous）上优于所有基线，尤其是连续情感过渡的平滑性远超 GPT-4+SDXL。

## 关键方法

### Emotion-Embedding Network

1. **V-A Encoder**：两个独立 MLP 分别编码 Valence 和 Arousal 值 → $e_v$, $e_a$
2. **Emotion Injection Transformer (EIT)**：修改 GPT-2（12 层，去因果掩码），每层通过 self-attention + cross-attention($e_v$) + cross-attention($e_a$) + FFN 逐步注入情感
3. **残差预测**：输出预测 emotional-neutral 残差，加到原始特征上得到 emotional prompt feature
4. 用 SDXL 的 cross-attention 接收注入的情感特征生成图像

### 损失函数

$$\mathcal{L} = \frac{1}{n}\mathbb{E}\left(\frac{1}{d(v,a)}\|\hat{f}_e - f_e^t\|^2\right)$$

- **Scaled Residual**：$f_e^t = f_n + \alpha(f_e - f_n)$，$\alpha=1.5$，放大情感信号
- **V-A Density Weighting**：用 KDE 估计 V-A 空间密度 $d(v,a)$，低密度区域权重更高，缓解训练数据不平衡

### 训练数据

从 OASIS、EMOTIC、FindingEmo 收集 39,843 张人工标注 V-A 值的图像，GPT-4 生成中性/情感 prompt 对，众包验证。

## 实验结果

- V-Error: 1.510 (最优), A-Error: 1.828 (最优)
- 4 种 baseline（Cross Attention、Time Embedding、Textual Inversion、GPT-4+SDXL）中，仅 EmotiCrafter 和 GPT-4+SDXL 能产生可区分的情感变化
- **连续性**：LPIPS-Continuous 0.220 vs GPT-4+SDXL 0.361，情感过渡更平滑
- 用户调研：V/A Ranking 一致性、Error、情感一致性和平滑性均显著优于基线
- 支持 V-A 粒度 0.2 的精细控制

## 限制

- Arousal 控制比 Valence 更难（与视觉情感分析文献一致）
- 常生成含人物的图像，即使 prompt 未提及（训练数据限制）
- 有时修改 prompt 语义以对齐情感（可加语义保持项解决）

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/情感计算与图像生成]]、[[Wiki/Concepts/Valence-Arousal 情感模型]]
- 与 [[Wiki/Sources/EmoEdit]] 的关键区别：EmoEdit 用离散情感词 + adapter 做编辑，EmotiCrafter 用连续 V-A 值 + EIT 做生成
- V-A 连续空间 vs 离散类别：更细粒度但 arousal 控制更难

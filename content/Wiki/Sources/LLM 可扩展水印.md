---
type: source
status: complete
created: '2026-05-07'
updated: '2026-05-07'
tags: [watermarking, llm, safety, detection, google]
sources:
  - "[[Clippings/Scalable watermarking for identifying large language model outputs]]"
---

# LLM 可扩展水印

## 基本信息

- **标题**: Scalable watermarking for identifying large language model outputs
- **作者**: Sumanth Dathathri, Abigail See, Sumedh Ghaisas, Po-Sen Huang, Rob McAdam, Johannes Welbl, Vandana Bachani, Alex Kaskasoli, Robert Stanforth, Tatiana Matejovicova, Jamie Hayes, Nidhi Vyas, Majd Al Merey, Jonah Brown-Cohen, Rudy Bunel, Borja Balle, Taylan Cemgil, Zahra Ahmed, Kitty Stacpoole, Ilia Shumailov, Ciprian Baetu, Sven Gowal, Demis Hassabis, Pushmeet Kohli
- **机构**: Google DeepMind
- **发表**: Nature, 2024 年 10 月 23 日在线发表
- **DOI**: [10.1038/s41586-024-08025-4](https://doi.org/10.1038/s41586-024-08025-4)
- **开源代码**: [google-deepmind/synthid-text](https://github.com/google-deepmind/synthid-text)

## 核心论点

大语言模型生成的合成文本已经高质量到难以与人类写作区分，这对信息生态系统的安全构成挑战。水印（watermarking）可以在生成阶段嵌入统计签名，用于事后识别 LLM 输出，且检测过程不需要访问原始 LLM。本文提出的 **SynthID-Text** 是首个在大规模生产系统中实际部署的生成式文本水印方案，已被应用于 Gemini 和 Gemini Advanced，服务数百万用户。

## 关键技术方法

### 生成式水印框架

SynthID-Text 属于**生成式水印**（generative watermarking），在自回归采样过程中嵌入水印，由三个组件构成：

1. **随机种子生成器**（random seed generator）：使用滑动窗口方法，对最近 $H=4$ 个 token 与水印密钥做哈希，生成每步的随机种子 $r_t$。
2. **采样算法**（sampling algorithm）：本文的核心创新——Tournament sampling。
3. **评分函数**（scoring function）：测量文本中水印签名的强度，输出分数与阈值比较以判定是否为水印文本。

### Tournament 采样（锦标赛采样）

这是本文提出的新型概率采样算法，核心思想是通过多轮锦标赛式淘汰选择在水印函数上得分较高的 token：

- **第一阶段**：从 LLM 分布中采样 $M = 2^m$ 个候选 token（$m$ 为锦标赛层数，默认 $m=30$）。
- **第二阶段**：将候选 token 配对，每对中使用第 $\ell$ 层的伪随机水印函数 $g_\ell$ 打分，高分者晋级；重复此过程直到决出最终赢家作为输出 token。
- **水印函数**：每层使用独立的伪随机函数 $g_\ell(x_t, r_t)$，为每个候选 token 分配一个 0 或 1 的分数（Bernoulli(0.5) 分布）。

### 无失真性保证

SynthID-Text 提供可配置的无失真性（non-distortion），论文明确了从弱到强的定义层次：

- **单 token 无失真**：输出 token 的边际分布等于原始 LLM 分布。
- **单序列无失真**：通过**重复上下文掩码**（repeated context masking）技术，当同一上下文窗口已用于水印时跳过水印步骤，避免重复偏置导致的文本质量退化。
- 默认配置为单序列无失真，平衡质量保持与可检测性。

也可配置为**有失真模式**（distortionary），通过增加每轮匹配的候选数（多于 2 个）提升可检测性，代价是一定的文本质量损失。

### 与推测采样的集成

生产系统常使用**推测采样**（speculative sampling）加速 LLM 推理——由小型草稿模型提议 token，大型目标模型验证。本文提出两种水印与推测采样的结合算法：

1. **高检测性水印推测采样**：保持水印检测率，但可能降低推测采样效率。
2. **快速水印推测采样**：保持推测采样效率，且对无失真水印理论上保证接受率不变。配合学习的贝叶斯评分函数可提升检测性。

### 检测过程

检测时不需要访问 LLM，只需 token 化文本、水印密钥和随机种子生成器。计算文本在所有时间步和所有锦标赛层上的平均 $g$ 值作为分数：

$$\text{Score}(x) = \frac{1}{mT} \sum_{t=1}^{T} \sum_{\ell=1}^{m} g_\ell(x_t, r_t)$$

还提出了加权均值评分、频率派假设检验评分和参数化贝叶斯评分函数等变体。

## 主要结果

### 质量保持

- **大规模在线实验**：在 Gemini 生产系统中对约 2000 万条水印和非水印响应进行对比，点赞率差异仅 0.01%，踩 rate 差异仅 0.02%，均统计不显著。
- **受控人工评估**：3000 个 ELI5 问题的 Gemma 7B-IT 回复，语法/连贯性、相关性、正确性、有用性和总体质量五个维度均无显著偏好差异。
- **自动评估**：困惑度和标准基准测试无显著差异。

### 检测性能

- **非失真类别**：SynthID-Text 在相同文本长度下优于 Gumbel 采样基线，在低熵设置（低温度、大模型）中优势更明显。
- **失真类别**：失真 SynthID-Text 相比 Soft Red List 提供更优的检测率-质量权衡曲线。
- 支持**选择性预测机制**（selective prediction），对不确定样本弃权以实现目标低错误率。

### 计算效率

- Gemma 7B-IT 在 4 个 TPU v5e 上：原始生成 15.527 ms/token，30 层 Tournament 采样仅增至 15.615 ms/token，延迟增加 0.57%。
- 快速水印推测采样可保持与无水印时几乎相同的接受率和整体延迟。

### 多语言一致性

附录实验表明 SynthID-Text 在不同语言上表现一致，优于事后检测器（post hoc detector）在未训练语言上的表现。

## 局限性

1. **需要协调**：生成式水印要求 LLM 服务提供方主动实施水印；不实施水印的第三方无法被检测。
2. **开源模型挑战**：去中心化部署的开源模型难以强制水印。
3. **对抗攻击脆弱性**：面临窃取（stealing）、伪造（spoofing）和擦除（scrubbing）攻击，尤其是 LLM 改写（paraphrasing）会削弱水印——尽管这通常会显著改变文本。
4. **低熵场景效果有限**：当 LLM 分布熵很低（如模型对特定提示几乎确定输出）时，锦标赛采样可用于嵌入水印的"余量"减少。
5. **非完整方案**：水印与其他检测方法（事后检测、检索匹配）互补，不能单独解决 AI 文本检测问题。

## 与相关工作的关系

| 相关工作 | 关系 |
|---|---|
| Gumbel 采样 (Aaronson & Kirchner 2022; Kuditipudi et al. 2024) | 非失真水印基线；SynthID-Text 在相同设置下提供更好检测率 |
| Soft Red List (Kirchenbauer et al. 2023) | 失真水印基线；SynthID-Text 提供更优的检测率-质量权衡 |
| 重复上下文掩码 (Hu et al. 2024) | SynthID-Text 采用此技术实现序列级无失真 |
| 推测采样 (Chen et al. 2023) | 本文首次将生成式水印与推测采样结合 |
| 事后检测 (DetectGPT, Ghostbuster, Binoculars 等) | 互补方案；事后检测不需要生成端配合但性能不稳定 |
| 数据驱动水印 (Gu et al. 2022) | 需触发短语、仅对特定使用场景有效，本文方法更通用 |
| 水印窃取攻击 (Jovanovic et al. 2024) | 本文讨论的已知威胁之一 |
| 改写逃避 (Zhang et al. 2024) | 本文承认此为已知弱点，附录中有相关评估 |

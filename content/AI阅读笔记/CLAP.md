---
title: 'CLAP: Learning Audio Concepts From Natural Language Supervision'
authors:
  - Benjamin Elizalde
  - Soham Deshmukh
  - Mahmoud Al Ismail
  - Huaming Wang
institutions: Microsoft
aliases:
  - CLAP
  - CLAP论文
date: '2026-04-30'
publish_date: 2022-06
venue: arXiv:2206.04769
tags:
  - 论文
  - 音频理解
  - 对比学习
  - 多模态
  - zero-shot
url: 'https://arxiv.org/abs/2206.04769'
code: 已开源
rating: ⭐⭐⭐⭐
status: Read
---
**一句话总结**：本文提出 CLAP（Contrastive Language-Audio Pretraining），将 CLIP 的对比学习范式迁移到音频领域，仅用 128K 音频-文本对训练，在 16 个下游任务上实现了 Zero-Shot SOTA，证明了自然语言监督可以成为音频基础模型的可扩展学习路径。

*Figure 1: CLAP 联合训练音频和文本编码器，通过对比学习将音频和文本描述映射到联合多模态空间。推理时直接计算 audio-text 余弦相似度实现 Zero-Shot 分类。*

---

## Intro

### Motivation

主流音频分析模型采用「一个类别标签对应多条录音」的监督范式，只能预测预定义类别，缺乏灵活性。自监督学习（SSL）虽可用无标签音频预训练，但排除了自然语言中的语义知识。计算机视觉领域已成功用自然语言监督学习图像表征（CLIP、Florence），但音频领域的同思路尚处于早期（Wav2CLIP、AudioCLIP 仅从 CLIP 蒸馏，未直接用自然语言训练）。

### 贡献

1. 提出 CLAP 模型，用 128K 音频-文本对训练，建立音频和自然语言的联合多模态空间
2. 实现 Zero-Shot 音频分类——无需训练阶段和预定义类别，推理时可灵活预测任意用户输入的类别
3. 在 8 个领域 16 个下游任务上验证 Zero-Shot 泛化能力，建立 Zero-Shot SOTA；监督微调后在 5 个任务上达到 SOTA

---

## Method 核心方法

### 1. 对比语言-音频预训练

CLAP 完全遵循 CLIP 的双塔对比学习范式：

- **音频编码器**：CNN14（80.8M 参数），在 AudioSet 上预训练，embedding 维度 2048
- **文本编码器**：BERT-base-uncased（110M 参数），取 [CLS] token 的最后一层 hidden state（维度 768）
- 两个编码器各接一个可学习线性投影层映射到 1024 维联合空间
- 对称交叉熵损失：$L = 0.5 * (l_{text}(C) + l_{audio}(C))$，其中 $C = \tau * (E_t \cdot E_a^T)$
- 温度参数 $\tau$ 可学习，初始值 0.007，logits 被 clip 到最大 100

### 2. 数据和训练设置

- 训练数据：来自 4 个数据集共 128,010 对（FSD50K 36.8K + ClothoV2 29.6K + AudioCaps 44.3K + MACS 17.3K）
- 音频预处理：44.1 KHz 采样率，64 Mel bins（50-8000 Hz），随机截取/填充到 5 秒
- 训练：两个编码器均不冻结，Adam 优化器，初始 lr=1e-3，plateau 衰减，40 epochs，8-24 块 16GB V100

### 3. Zero-Shot 分类

用模板 prompt（如 "This is a sound of [class label]"）构造文本输入，计算测试音频与所有类别文本的余弦相似度，经 softmax/sigmoid 得到分类概率。

---

## 实验/评估/结果

### Zero-Shot 音频分类（16 个下游任务）

| 任务 | CLAP Acc | 对比方法 | 提升 |
|------|---------|---------|------|
| **ESC50**（环境声音） | **82.6%** | 人类水平 81%, AudioCLIP 69% | +12% |
| **US8K**（城市声音） | **73%** | AudioCLIP 65% | +8% |
| **FSD50K**（多标签） | **30.24% mAP** | Wav2CLIP 3% | +27% |
| **GTZAN 音乐/语音** | **100%** | 超越所有监督模型 | - |

在所有任务上均好于 random baseline，声音事件分类域表现最好。

### 监督微调

5 个任务达到 SOTA：GTZAN 音乐语音 100%、GTZAN 音乐流派 91.3%、Mridangam Stroke 97.94%、Mridangam Tonic 95.34%、Vocal Sound 97.95%。

### 消融：冻结策略

| 策略 | Avg ZS Score |
|------|-------------|
| 冻结两者 | 0.281 |
| 只冻音频编码器 | 0.282 |
| 只冻文本编码器 | **0.311** |
| 都不冻 | **0.327** |

*冻结文本编码器比冻结音频编码器效果好——选好音频编码器即可转化为零样本分类器。*

---

## 结论

CLAP 证明了自然语言监督可以成为音频理解的可扩展范式。仅用相当于 CV 模型 0.001% 的数据量，就在 Zero-Shot 场景下超越了此前最好的方法，并在 5 个监督任务上达到 SOTA。这为构建可泛化到广泛任务的音频基础模型提供了可行路径。

---

## 思考

### 优点

1. **简约而有效的范式迁移**：将 CLIP 的双塔对比学习思路干净地移植到音频域，没有引入额外复杂机制。这证明了对比语言-信号预训练范式的跨模态普适性。

2. **小数据下的强泛化**：仅 128K 对（CLIP 用 400M 对），就能在 16 个下游任务上实现有意义的 Zero-Shot 性能。说明音频域的语义空间比视觉域更紧凑，或者说自然语言描述已足够覆盖常见音频概念的语义结构。

3. **有启发的消融实验**：冻结文本编码器优于冻结音频编码器的发现，暗示了预训练 BERT 已有强大的语义归纳偏置，音频编码器则主要需要适配这个语义空间。这对后续工作选择训练策略有指导意义。

4. **Prompt 敏感性的系统分析**：提供了不同 prompt 模板的定量对比，为音频 Zero-Shot 的 prompt 设计提供了经验指导。

### 缺点与待解决问题

1. **数据规模与领域覆盖严重不足**：128K 对中几乎没有人类语音的情感/内容描述，导致在情感识别和关键词识别等语音任务上表现很弱（仅略好于 random）。论文已认识到这一点，但未给出解决方案。

2. **训练成本披露不完整**：缺乏总 GPU 小时等可比较的训练成本信息。

3. **架构保守**：CNN14 + BERT-base 在 2022 年已非最优选择。后续工作如 CLAPv2 改用 HTS-AT + RoBERTa 取得了更好效果。这里更多是概念验证而非最优实现。

4. **与视觉域的对比不够深入**：论文指出音频和视觉域在数据规模需求上的巨大差异（128K vs 400M），但没有深入分析原因。这实际上是理解不同模态语义密度和对比学习数据效率的关键问题。

### 与已有 Wiki 的连接

- 关联概念：[[CLIP]]、[[对比学习]]、[[Zero-Shot Learning]]
- 后续工作：CLAPv2（更大数据、更强编码器）、WavCaps（自动音频描述数据）、AudioLDM（用 CLAP 做生成条件）

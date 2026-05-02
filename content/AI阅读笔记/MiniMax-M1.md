---
title: "MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention"
authors:
  - MiniMax
institutions: MiniMax
aliases:
  - MiniMax-M1
  - M1
date: '2026-04-30'
publish_date: 2025-06
venue: 'arXiv:2506.13585'
tags:
  - 论文
  - LLM
  - 推理模型
  - Lightning Attention
  - 混合注意力
  - CISPO
  - RL-Scaling
  - MoE
url: 'https://arxiv.org/abs/2506.13585'
code: https://github.com/MiniMax-AI/MiniMax-M1
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：MiniMax-M1 是全球首个开源大规模混合注意力推理模型（456B 总参数/45.9B 激活），基于 MiniMax-Text-01 的 hybrid-MoE 架构（每 7 个 lightning attention 块后跟 1 个 softmax attention 块），提出 CISPO RL 算法（裁剪 importance sampling 权重而非 token 更新），仅用 512 H800 GPU / 3 周 / $0.53M 完成完整 RL 训练，在 100K 生成长度时计算量为 DeepSeek R1 的 25%。

![](https://arxiv.org/html/2506.13585v1/x1.png)

*Figure 1: 左：benchmark 性能对比；右：推理 FLOPs 随生成长度缩放曲线。*

---

## Intro

### Motivation

推理模型的 success 源于 test-time compute 的扩展，但传统 softmax attention 的 $O(L^2)$ 复杂度制约了推理长度的持续增长。现有线性注意力、SSM 等方法未被大规模推理模型验证。

### 贡献

1. **首个开源大规模 hybrid-attention 推理模型**：lightning attention（线性注意力）大幅降低长推理计算复杂度
2. **CISPO 算法**：裁剪 IS 权重而非 token 更新，保留所有 token 梯度贡献（比 DAPO 快 2x）
3. **RL 训练的精度与稳定性方案**：FP32 LM head、AdamW 超参数调优、重复检测
4. **多阶段长度扩展策略**：40K → 80K 渐进式窗口扩展
5. **极低 RL 成本**：$0.53M，3 周

---

## Method 核心方法

### 1. 继续预训练 + SFT

- 从 MiniMax-Text-01 继续训练 7.5T tokens（STEM/Code/Book 70%）
- 长上下文平滑扩展：32K → 1M（4 阶段），解决 hybrid 架构下前后层衰减率差异导致的梯度爆炸
- SFT 冷启动：60% 数学和代码数据

### 2. CISPO 算法（核心创新）

**问题**：PPO/GRPO 的 token clipping 丢弃了低概率但关键的 token（如「However」「Wait」「Aha」——推理路径的分叉点）

**CISPO 解决方案**：

$$
\mathcal{J}_{\text{CISPO}}(\theta)=\mathbb{E}\left[\frac{1}{\sum|o_i|}\sum_{i}\sum_{t}\texttt{sg}(\hat{r}_{i,t}(\theta))\hat{A}_{i,t}\log\pi_{\theta}(o_{i,t})\right]
$$

其中 $\hat{r}_{i,t}(\theta)=\text{clip}(r_{i,t}(\theta), 1-\epsilon^{IS}_{low}, 1+\epsilon^{IS}_{high})$

- 裁剪 IS 权重而非 token 梯度，保留所有 token 的梯度贡献
- 在 Qwen2.5-32B 验证：比 DAPO 快 2x 达到相同性能
- 无 KL penalty（与 DAPO / R1 一致）

![](https://arxiv.org/html/2506.13585/x2.png)

*Figure 2: CISPO 与 GRPO、DAPO 在 AIME 2024 上的对比（Qwen2.5-32B-base）。CISPO 显著优于两者，以 50% 训练步数达到 DAPO 同等性能。*

### 3. Hybrid Attention RL 挑战与解决

**精度失配**：训练和推理 kernel 之间概率差异大（相关系数从 ~0.9x 修复到 0.99x）
- 根因：LM head 输出层激活值过大
- 修复：LM head 精度提升到 FP32

**优化器超参敏感**：默认配置（betas=(0.9,0.999), eps=1e-8）不收敛
- 梯度量级范围 1e-18 到 1e-5（多数 < 1e-14）
- 修复：betas=(0.9, 0.95), eps=1e-15

**重复检测**：连续 3,000 tokens 概率 > 0.99 则提前终止

### 4. 多阶段长度扩展

40K → 48K → 56K → 64K → 72K → 80K 渐进式窗口扩展，依据 perplexity 收敛和 99th 分位输出长度判断是否进入下一阶段。

**模式坍缩（Pattern Collapse）**：后期序列退化为乱码
- 原因：负样本长度增长快于正样本，后期段严重负梯度累积
- 修复：(1) 重复检测 + 提前终止 (2) sample-level loss + token-level 归一化 (3) 降低梯度裁剪阈值和 $\epsilon^{IS}_{high}$

### 5. RL 数据

- 数学推理：近 50K 竞赛级问题（pass@10 过滤 0~0.9）
- 逻辑推理：53K 样本（SynLogic 框架，41 种任务）
- 竞赛编程：30K（LLM 生成测试用例）
- SWE：数千真实 GitHub issues/PRs + 容器化沙箱
- 通用任务：25K（GenRM 奖励模型，含 ground-truth 和开放域）

---

## 实验/评估/结果

### 主结果

![](https://arxiv.org/html/2506.13585/x4.png)

*Figure 4: RL 训练过程中 AIME 和 LiveCodeBench 的 accuracy 与生成长度变化。accuracy 和生成长度随训练同步增长，AIME 2024 从 68% 升至 80%，平均生成长度超 20K tokens。*

| Benchmark | M1-80k | DS-R1 | DS-R1-0528 | Qwen3-235B | o3 | Gemini 2.5 Pro |
|-----------|--------|-------|------------|------------|-----|---------------|
| AIME 2024 | 86.0 | 79.8 | 91.4 | 85.7 | 91.6 | 92.0 |
| AIME 2025 | 76.9 | 70.0 | 87.5 | 81.5 | 88.9 | 88.0 |
| LiveCodeBench | 65.0 | 55.9 | 73.1 | 65.9 | 75.8 | 77.1 |
| SWE-bench Verified | 56.0 | 49.2 | 57.6 | 34.4 | 69.1 | 67.2 |
| TAU-Bench airline | 62.0 | - | 53.5 | 34.7 | 52.0 | 50.0 |
| OpenAI-MRCR (1M) | 56.2 | - | - | - | - | 58.8 |
| LongBench v2 | 61.5 | 58.3 | 52.1 | 50.1 | 58.8 | 65.0 |
| GPQA Diamond | 70.0 | 71.5 | 81.0 | 71.1 | 83.3 | 86.4 |

### 推理效率

| 模型 | 生成长度 64K | 生成长度 100K |
|------|-------------|-------------|
| DS-R1 (softmax) | 100% FLOPs | 100% FLOPs |
| M1 (hybrid) | < 50% FLOPs | ~25% FLOPs |

### 1M 上下文能力

- OpenAI-MRCR (1M): 56.2（vs Gemini 2.5 Pro 58.8）
- 128K MRCR: 73.4（vs DS-R1 35.8）

---

## 结论

MiniMax-M1 验证了 hybrid-attention 架构在大规模推理模型上的可行性。CISPO + 精度/稳定性方案使 RL 训练成本仅为 $0.53M。在长上下文和 agentic tool-use 任务上表现突出，但在数学/代码竞赛上仍落后于最新 DS-R1-0528。

---

## 思考

### 优点

1. **Hybrid-attention 路线被首次大规模验证**：lightning attention 在 456B 推理模型上成功运行，证明线性注意力不只是一种学术探索
2. **CISPO 算法简洁有效**：从「clipping token update」到「clipping IS weight」的思路转变，保留了关键低概率 token 的梯度，2x 加速
3. **RL 训练的工程细节价值高**：FP32 LM head、eps=1e-15、重复检测等对社区实践有直接参考意义
4. **1M 上下文能力在推理模型中独一无二**：8x DS-R1 的上下文，大幅领先所有开源推理模型
5. **工具调用能力突出**：TAU-Bench airline 62.0 超越 o3 和 Gemini 2.5 Pro

### 缺点

1. **数学和代码竞赛能力并非顶级**：AIME 2025 仅 76.9（vs DS-R1-0528 的 87.5），更接近上一代模型
2. **模型规模带来的部署负担**：456B 总参数，单卡难以部署
3. **Lightning attention 的生态支持有限**：需要在 vLLM/Transformers 中添加专门支持
4. **模式坍缩问题说明长上下文 RL 仍有根本性挑战**：需要多种启发式方法组合解决
5. **GenRM 的长度偏差问题**：虽设计了在线监控和重校准，但本质上仍是脆弱的多步流程

---
type: source
status: complete
created: '2026-05-07'
updated: '2026-05-07'
tags: [vlm, qwen, alibaba, multimodal, vision-language, reasoning, long-context, moe]
sources:
  - "[[Clippings/Qwen3-VL Technical Report]]"
---

# Qwen3-VL 技术报告

## 1. 基本信息

- **标题**: Qwen3-VL Technical Report
- **作者**: Shuai Bai et al.（Qwen Team，阿里巴巴）
- **机构**: 阿里巴巴集团
- **年份**: 2025
- **论文来源**: [arXiv:2511.21631](https://arxiv.org/abs/2511.21631)
- **模型家族**: 4 个 Dense 变体（2B/4B/8B/32B）+ 2 个 MoE 变体（30B-A3B/235B-A22B）
- **许可**: Apache 2.0

## 2. 核心论点

1. **纯文本能力不退化**: 多模态训练不应削弱 LLM 的语言能力；Qwen3-VL 在多个纯文本基准上达到甚至超过同系列纯文本模型，实现了视觉与语言能力的真正融合。

2. **原生长上下文支持**: 原生支持 256K token 的交错多模态上下文（文本+图像+视频），在长文档和长视频理解任务上表现突出，30 分钟视频内 Needle-in-a-Haystack 达到 100% 准确率。

3. **多模态推理领先**: 在 MMMU、MathVista、MathVision 等综合评测中取得 SOTA 或接近 SOTA 的表现，Thinking 模式在数学和逻辑推理任务上优势明显。

4. **架构三重升级**: Interleaved MRoPE（平衡频谱的位置编码）、DeepStack（多层视觉特征融合）、文本时间戳（替代绝对时间编码），三项改进共同提升了时空建模能力。

5. **Dense + MoE 全尺寸覆盖**: 从 2B 边缘部署到 235B-A22B 旗舰模型，覆盖不同延迟-质量权衡需求，且 MoE 架构在同等 token 预算下表现优异。

## 3. 关键技术方法（相比 Qwen2.5-VL 的改进）

### 3.1 架构改进

- **Interleaved MRoPE**: Qwen2.5-VL 的 MRoPE 将嵌入维度分为时间(t)、水平(h)、垂直(w)三组，导致频谱不均衡。Qwen3-VL 改为交错分配，使每个时空轴在低频和高频段均匀表示，显著改善长视频的位置建模。

- **DeepStack 跨层融合**: 从 ViT 中间层提取三个不同层级的视觉特征，通过专用 merger 投影后直接注入 LLM 前三层的隐藏状态，保留从低级到高级的丰富视觉信息，不增加上下文长度。

- **文本时间戳替代绝对时间编码**: Qwen2.5-VL 用 MRoPE 绝对时间做视频对齐，长视频产生过大且稀疏的时间位置 ID。Qwen3-VL 改用显式文本时间戳 token（如 `<3.0 seconds>`），同时训练秒格式和 HMS 格式，更精确地捕获时序信息。

- **平方根重加权损失**: 从逐样本损失改为平方根归一化的逐 token 损失，更好地平衡文本和多模态数据的训练贡献。

### 3.2 视觉编码器

- 基于 **SigLIP-2** 架构继续训练，支持动态输入分辨率，使用 2D-RoPE 和绝对位置嵌入插值。小模型（2B/4B）使用 SigLIP2-Large（300M），其余使用 SigLIP2-SO-400M。
- 消融实验表明自研 Qwen3-ViT 在 OmniBench 等基准上优于原始 SigLIP-2。

### 3.3 训练流程

**预训练（4 阶段）**:
| 阶段 | 目标 | 可训练参数 | Token 预算 | 序列长度 |
|------|------|-----------|-----------|---------|
| S0 | 视觉-语言对齐 | 仅 Merger | 67B | 8K |
| S1 | 多模态预训练 | 全部 | ~1T | 8K |
| S2 | 长上下文预训练 | 全部 | ~1T | 32K |
| S3 | 超长上下文适配 | 全部 | 100B | 262K |

**后训练（3 阶段）**:
1. **SFT**: 32K 上下文训练 + 256K 长上下文扩展；分为 non-thinking 和 thinking（CoT）两种格式
2. **强到弱蒸馏**: 从强教师模型蒸馏到轻量学生模型，使用纯文本数据微调 LLM backbone
3. **强化学习**: 分为 Reasoning RL（数学、代码、逻辑、grounding 等可验证任务，使用 SAPO 算法）和 General RL（指令遵循、偏好对齐、纠正错误先验）

### 3.4 数据创新

- **图像描述**: 使用 Qwen2.5-VL-32B 重标注，语义去重 + 聚类增强低频概念
- **OCR**: 从 10 种语言扩展到 39 种语言，3000 万多样本
- **文档解析**: 统一标注框架 QwenVL-HTML / QwenVL-Markdown
- **Grounding**: 归一化坐标系 [0, 1000]，支持框和点两种模态
- **3D 理解**: 单目图像 3D 框预测，统一虚拟相机坐标系
- **代码**: 多模态代码数据（UI 截图转 HTML/CSS、图像转 SVG、视觉编程挑战等）
- **视频**: 短到长描述合成策略，时序密集标注，长度自适应采样
- **Agent**: GUI 交互数据（桌面/移动/网页）、函数调用轨迹、搜索增强

### 3.5 Thinking with Images

受"用图像思考"启发，通过两阶段训练赋予模型视觉代理能力：think → act → analyze feedback → answer。使用多轮工具集成 RL，包含答案准确性奖励、多轮推理奖励和工具调用奖励。

## 4. 主要结果

### 旗舰模型（235B-A22B）

- **多模态推理**: MathVista mini 85.8、MathVision 74.6、MMMU 80.6，在多个基准上超过 Gemini 2.5 Pro 和 GPT-5
- **通用 VQA**: MMBench-EN 89.3、RealWorldQA 81.3、MMStar 78.7
- **文档理解**: DocVQA 97.1、InfoVQA 89.2、OCRBench 920、MMLongBenchDoc 57.0（SOTA）
- **Grounding**: RefCOCO-avg 92.1、CountBench 93.7、ODinW-13 48.6 mAP
- **空间理解**: EmbSpatialBench 84.3、RefSpatialBench 69.9、RoboSpatialHome 73.9，大幅领先 Gemini-2.5-Pro
- **视频理解**: MVBench 76.5、Video-MME 79.2、MLVU 84.3
- **对齐**: HallusionBench 66.7（超过 Gemini/GPT-5/Claude）、MIA-Bench 92.7
- **纯文本推理**: AIME-25 89.7（Thinking）、LiveCodeBench v6 70.1（Thinking），超过 OpenAI o3

### 中等模型（32B / 30B-A3B）

- Qwen3-VL-32B 超过 Gemini-2.5-Flash 和 GPT-5-mini，甚至超过上一代 Qwen2.5-VL-72B
- MoE 变体 30B-A3B 仅激活 3B 参数即可达到竞争力表现

### 小模型（2B/4B/8B）

- Qwen3-VL-8B 在所有小模型基准上取得最高分，AIME-25 Thinking 达到 80.3
- 体现了强到弱蒸馏的有效性

### 长上下文能力

- 256K token 窗口内视频 Needle-in-a-Haystack 100% 准确率
- YaRN 扩展到 1M token（约 2 小时视频）仍保持 99.5% 准确率

## 5. 局限性

1. **评测公平性受限**: 视频评测中各模型输入帧数不一致（Gemini 512、GPT-5 256、Claude 100），可能影响比较的公平性。

2. **3D Grounding 仍有提升空间**: Hypersim 基准上得分仅 11-14，SUN RGB-D 34-39，表明复杂室内 3D 场景理解仍是挑战。

3. **ZeroBench 表现一般**: 旗舰模型在 ZeroBench 上仅 2-4 分，说明极端视觉推理仍有瓶颈。

4. **Agent 任务差距**: OSWorld 得分 38.1（235B），与专用 Agent 系统相比仍有明显差距，GUI Agent 的可靠性和泛化性有待提高。

5. **生成能力缺失**: 报告聚焦于理解任务，未涉及图像/视频生成能力，统一理解-生成架构仍在探索中。

6. **训练成本**: 预训练约 2T+ token，使用最多 10,000 GPU，计算资源需求巨大。

## 6. 与相关工作的关系

- **vs Qwen2.5-VL**: 直接继承者，在架构（Interleaved MRoPE、DeepStack、文本时间戳）、数据（39 种语言 OCR、3D grounding、Agent 数据）和训练策略（平方根损失、4 阶段预训练、RL）上全面升级。Qwen3-VL-32B 已超过 Qwen2.5-VL-72B。

- **vs Gemini 2.5 Pro**: 旗舰模型在多数多模态基准上与之持平或超越，空间理解（RefSpatial、RoboSpatial）大幅领先；纯文本推理 Thinking 模式接近或超过 o3。

- **vs GPT-5**: 在 OCR/文档理解（DocVQA、InfoVQA、ChartQA）上明显领先；数学推理（MathVista、MathVision）也占优；Agent 任务上差距较小。

- **vs Claude Opus 4.1**: 在 HallusionBench、MIA-Bench 等对齐基准上显著领先；文档和 grounding 任务优势明显。

- **DeepStack**: 原始方法（Meng et al., 2024）用于多尺度视觉输入堆叠，Qwen3-VL 将其扩展为从 ViT 中间层提取特征，保留多层级视觉信息。

- **MRoPE**: 源自 Qwen2-VL，本工作通过交错分配解决频谱不均衡问题，参考了 Huang et al. (2025) 的方法。

- **SAPO**: 使用自研的 Soft Adaptive Policy Optimization（Gao et al., 2025）进行 RL 训练，在多种任务和模型规模上表现一致。

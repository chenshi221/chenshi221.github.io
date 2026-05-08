---
type: source
status: complete
created: '2026-05-07'
updated: '2026-05-07'
tags: [image-generation, agent, reinforcement-learning, search, multimodal, GRPO]
sources:
  - "[[Clippings/Gen-Searcher: Reinforcing Agentic Search for Image Generation]]"
---

# Gen-Searcher: Reinforcing Agentic Search for Image Generation

## 基本信息

- **标题**: Gen-Searcher: Reinforcing Agentic Search for Image Generation
- **作者**: Kaituo Feng, Manyuan Zhang, Shawn Chen, Yunlong Lin, Kaixuan Fan, Yilei Jiang, Hongyu Li, Dian Zheng, Chenyang Wang, Xiangyu Yue
- **机构**: MMLab (CUHK), UCLA, UC Berkeley
- **年份**: 2026
- **论文**: arXiv:2603.28767
- **主页**: https://gen-searcher.vercel.app/

## 核心论点

1. **现有图像生成模型受限于冻结的参数知识**: 当前 text-to-image 模型虽然能生成高保真图像，但依赖预训练时获得的静态知识，面对需要外部知识或最新信息的真实场景（如特定地标、公众人物、新产品）表现不佳。
2. **RAG 方法不足以应对复杂知识密集型生成**: 基于静态数据库的检索增强方法受限于覆盖范围和时效性，单轮浅层检索无法满足需要多跳推理的复杂查询。
3. **首个搜索增强图像生成智能体**: Gen-Searcher 是第一个通过训练（而非手工 prompt 工作流）实现多跳网络搜索和推理来收集文本知识与参考图像的搜索增强图像生成智能体。
4. **双重奖励反馈设计解决 RL 训练不稳定问题**: 纯图像奖励因开源生成器的高方差而噪声过大，纯文本奖励又忽略了实际生成效果；结合两者提供更稳定、信息更丰富的训练信号。
5. **搜索策略可跨生成器迁移**: 虽然 RL 训练阶段使用 Qwen-Image 作为生成器，学到的搜索策略可直接迁移到 Seedream 4.5 和 Nano Banana Pro 等其他模型，无需额外训练。

## 关键技术方法

### 数据管线

- **文本 prompt 构造**: 通过两种策略生成需要深度搜索的 prompt——(1) 用 Gemini 3 Pro 生成覆盖约 20 个类别（动漫、建筑、艺术、名人、物理、化学等）的多跳搜索密集型 prompt；(2) 将已有深度研究数据集转换为面向图像生成的 prompt。
- **智能体轨迹生成**: 使用 Gemini 3 Pro 配合搜索工具（`search` 文本搜索、`image_search` 图像搜索、`browse` 网页浏览）进行多轮推理，生成包含搜索轨迹、grounded prompt 和参考图像的训练数据。
- **Ground-truth 图像合成**: 使用 Nano Banana Pro 合成最终图像，经 Seed1.8 多维度打分和规则过滤后得到约 17K 高质量样本。

### 训练数据集

- **Gen-Searcher-SFT-10k**: 用于监督微调，教会模型基本的多轮工具调用能力。
- **Gen-Searcher-RL-6k**: 用于智能体强化学习，优化搜索策略和工具调用轨迹。

### KnowGen 基准

- 630 个人工验证样本，分为 Science & Knowledge 和 Pop Culture & News 两个子集。
- **K-Score 评估指标**: 从 faithfulness（0.1）、visual correctness（0.4）、text accuracy（0.4）、aesthetics（0.1）四个维度评估，使用 GPT-4.1 作为评判模型。

### 两阶段训练

1. **SFT 阶段**: 在 Gen-Searcher-SFT-10k 上训练，初始化自 Qwen3-VL-8B-Instruct，学习多轮工具使用、搜索查询、反馈解读、参考图像选择和 grounded prompt 构造。
2. **Agentic RL 阶段**: 在 Gen-Searcher-RL-6k 上使用 GRPO 优化，图像生成器（Qwen-Image-Edit-2509）在训练中保持冻结。

### 双重奖励设计

$$R = (1 - \alpha) R_{\text{image}} + \alpha R_{\text{text}}, \quad \alpha = 0.5$$

- **文本奖励** ($R_{\text{text}}$): 评估输出文本是否包含足够、正确且与生成相关的信息，使用 GPT-4.1 在五级量表上打分。
- **图像奖励** ($R_{\text{image}}$): 使用 K-Score 评估最终生成图像质量。
- 两者互补：文本奖励监督信息收集质量，图像奖励对齐最终生成效果。

## 主要结果

### KnowGen 基准

- Gen-Searcher-8B + Qwen-Image: K-Score 从 14.98 提升至 31.52（+16.54 点）。
- Gen-Searcher-8B + Seedream 4.5: 从 31.01 提升至 47.29（+16.28 点）。
- Gen-Searcher-8B + Nano Banana Pro: 从 50.38 提升至 53.30，达到最优总体结果。
- 增益主要来自 visual correctness 和 text accuracy 两个维度。

### WISE 基准

- Gen-Searcher-8B + Qwen-Image 整体分数从 0.62 提升至 0.77（+0.15），尤其在 Chemistry 类别从 0.40 跃升至 0.75。

### 消融实验

| 方法 | KnowGen K-Score |
|---|---|
| Qwen-Image (基线) | 14.98 |
| + prompt 工作流 | 22.91 |
| + Gen-Searcher-SFT | 28.15 |
| + Gen-Searcher (无文本奖励) | 29.59 |
| + Gen-Searcher (无图像奖励) | 29.36 |
| + Gen-Searcher (完整) | 31.52 |

- SFT 相比手工 prompt 工作流有显著优势；RL 在 SFT 基础上进一步提升；双重奖励缺一不可。

### 跨生成器迁移性

- 在 Qwen-Image 上训练的搜索策略可零样本迁移到 Seedream 4.5 和 Nano Banana Pro，无需额外训练。

## 局限性

1. **下游生成器能力仍是瓶颈**: 即使搜索到了正确信息，开源生成器（如 Qwen-Image）仍可能因多主体一致性差、文本渲染能力弱等原因生成不准确的图像。
2. **美学维度偶尔下降**: 整合多个参考图像的视觉信息时，生成器可能无法始终产出构图最理想的图像。
3. **文本奖励与图像奖励的权衡**: 虽然双奖励设计有效，但 $\alpha$ 的选择仍需调参，在 0.3-0.6 范围内表现稳定，但极端值会导致性能下降。
4. **依赖商业模型进行数据构建**: 数据管线依赖 Gemini 3 Pro 和 Nano Banana Pro 等闭源模型，开源复现成本较高。
5. **搜索工具的覆盖范围**: 当前仅支持文本搜索、图像搜索和网页浏览三类工具，未涉及视频或其他模态的检索。

## 与相关工作的关系

- **与 RAG 图像生成方法的区别**: Re-Imagen 等方法依赖静态数据库的单轮浅层检索，Gen-Searcher 通过多跳网络搜索和主动推理实现更深层的知识获取。
- **与 prompt 工作流方法的区别**: IA-T2I、Mind-Brush 等基于手工设计的 prompt 工作流缺乏训练，搜索行为脆弱且次优；Gen-Searcher 通过 SFT + RL 训练自适应地规划搜索步骤。
- **与智能体 RL 工作的关系**: 延续 ARPO、GiGPO、Vision-DeepResearch 等智能体强化学习范式，首次将其应用于知识密集型图像生成场景。
- **与 GRPO 的关系**: 采用 DeepSeek-R1 提出的 GRPO 算法进行策略优化，结合双重奖励反馈适配图像生成任务的特殊挑战。

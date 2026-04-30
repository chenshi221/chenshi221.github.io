---
type: source
status: processed
source_file: "[[Clippings/OmniGen2 Towards Instruction-Aligned Multimodal Generation.md]]"
title: "OmniGen2: Towards Instruction-Aligned Multimodal Generation"
site: "arXiv (BAAI)"
author: "Chenyuan Wu, Jiahao Wang, Pengfei Zheng, Shitao Xiao, Zheng Liu et al."
published: "2025-06"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [unified-multimodal, omnigen, instruction-alignment, grpo]
aliases: [OmniGen2]
---

# OmniGen2: Towards Instruction-Aligned Multimodal Generation

## 核心结论

- OmniGen2 采用**解耦的双 Transformer 架构**：VLM（Qwen2.5-VL-3B）理解指令，Diffusion Transformer 生成图像，通过 VLM 的变长隐状态连接两者，避免信息瓶颈。
- 提出 **Omni-RoPE** 三维位置编码（实例 ID, h, w），解决多图像场景中的空间一致性问题。
- 采用**渐进式多任务强化学习（GRPO）** 进行指令对齐，按 T2I → Edit → IC 顺序训练，促进跨任务知识迁移。

## 关键事实

- **数据构建**：利用视频数据构建编辑和上下文生成（in-context generation）训练样本，解决高质量编辑数据稀缺问题。
- **OmniContext 基准**：提出专门评估上下文生成能力的基准，涵盖个体、物体和场景一致性。
- **架构特点**：Diffusion Decoder 直接接收 VLM 全部隐状态（而非固定长度 query tokens），保留完整语义信息。参数在 Diffusion Decoder 中跨模态共享。
- **指令对齐**：GRPO 训练分阶段进行，不同阶段使用不同奖励信号（EditScore 用于编辑、Qwen2.5-VL-72B 用于 IC、GenEval 用于 T2I）。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 关联：[[Wiki/Sources/BAGEL]]（BAGEL 是端到端联合训练；OmniGen2 是解耦 VLM + Diffusion）
- 关联：[[Wiki/Sources/UniWorld-V1]]（两者都使用 VLM + Diffusion 架构，但 OmniGen2 不使用额外 SigLIP 编码器）

## 可能的矛盾或待核实点

- VLM 冻结训练是否会限制生成能力的上限？
- GRPO 分阶段训练中的奖励设计对泛化能力的影响程度。

## 后续问题

- Omni-RoPE 是否可以推广到其他统一模型？
- 解耦架构 vs 集成架构（如 BAGEL）在长期扩展性上孰优孰劣？

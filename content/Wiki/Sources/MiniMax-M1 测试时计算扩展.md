---
type: source
status: processed
source_file: >-
  [[Clippings/MiniMax-M1 Scaling Test-Time Compute Efficiently with Lightning
  Attention]]
title: 'MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention'
site: arxiv
author: MiniMax
published: '2025'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - MiniMax-M1
  - Lightning Attention
  - test-time compute
  - 推理模型
  - CISPO
aliases:
  - MiniMax-M1
---
# MiniMax-M1 测试时计算扩展

## 核心结论

MiniMax-M1 是全球首个开源权重的大规模混合注意力推理模型。基于 MiniMax-Text-01（456B 总/45.9B 激活），结合 Lightning Attention 实现高效测试时计算扩展。相比 DeepSeek R1，在 100K token 生成长度下仅消耗 25% 的 FLOPs。支持 100 万 token 上下文（8 倍于 DeepSeek R1）。提出 CISPO（Clipped Importance Sampling for Policy Optimization）新 RL 算法，RL 训练仅需 512 H800 GPU 三周（$534,700 成本）。

## 关键事实

- 作者：MiniMax，2025
- 架构：Hybrid MoE + Lightning Attention（交替使用标准 attention 和线性 attention）
- 上下文：原生支持 100 万 tokens
- 参数：456B 总 / 45.9B 激活
- 发布两个版本：40K 和 80K thinking budget（思考预算）
- CISPO：clip importance sampling weights（而非 token updates），在多种 RL 变体中表现最优
- RL 训练数据：传统数学推理 + sandbox-based 真实软件工程环境

## 方法或论证路径

- Lightning Attention 是线性复杂度注意力（O(N) vs O(N^2)），适合极长上下文
- 混合设计：在需要精确 attention 的层使用标准 attention，在可接受的层使用 Lightning Attention
- CISPO vs PPO/GRPO：通过裁剪重要性采样权重而非 token 更新，实现更稳定训练
- Thinking budget：类似 Qwen3 的 thinking budget 机制，允许用户控制推理深度
- 效率对比：以 DeepSeek R1 为基准，展示长文本生成场景下的计算优势

## 与现有 Wiki 的关系

- 关联：[[Wiki/Concepts/测试时计算扩展]]、[[Wiki/Topics/推理增强方法]]
- 关联：[[Wiki/Entities/Kimi 系列模型]]、[[Wiki/Entities/DeepSeek 系列模型]]
- 补充：为推理模型的经济效率提供了新维度——不仅是模型规模效率，更是"长推理效率"

## 可能的矛盾或待核实点

- Lightning Attention 在需要密集信息交互的任务（如多轮对话）上是否有质量损失
- 456B 总参数的模型部署成本（需多卡装载）是否抵消了推理 FLOPs 节省的优势

## 后续问题

- 极长上下文思考（80K+ tokens）的质量天花板在哪里？
- 混合注意力的最优切换比例（全注意 vs 线性注意的层数分配）

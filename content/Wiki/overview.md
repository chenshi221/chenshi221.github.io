---
type: synthesis
status: active
created: '2026-04-29'
updated: '2026-04-30'
tags:
  - meta
---

# 知识库总览

## 研究方向

本知识库覆盖以下主要领域：

- **大语言模型（LLM）基础** — Transformer 架构、Scaling Laws、RLHF 对齐、GPT 系列演进、LoRA/MoE/RAG/NSA 等关键技术
- **国产大模型** — DeepSeek（LLM→V2→Coder-V2→V3→V3.2→R1）、Kimi（k1.5→K2→K2.5→VL）、Qwen3 三条主线
- **扩散模型与生成** — DDPM→Flow Matching→Rectified Flow→FLUX/SD3→Seedream 2/3/4 的理论到工业全链
- **推理增强方法** — CoT → ToT → GoT → 推理模型的完整演进
- **计算机视觉** — ViT/Swin/MLP-Mixer、MAE、YOLO/DINO DETR 目标检测
- **图像编辑与生成** — 指令编辑→统一多模态→推理Agent→图层分解的全栈知识
- **情感计算** — 情感图像编辑、V-A 连续模型、情感偏差审计
- **LLM Agent** — Agent 架构框架、多模态 Agent、Agentic 图像编辑
- **多模态 Embedding 与检索** — MLLM Embedding、全模态检索、工业部署
- **推荐系统** — 生成式推荐（OneRec 系列）、CTR 预估统一化（OneTrans/HyFormer/InterFormer）
- **Benchmark 与评估** — 图像编辑/OCT/Doc 解析/PDF 文档处理 benchmark

## 操作框架

本 vault 由 [[CLAUDE.md]] 定义操作规则。核心工作流：来源摄取 → 知识查询 → Wiki 体检 → 深度分析。

## Wiki 统计

| 指标 | 数量 |
|------|------|
| 已处理来源 | **97** |
| 主题页 | **14** |
| 概念页 | **27**（含深度分析章节） |
| 实体页 | **27**（含批判性评估章节） |
| 比较页 | **11** |
| 问答页 | **2** |
| 未处理 Clippings | **0** ✅ |

## 最新摄取（2026-04-30）

- ✅ 扩散/生成模型基础（6篇）：DDPM → Flow Matching → FLUX Kontext → Seedream 2/3/4
- ✅ 推荐系统全栈（8篇）：OneRec×4 + OpenOneRec + OneTrans + HyFormer + InterFormer
- ✅ 核心基础论文 + 补充（14篇）：LoRA / MAE / MoE 原版 / Switch Transformer / RAG / NSA / DINO DETR / YOLO / DeepSeek-V2 / Coder-V2 / Qwen-Image / MiniMax-M1 / PyTorch / gpt-oss

## 知识体系特征

- 覆盖从 2017（Transformer/MoE）到 2025（Seedream 4.0）的 8 年 AI 发展全景
- 每条知识链都有"事实层 + 分析层"：来源摘要 → 概念/实体 → 主题 → 比较 → 深度分析/批判性评估
- 跨领域连接丰富：推荐系统 ↔ LLM（Scaling Laws），扩散模型 ↔ 图像编辑（Flow Matching），MoE ↔ 国产大模型（历史演进）

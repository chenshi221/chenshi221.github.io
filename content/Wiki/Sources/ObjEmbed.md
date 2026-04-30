---
type: source
status: processed
source_file: '[[Clippings/ObjEmbed Towards Universal Multimodal Object Embeddings]]'
title: 'ObjEmbed: Towards Universal Multimodal Object Embeddings'
site: arXiv 2602.01753
author: 'Shenghao Fu et al. (Sun Yat-sen University, WeChat)'
published: '2024'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - embedding
  - multimodal
  - object-detection
  - region-level
aliases:
  - ObjEmbed 论文
---
# ObjEmbed: Towards Universal Multimodal Object Embeddings

## 核心结论

- 提出 ObjEmbed，首个 MLLM 基的目标级 Embedding 模型，将图像中所有物体编码为独立 embedding。
- 为每个目标生成双 embedding：语义 embedding（object token）+ 定位质量预测 embedding（IoU token），最终匹配分数 = 语义相似度 x 预测 IoU。

## 关键事实

- 来源：中山大学、微信 CV 团队，2024。
- 基座模型：Qwen3-VL-instruct，引入 5 个特殊 token（object、iou、global、local_text、global_text）。
- 一次前向传播编码所有物体（100 个 proposal + 全局图），总序列 <2000 tokens，高效。
- 支持三大任务：(1) 目标检测与指代理解；(2) 局部图像检索；(3) 全局图像检索。
- 在 18 个 benchmark 上表现优异，COCO mAP 53.0%，局部检索比全局方法高约 20 点。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/多模态 Embedding 与检索]]
- 关联：[[Wiki/Concepts/多模态 Embedding 模型]]
- 关联：[[Wiki/Comparisons/多模态 Embedding 模型比较]]

## 后续问题

- IoU 预测的准确性是否受 proposal generator 质量影响？
- 如何扩展到视频帧级物体跟踪？

---
type: source
status: processed
source_file: >-
  [[Clippings/DINO DETR with Improved DeNoising Anchor Boxes for End-to-End
  Object Detection]]
title: >-
  DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object
  Detection
site: arxiv
author: 'Hao Zhang, Feng Li et al. (IDEA & Tsinghua)'
published: '2022'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - DINO
  - DETR
  - 目标检测
  - 端到端
  - 去噪训练
aliases:
  - DINO
---
# DINO DETR 目标检测

## 核心结论

DINO（DETR with Improved deNoising anchOr boxes）提出了改进的端到端目标检测器，通过三种技术创新大幅提升 DETR 类模型的性能和训练效率：对比式去噪训练、混合查询选择、双层前视（look forward twice）框预测。ResNet-50 + COCO 上 12 epoch 达 49.4 AP，24 epoch 达 51.3 AP（比 DN-DETR 提升 +6.0 和 +2.7 AP）。

## 关键事实

- 作者：Hao Zhang、Feng Li 等（IDEA & 清华 & 港科大），2022
- 三项创新：
  1. 对比式去噪训练（Contrastive DeNoising Training）：同时使用正负两个噪声尺度的 anchor，正样本靠近 ground truth，负样本远离，帮助模型学习拒绝错误预测
  2. 混合查询选择（Mixed Query Selection）：结合可学习查询（content queries）和从 encoder 初始化位置查询（position queries）
  3. 双层前视（Look Forward Twice）：decoder 层之间传递更精细的梯度信息，加速收敛
- SwinL backbone + Objects365 预训练：COCO val2017 63.2 AP，test-dev 63.3 AP
- 模型和数据规模良好 scaling

## 方法或论证路径

- 继承 DETR → Deformable DETR → DN-DETR 的技术栈
- CDN：正负噪声组共享 anchor，正 group 匹配 ground truth，负 group 被推离
- Mixed Query Selection：只从 encoder 选 Top-K 位置初始化位置查询，content 查询仍可学习
- 在 COCO 上 12/24/36 epoch 多尺度训练全面消融实验

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/目标检测基础]]
- 补充：DETR 家族演进的重要节点——DN-DETR→DINO→DINOv2（自监督 vision backbone）

## 可能的矛盾或待核实点

- 与后续 DINOv2（自监督预训练 vision backbone）是完全不同的工作，需注意区分

## 后续问题

- DETR 类模型在实时检测场景（vs YOLO）的差距是否已填补？
- CDN 对比式去噪在检测外任务（分割、跟踪）的适用性

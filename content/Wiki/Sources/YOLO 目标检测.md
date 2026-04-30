---
type: source
status: processed
source_file: '[[Clippings/You Only Look Once Unified, Real-Time Object Detection]]'
title: 'You Only Look Once: Unified, Real-Time Object Detection'
site: arxiv
author: Joseph Redmon et al.
published: '2015'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - YOLO
  - 目标检测
  - 一阶段检测
  - 实时
  - CV
aliases:
  - YOLO
---
# YOLO 目标检测

## 核心结论

Joseph Redmon 等提出 YOLO（You Only Look Once），将目标检测重新定义为回归问题：单个神经网络直接从全图一次性预测 bounding box 和类别概率。YOLO 是第一个实现实时目标检测的统一框架，Base 版本 45 FPS，Fast 版本 155 FPS（当时其他方法的两倍 mAP），且对艺术品的泛化能力远超 DPM 和 R-CNN。

## 关键事实

- 作者：Joseph Redmon、Santosh Divvala、Ross Girshick、Ali Farhadi，2015
- 核心思路：将检测视为回归——输入整张图，输出 S×S 网格上每个格子的 B 个 bounding box + C 个类别概率
- 架构：24 卷积层 + 2 全连接层（受 GoogleNet 启发）
- 速度：45 FPS（Base）、155 FPS（Fast YOLO，更小网络）
- 泛化能力：在 Picasso 和 People-Art 数据集上大幅超越其他检测器

## 方法或论证路径

- 网格划分：图像分为 S×S 网格，每个格子负责检测中心落在其中的物体
- 联合损失：位置损失、置信度损失、分类损失的加权组合
- 局限性：定位误差较大（尤其是小物体），但假阳性（false positive）极少
- 对比：与 R-CNN 的二阶段检测（proposal + 分类）形成"一阶段 vs 二阶段"的方法论对立

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/目标检测基础]]
- 补充：YOLO 是一阶段检测的开创者，后续 YOLOv2~v10 持续演进至今仍是实时检测主流方案

## 可能的矛盾或待核实点

- YOLO 的"小物体检测差"被后续版本逐步修复（特征金字塔、多尺度训练等）
- 论文声称的定位误差劣势在多大程度上是方法固有的 vs 被后续迭代解决的

## 后续问题

- YOLO 与 DETR 类模型在 2024+ 年最新版本上的性能对比
- YOLO 在非 RGB 模态（热成像、红外）中的适用性

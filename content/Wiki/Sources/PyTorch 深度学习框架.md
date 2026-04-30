---
type: source
status: processed
source_file: >-
  [[Clippings/PyTorch An Imperative Style, High-Performance Deep Learning
  Library]]
title: 'PyTorch: An Imperative Style, High-Performance Deep Learning Library'
site: arxiv
author: Adam Paszke et al. (FAIR)
published: '2019'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - PyTorch
  - 深度学习框架
  - define-by-run
  - 自动微分
  - FAIR
aliases:
  - PyTorch
---
# PyTorch 深度学习框架

## 核心结论

PyTorch 论文系统阐述了其设计哲学：(1) 深度模型就是普通 Python 程序；(2) 命令式定义即运行（define-by-run）；(3) 性能与易用性可以兼得。四大设计原则：Pythonic、研究人员优先、实用性能、Worse is Better（宁可简单不完整，也不复杂难维护）。这些原则使得 PyTorch 成为研究社区最主流的深度学习框架。

## 关键事实

- 作者：Adam Paszke、Sam Gross、Soumith Chintala 等（Facebook AI Research & 多机构），2019
- 核心技术特点：
  - 动态计算图（define-by-run），无需静态图组图
  - 自动微分系统（operator overloading）= 任意 Python 程序可微
  - C++ 核心：高性能算子 + Python 绑定
  - CPU-GPU 异步执行（CUDA stream），Python 控制流与 GPU 计算并行
  - 自定义缓存内存分配器（caching allocator）
  - 引用计数即时释放 GPU 内存（非 GC）
  - torch.multiprocessing：为多卡训练优化的进程间共享内存通信

## 方法或论证路径

- "Worse is Better"：接受内部实现简单性而非全面性，以换取快速迭代
- 与 Chainer、DyNet 同属命令式框架流派，但通过 C++ 核心 + CUDA 异步实现了静态图框架的性能
- 在 5 个标准模型（AlexNet、VGG-19、ResNet-50、MobileNet、GNMTv2、NCF）上性能与 MXNet、TensorFlow 竞争
- arXiv 引用率：2018 年起 PyTorch 迅速成为主流（2019 年 ICLR 296 篇提交提及）

## 与现有 Wiki 的关系

- 关联：需要创建 [[Wiki/Concepts/PyTorch 设计哲学]]
- 补充：PyTorch 是几乎所有已处理论文的实验基础，框架设计影响了整个研究社区的工作方式

## 后续问题

- PyTorch 2.0 的 torch.compile（Dynamo + Inductor）如何改变"Worse is Better"的权衡？
- JAX 的崛起是否对 PyTorch 的 define-by-run 范式构成挑战？

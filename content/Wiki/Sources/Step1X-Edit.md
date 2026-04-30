---
type: source
status: processed
source_file: "[[Clippings/Step1X-Edit A Practical Framework for General Image Editing.md]]"
title: "Step1X-Edit: A Practical Framework for General Image Editing"
site: "arXiv (2504.17761v5)"
author: "Step1X-Image Team (StepFun)"
published: "2025-04"
processed: "2026-04-30"
updated: "2026-04-30"
tags: [image-editing, MLLM, diffusion, benchmark, open-source]
aliases: [Step1X-Edit]
---

# Step1X-Edit: A Practical Framework for General Image Editing

StepFun（阶跃星辰），2025。**对标 GPT-4o / Gemini Flash 的开源图像编辑方案**。

## 核心结论

- 提出 **Step1X-Edit**，目标是与 GPT-4o 和 Gemini2 Flash 等闭源模型的编辑能力对标。
- 架构：**MLLM 处理参考图像 + 编辑指令 → latent embedding → 扩散图像解码器**。
- 构建覆盖 **11 种编辑任务**的数据生成管线。
- 引入 **GEdit-Bench**：基于真实用户指令的新型 benchmark。
- 在 GEdit-Bench 上大幅超越开源 baseline，逼近闭源模型。

## 架构特点

### MLLM + Diffusion 混合架构
1. MLLM 编码参考图像 + 用户编辑指令
2. 提取 latent embedding（包含编辑意图和图像理解）
3. 扩散图像解码器从 latent 生成目标图像
4. 联合端到端训练

### 11 种编辑任务
覆盖添加、删除、替换、风格迁移、颜色调整、背景替换、人脸编辑、姿态编辑、文本编辑、场景变换、属性编辑等

### GEdit-Bench
- 基于真实用户指令（非模板生成）
- 评估编辑准确性、图像质量、指令跟随度
- 比现有 benchmark 更贴近实际使用场景

## 在图像编辑演进中的位置

代表了从 InstructPix2Pix 的"纯扩散"到 **MLLM-guided 扩散** 的架构演进趋势。类似思路包括：
- GPT-4o / Gemini Flash（闭源，MLLM 原生编辑）
- Step1X-Edit（开源对标方案）

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 是当前图像编辑技术的前沿代表之一（MLLM + Diffusion 混合架构）
- GEdit-Bench 提供了一个更贴近真实使用的评估标准

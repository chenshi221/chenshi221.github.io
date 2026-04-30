---
type: source
status: processed
site: arXiv
source_file: '[[Clippings/GPT-4o System Card.md]]'
title: GPT-4o System Card
author: OpenAI
published: '2024'
processed: '2026-04-30'
tags:
  - GPT-4o
  - omni-modal
  - safety
  - multimodal
  - OpenAI
---
# GPT-4o System Card

## 核心结论

- **GPT-4o**（o 代表 omni）是 OpenAI 的全模态自回归模型，接受文本、音频、图像、视频任意组合输入，生成文本、音频、图像输出。
- 端到端训练（文本+视觉+音频用同一神经网络），语音响应延迟低至 232ms（平均 320ms），接近人类对话反应时间。
- 在英文和代码方面匹配 GPT-4 Turbo 性能，非英语语言有显著提升，视觉和音频理解能力大幅超越此前模型。
- API 比 GPT-4 Turbo 更快且便宜 50%。

## 关键方法或创新点

- **全模态统一**：首次将文本、视觉和音频真正端到端训练在一个模型中（而非多个模型拼接组合）。
- **Preparedness Framework** 安全评估：包含第三方独立评估，覆盖网络安全、生物威胁、说服力、模型自主性等风险维度。
- **语音安全**（speech-to-speech）：重点评估语音模态带来的新型风险，包括说话者识别、语音生成等。
- 遵循白宫自愿 AI 安全承诺进行系统评估和公开披露。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Concepts/GPT 系列模型]]、[[Wiki/Topics/大语言模型基础]]
- GPT-4o 是 GPT 系列从「多模态输入文本输出」到「全模态输入输出」的关键一步。
- 统一的端到端多模态训练理念与 [[Wiki/Topics/扩散模型图像编辑与生成]] 中的 UMM 模型思想一致。
- 在图像编辑方面，GPT-4o 的图像生成能力激发了多篇对标研究（如 Step1X-Edit、DreamOmni2 等）。

## 局限或注意事项

- System Card 侧重安全评估，技术架构细节有限（如模型规模、具体训练数据）。
- 语音模态引入了新型风险（speaker impersonation、未经同意的语音克隆等），缓解措施仍在探索中。
- 全模态模型的对齐和控制比纯文本模型更复杂。

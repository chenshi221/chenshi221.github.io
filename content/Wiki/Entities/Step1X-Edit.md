---
type: entity
status: active
created: "2026-04-30"
updated: "2026-04-30"
aliases: [Step1X-Edit]
tags: [image-editing, MLLM, diffusion, open-source, benchmark]
sources: ["[[Wiki/Sources/Step1X-Edit]]"]
---

# Step1X-Edit

## 基本信息

| 属性 | 值 |
|------|-----|
| 全称 | Step1X-Edit: A Practical Framework for General Image Editing |
| 作者 | Step1X-Image Team |
| 机构 | StepFun（阶跃星辰） |
| 年份 | 2025 (arXiv 2504.17761) |
| 类型 | MLLM + Diffusion 混合编辑模型 |

## 定位

**对标 GPT-4o / Gemini2 Flash 的开源图像编辑方案**。目标是缩小开源模型与闭源模型在图像编辑能力上的差距。

## 架构

```
参考图像 + 编辑指令
      ↓
  MLLM 编码（理解意图 + 图像内容）
      ↓
  Latent Embedding（编辑意图 + 视觉信息）
      ↓
  扩散图像解码器 → 目标图像
```

端到端联合训练，MLLM 提供深层指令理解，扩散解码器保证图像质量。

## 数据与评估

- **训练数据**：自建数据生成管线，覆盖 11 种编辑任务
- **GEdit-Bench**：基于真实用户指令的新型 benchmark（非模板生成），更贴近实际使用场景

## 在编辑演进中的位置

代表了从"纯扩散编辑"到 **MLLM-guided 扩散** 的架构演进：

```
InstructPix2Pix (纯扩散)
  → Step1X-Edit (MLLM + 扩散)
    → GPT-4o / Gemini Flash (MLLM 原生编辑)
```

与 InstructPix2Pix 的关键区别：MLLM 在编码指令时已融入场景理解，可以处理"让这张图更有氛围感"这类模糊指令。

## 在 Wiki 中的关联

- 比较：[[Wiki/Comparisons/编辑方法能力演进]]
- 主题：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 相关：[[Wiki/Entities/InstructPix2Pix]]

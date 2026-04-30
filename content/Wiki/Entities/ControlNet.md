---
type: entity
status: active
created: "2026-04-30"
updated: "2026-04-30"
aliases: [ControlNet]
tags: [image-generation, control, diffusion, foundational, DiT]
sources: ["[[Wiki/Sources/ControlNet]]"]
---

# ControlNet

## 基本信息

| 属性 | 值 |
|------|-----|
| 全称 | Adding Conditional Control to Text-to-Image Diffusion Models |
| 作者 | Lvmin Zhang, Anyi Rao, Maneesh Agrawala |
| 机构 | Stanford University |
| 年份 | 2023 (arXiv 2302.05543) |
| 类型 | 扩散模型控制架构 |

## 历史地位

**可控生成的里程碑工作**。被引超千次，"锁定基座 + 轻量适配"范式影响深远。催生了 T2I-Adapter、IP-Adapter、UniControl 等大量后续工作。

## 核心设计

### Zero Convolution
- 初始化为零的 1×1 卷积 → 训练初期不引入噪声
- 参数从零逐步增长，保护预训练权重完整性

### 锁定 + 复制架构
- 锁定原始 SD 编码器 → 保留大规模预训练知识
- 复制可训练副本 → 学习条件控制信号
- 零卷积连接副本到原始解码器

### 支持的条件类型
Canny 边缘、HED 边界、深度图、法线图、人体姿态、语义分割、涂鸦等。每种条件需单独训练一个 ControlNet。

## 影响

- EmoEdit 的 Emotion adapter 设计哲学直接继承（锁定基座，训 adapter）
- OminiControl 将其思路迁移到 DiT 架构（极简化，仅 0.1% 额外参数）
- 社区贡献了大量预训练 ControlNet 条件模型，形成生态

## 演进

| 架构 | ControlNet (2023) | OminiControl (2024) |
|------|-------------------|---------------------|
| 基座 | UNet (SD) | DiT |
| 额外参数 | 较多（复制编码器） | 0.1% |
| 设计 | 锁定+复制+零卷积 | 融入已有 DiT 架构 |

## 在 Wiki 中的关联

- 比较：[[Wiki/Comparisons/编辑方法能力演进]]
- 主题：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 相关：[[Wiki/Entities/OminiControl]]

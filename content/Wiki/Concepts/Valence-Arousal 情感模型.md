---
type: concept
status: active
created: "2026-04-30"
updated: "2026-04-30"
aliases: [V-A Model, Valence-Arousal, 效价-唤醒模型, Russell's circumplex model]
tags: [emotion, psychology, continuous-emotion, valence, arousal]
sources: ["[[Wiki/Sources/EmotiCrafter]]", "[[Wiki/Sources/EmoArt]]", "[[Wiki/Sources/Affective Image Editing]]"]
confidence: high
---

# Valence-Arousal 情感模型

## 定义

Valence-Arousal (V-A) 模型是心理学中广泛使用的连续情感表示模型（Russell, 1980）。它在二维笛卡尔空间中表示情感：

- **Valence（效价）**：愉悦度，从负向 (unpleasant) 到正向 (pleasant)
- **Arousal（唤醒度）**：激活度，从平静 (calm) 到兴奋 (excited)

## 与离散情感的关系

V-A 空间中的不同区域对应不同的离散情感：

| 区域 | Valence | Arousal | 典型情感 |
|------|---------|---------|----------|
| 第一象限 | + | + | Excited, Happy, Amused |
| 第二象限 | - | + | Angry, Alarmed, Afraid |
| 第三象限 | - | - | Sad, Depressed, Bored |
| 第四象限 | + | - | Calm, Content, Relaxed |

## 在图像生成中的应用

### 离散 vs 连续

| 属性 | 离散情感 (EmoEdit) | 连续 V-A (EmotiCrafter) |
|------|---------------------|--------------------------|
| 粒度 | 8 类固定 | 任意精度 (如 0.2 步长) |
| 过渡 | 不可渐变 | 平滑连续过渡 |
| 可解释性 | 高（直觉理解） | 中（需熟悉 V-A 空间） |
| Arousal 控制 | N/A | 挑战更大（人类标注一致性低） |

### 关键方法

- **EmotiCrafter**：用 V-A 值直接控制 SDXL 生成，emotion injection transformer 注入 V-A 特征
- **EmoArt**：为 132K 艺术作品标注二值 A-V 标签 + 12 类离散情感
- **AIEdiT**：构建连续情感谱，与 V-A 空间思路相近

## EmoArt 数据中的 V-A 分布

- 87.93% 正向 Valence、76.41% 低 Arousal → 艺术图像偏向"平静愉悦"
- 中国画极端偏向低 Arousal + 正向 Valence (99%+)
- 表现主义/超现实主义则有更多高 Arousal 和负向情感

## 开放问题

- Arousal 的视觉表达更难预测和控制（inter-annotator agreement 低）
- V-A 空间的哪些区域最适合情感增强的图像编辑？
- 是否可以学习从离散情感标签自动映射到 V-A 坐标？

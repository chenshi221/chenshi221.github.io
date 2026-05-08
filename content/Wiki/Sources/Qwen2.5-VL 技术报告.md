---
type: source
status: complete
created: '2026-05-07'
updated: '2026-05-07'
tags: [vlm, qwen, alibaba, multimodal, vision-language]
sources:
  - "[[Clippings/Qwen2.5-VL Technical Report]]"
---

# Qwen2.5-VL 技术报告

## 基本信息

- **标题**: Qwen2.5-VL Technical Report
- **作者**: Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song 等 (Core Contributors); An Yang, Binyuan Hui, Bowen Yu 等 (Contributors)
- **机构**: Qwen Team, Alibaba Group
- **年份**: 2025
- **arXiv**: 2502.13923
- **模型规模**: 3B / 7B / 72B 三种尺寸

## 核心论点

1. **细粒度视觉感知是 LVLM 的基础层**：Qwen2.5-VL 专注于探索细粒度感知能力，包括精确的目标定位（bounding box 和 point）、文档解析和长视频理解，为 LVLM 奠定坚实基础。

2. **原生动态分辨率处理**：不同于传统方法对坐标进行归一化，模型直接使用输入图像的实际尺寸来表示空间特征，使模型能够内在地学习尺度信息。

3. **绝对时间编码**：通过将 MRoPE 的时间分量与绝对时间对齐，模型能够理解不同 FPS 采样率视频中的时间动态，实现秒级事件定位。

4. **预训练数据大幅扩展**：预训练语料从 Qwen2-VL 的 1.2 万亿 token 扩展到 4.1 万亿 token，涵盖图像描述、交错图文数据、OCR、视觉知识、定位数据、文档解析数据、视频和 Agent 数据。

5. **Agent 能力增强**：通过统一的函数调用格式和多步轨迹数据，模型在计算机和移动设备操作方面展现出强大的 Agent 能力。

## 关键技术方法

### 视觉编码器 (Vision Encoder)

- **架构设计**：重新设计的 ViT，采用 2D-RoPE 和窗口注意力机制
- **窗口注意力**：大多数层使用窗口注意力（最大窗口 112x112，对应 8x8 patches），仅 4 层使用全自注意力，计算复杂度从二次降为线性
- **Patch 划分**：stride 为 14 的 patch 划分；视频使用 3D patch 分区，连续两帧分组
- **架构对齐 LLM**：采用 RMSNorm 和 SwiGLU 激活函数
- **训练策略**：从头训练，经历 CLIP 预训练、视觉-语言对齐和端到端微调阶段

### 动态分辨率与帧率

- **空间维度**：动态将不同尺寸图像转换为对应长度的 token 序列
- **时间维度**：动态 FPS 训练和绝对时间编码，适应可变帧率
- **输入处理**：图像高度和宽度调整为 28 的倍数后输入 ViT

### 多模态旋转位置编码 (MRoPE)

- **三维度分解**：将位置编码分解为时间、高度和宽度三个分量
- **文本输入**：三个分量使用相同位置 ID，等效于传统 1D RoPE
- **图像输入**：时间 ID 不变，高度和宽度根据空间位置分配
- **视频输入**：时间 ID 随帧递增，且与绝对时间对齐

### 视觉-语言融合 (MLP-based Merger)

- 将相邻 4 个 patch 特征分组，通过两层 MLP 投影到与文本嵌入对齐的维度
- 有效压缩图像特征序列长度，降低计算成本

### 预训练数据构建

- **交错图文数据**：四阶段评分系统（文本质量、图文相关性、图文互补性、信息密度平衡）
- **定位数据**：使用绝对坐标，支持超过 10,000 个物体类别
- **文档解析数据**：统一 HTML 格式表示（含表格、图表、公式、乐谱、化学式等）
- **OCR 数据**：多语言支持（法语、德语、意大利语、西班牙语、阿拉伯语、日语、韩语等）
- **视频数据**：动态 FPS 采样，超过半小时的长视频专门构建
- **Agent 数据**：统一移动、Web 和桌面平台的操作为函数调用格式

### 后训练 (Post-training)

- **SFT 阶段**：约 200 万条数据，纯文本和多模态各占 50%
- **数据过滤**：两阶段流水线（领域分类 + 领域定制过滤）
- **拒绝采样**：用于增强推理能力，特别是数学和代码任务
- **DPO 阶段**：基于偏好数据的对齐优化

## 主要结果

### 综合能力

- **MMMU val**: 70.2 (72B)，与 GPT-4o (69.1) 和 Claude 3.5 Sonnet (68.3) 相当
- **MathVista mini**: 74.8 (72B)，超过前代开源 SOTA (72.3)
- **MMBench-EN**: 88.6 (72B)，超过 InternVL2.5 (88.3)

### 文档理解与 OCR

- **DocVQA test**: 96.4 (72B)，超越所有对比模型
- **InfoVQA test**: 87.3 (72B)，大幅领先 InternVL2.5 (84.1)
- **OCRBench**: 885 (72B)，超过 InternVL2.5 (854)
- **OCRBench v2**: 英文 61.5 / 中文 63.7 (72B)，大幅超过 Gemini 1.5-Pro

### 空间理解

- **RefCOCO val**: 92.7 (72B)，接近 Grounding DINO (90.6)
- **ODinW**: 43.1 mAP (72B)，超越大多数 LVLM
- **CountBench**: 93.6 (72B)，超过所有对比模型

### 视频理解

- **LVBench**: 47.3 (72B)，大幅超过 GPT-4o (30.8)
- **MLVU M-Avg**: 74.6 (72B)，超过 GPT-4o (64.6)
- **Charades-STA mIoU**: 50.9 (72B)，超过 GPT-4o (35.7)
- **EgoSchema test**: 76.2 (72B)，超过 GPT-4o (72.2)

### Agent 能力

- **ScreenSpot Pro**: 43.6 (72B)，大幅超过 Aguvis-72B (23.6)
- **Android Control Low EM**: 93.7 (72B)，超过所有对比模型
- **AndroidWorld SR**: 35% (72B)，无需 SoM 标记即可完成任务
- **MobileMiniWob++ SR**: 68% (72B)，超过 GPT-4o (61%)

### 纯文本任务

- **MMLU-Pro**: 71.2 (72B)，与 Qwen2.5-72B (71.1) 持平
- **MATH**: 83.0 (72B)，与 Qwen2.5-72B (83.1) 持平
- **HumanEval**: 87.8 (72B)，超过 Llama-3.1-70B (80.5)
- **LiveBench**: 57.0 (72B)，超过所有对比的纯 LLM

## 局限性

1. **视觉幻觉问题**：在 HallBench 上得分 55.2 (72B)，低于 Qwen2-VL (58.1)，表明视觉幻觉问题可能未完全解决。

2. **长视频理解仍有提升空间**：在 LongVideoBench val 上得分 60.7 (72B)，低于 GPT-4o (66.7) 和 Gemini 1.5-Pro (64.0)。

3. **某些专业领域**：在 MMVU val 上得分 62.9 (72B)，低于 GPT-4o (67.4) 和 Gemini 1.5-Pro (65.4)，表明在某些专业视频理解任务上仍有差距。

4. **OSWorld 任务**：得分 8.83 (72B)，虽然超过 Qwen2-VL (2.42) 和 Gemini 2.0 (4.70)，但与 Claude (14.90) 仍有较大差距。

5. **模态对齐挑战**：论文承认在 CoT 推理中实现最优模态对齐仍是持续挑战。

## 与相关工作的关系

### 与 Qwen2-VL 的关系

- 直接继承者，在 Qwen2-VL 基础上进行全面升级
- 预训练数据从 1.2T 扩展到 4.1T tokens
- MRoPE 从相对时间 ID 升级为绝对时间对齐
- ViT 从头训练，引入窗口注意力机制

### 与 GPT-4o 和 Claude 3.5 Sonnet 的关系

- 72B 模型在多数基准上匹配或超越这些闭源模型
- 在文档理解和图表理解方面表现尤为突出
- Agent 能力与这些模型相当或更优

### 与 InternVL2.5 的关系

- 在多数基准上超越 InternVL2.5-78B
- 在 OCR 和文档解析方面优势明显
- 在视频理解和 Agent 任务上也有显著优势

### 与 Grounding DINO 和 Molmo 的关系

- 在目标定位任务上接近专业检测模型 Grounding DINO
- 点定位能力与 Molmo 72B 相当
- 计数能力超过所有对比模型

### 架构创新位置

- 窗口注意力 ViT：借鉴了 Swin Transformer 的思想，但针对多模态场景重新设计
- 动态分辨率：继承并扩展了 Qwen2-VL 的理念
- 绝对时间编码：对 MRoPE 的重要改进，区别于其他使用文本时间戳或额外头的方法

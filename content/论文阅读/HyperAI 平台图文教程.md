# HyperAI 平台图文教程

> HyperAI 是一家总部位于荷兰阿姆斯特丹的欧洲本土 GPU 云平台，主打 **低延迟、数据主权、欧洲合规**，面向 AI 训练和推理场景。

## 1. 平台概览

| 项目 | 信息 |
|------|------|
| 官网 | https://www.hyperai.ai |
| 总部 | Johan Huizingalaan 763A, 1066VH Amsterdam, Netherlands |
| 联系邮箱 | info@hyperai.ai |
| 创始人 | Andrew Foe (CEO) & Pieter Taks (CTO) |
| 合作伙伴 | Nvidia Inception Partner |

**核心卖点：**
- 🟢 **HyperLocal** — 数据存储在欧洲，符合 GDPR 等本地合规要求
- 🟢 **HyperPerformance** — 单向延迟 < 1ms，接近本地部署体验
- 🟢 **HyperSecure** — 数据不出欧洲，安全可控

---

## 2. 产品矩阵

HyperAI 提供 4 个核心产品：

### 2.1 HyperCLOUD（核心产品）

下一代 AI 云计算平台，基于 Nvidia A100/H100 GPU，支持按小时或按月/年计费。

**特点：**
- 最新 Nvidia GPU（A100 80G、H100）
- 欧洲本地数据中心，保证数据主权
- 灵活定价：Spot 按小时、Dedicated 按月、Enterprise 定制
- 白手套级别的客户支持

### 2.2 HyperSDK

预装 Nvidia AI SDK，开箱即用。

**包含组件：**
- TensorFlow、PyTorch、CUDA
- DeepStream（视频分析）
- Triton Inference Server（模型服务）
- RAPIDS（端到端数据科学流水线）

### 2.3 HyperSUPPORT

用户友好的客户门户，用于管理和监控项目。

**功能：**
- 部署管理
- 性能追踪
- 资源利用率分析
- 详细的分析和报告

### 2.4 HyperPOD（Beta）

创新的浸没式冷却方案，专为高密度 AI 硬件设计。

**特点：**
- PUE（能源使用效率）仅 1.05
- 25KW 功率容量
- 目标碳负排放
- 支持部署在 HyperCLOUD 或本地

---

## 3. 价格方案

HyperCLOUD 提供三种定价方案：

### 方案一：Spot（按需/竞价）
- 💰 **€1.25/小时**（约合 ¥10/小时）
- 1x Nvidia A100 80G GPU
- 24 核 CPU
- 240G RAM
- 100GB NVMe 存储
- Hyper Local 数据存储
- ✅ 适合：短期实验、调试、小规模训练

### 方案二：Dedicated（独占）
- 💰 **€1500/月**（约合 ¥12,000/月）
- 与 Spot 相同配置
- 独占资源，不受抢占影响
- ✅ 适合：持续训练、生产环境

### 方案三：Enterprise（企业定制）
- 💰 需联系定价
- Nvidia DGX A100/H100
- 全闪存 NVMe 数据湖
- 完整集群 POD
- 私有本地数据中心
- ✅ 适合：大规模训练、企业级部署

> 💡 **性价比分析**：Spot 方案 €1.25/hr 的 A100 80G 价格在欧洲市场具有一定竞争力，尤其适合对数据主权有要求的欧洲用户。但相比国内平台（如 AutoDL 等）价格偏高。

---

## 4. 注册与下单流程

### Step 1：访问官网

打开 https://www.hyperai.ai ，点击右上角或页面中的 **「ORDER NOW」** 按钮。

MEDIA:/Users/chenshi/.hermes/cache/screenshots/browser_screenshot_3c104b747dfa41bba2385983c2c4fce0.png

### Step 2：选择服务类型

进入订单表单页面，选择你需要的服务类型：

- **SPOT** — 按需竞价实例
- **DEDICATED** — 独占资源
- **ENTERPRISE** — 企业定制

MEDIA:/Users/chenshi/.hermes/cache/screenshots/browser_screenshot_c9605e982312490a83b6a60b67df8403.png

### Step 3：填写订单信息

填写以下信息：
- **NAME** — 你的姓名
- **EMAIL** — 邮箱地址
- **MOBILE NUMBER** — 手机号
- **COMPANY NAME** — 公司名称（个人可填个人或学校）
- **ADDITIONAL NOTES** — 额外需求说明

### Step 4：提交并等待审核

点击 **「SEND REQUEST」** 提交订单。HyperAI 团队会审核并联系你完成开通。

> ⚠️ **注意**：HyperAI 采用的是人工审核制，不像 AWS/GCP 那样即时开通。提交后需等待工作人员联系你进行 onboarding。

---

## 5. 使用体验

### 5.1 预装环境

开通后，实例预装了完整的 Nvidia AI SDK，包括：
- PyTorch / TensorFlow（主流深度学习框架）
- CUDA / cuDNN（GPU 加速库）
- DeepStream（视频分析）
- Triton Inference Server（模型推理服务）
- RAPIDS（GPU 加速数据科学）

### 5.2 客户门户

通过 HyperSUPPORT 客户门户可以：
- 监控 GPU 使用率和性能
- 管理实例的启动/停止
- 查看费用和账单
- 提交技术支持请求

### 5.3 网络延迟

HyperAI 主打低延迟，单向延迟 < 1ms，适合：
- 实时推理服务
- 对延迟敏感的 AI 应用
- 需要频繁数据传输的场景

---

## 6. 适用场景

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| 学术研究/实验 | Spot | 便宜灵活，按需使用 |
| 模型微调 | Dedicated | 需要稳定长时间运行 |
| 生产推理服务 | Dedicated/Enterprise | 低延迟，数据合规 |
| 大规模预训练 | Enterprise | 需要集群和高带宽 |
| GDPR 合规场景 | 任意方案 | 数据不出欧洲 |

---

## 7. 优缺点总结

### ✅ 优点
- 欧洲本土，数据主权合规（GDPR）
- 低延迟（< 1ms）
- 预装完整 AI 开发环境
- 白手套客户支持
- 价格相对欧洲市场有竞争力

### ❌ 缺点
- 非即时开通，需人工审核
- 价格相比中国/美国平台偏高
- 目前 GPU 型号选择有限（主要是 A100/H100）
- HyperPOD 仍在 Beta 阶段
- 社区和文档相对较少

---

## 8. 联系方式

- 🌐 官网：https://www.hyperai.ai
- 📧 邮箱：info@hyperai.ai
- 📍 地址：Johan Huizingalaan 763A, 1066VH Amsterdam, Netherlands
- 💼 LinkedIn：可在官网底部找到链接

---

## 9. 总结

HyperAI 是一个定位明确的欧洲 GPU 云平台，核心优势在于 **数据主权合规** 和 **低延迟**。如果你在欧洲工作或研究，且对数据存储位置有合规要求，HyperAI 是一个值得考虑的选择。对于价格敏感的用户，建议对比 Lambda Labs、CoreWeave、Vast.ai 等平台后再做决定。

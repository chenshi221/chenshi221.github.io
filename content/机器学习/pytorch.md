# PyTorch 学习笔记

## Tensor

Tensor（张量）是 PyTorch 的核心数据结构，本质是一个多维数组，与 NumPy 的 `ndarray` 类似，但可以在 GPU 上加速计算，并支持自动求导。

### Tensor 创建

```python
import torch

# 从 Python 列表创建
t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# 常用创建函数
torch.zeros(3, 4)           # 全零矩阵
torch.ones(3, 4)            # 全一矩阵
torch.eye(3)                # 单位矩阵
torch.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)    # [0.0, 0.25, 0.5, 0.75, 1.0]

# 随机创建
torch.rand(3, 4)            # [0, 1) 均匀分布
torch.randn(3, 4)           # 标准正态分布 N(0,1)
torch.randint(0, 10, (3,))  # [0, 10) 整数随机

# 特殊值
torch.full((3, 4), 3.14)    # 全部填充指定值
torch.empty(3, 4)           # 未初始化（速度快但值不确定）
```

### Tensor 属性

```python
t = torch.randn(3, 4)

t.shape       # torch.Size([3, 4])  形状
t.size()      # 同上
t.dim()       # 2  维度数
t.dtype       # torch.float32  数据类型
t.device      # device(type='cpu')  所在设备
t.requires_grad  # False  是否需要梯度
t.numel()     # 12  元素总数
```

### 常用数据类型

| 类型 | dtype | 说明 |
| --- | --- | --- |
| 32位浮点 | `torch.float32` / `torch.float` | 默认类型，模型训练常用 |
| 16位浮点 | `torch.float16` / `torch.half` | 混合精度训练 |
| 64位浮点 | `torch.float64` / `torch.double` | 高精度计算 |
| 8位整数 | `torch.int8` | 量化模型 |
| 32位整数 | `torch.int32` / `torch.int` | 索引 |
| 64位整数 | `torch.int64` / `torch.long` | 标签/索引常用 |
| 布尔 | `torch.bool` | 掩码 |

类型转换：

```python
t.float()         # 转为 float32
t.long()          # 转为 int64
t.to(torch.half)  # 转为 float16
t.bool()          # 转为 bool
```

### Tensor 索引与切片

```python
t = torch.randn(4, 5)

# 基本索引（与 NumPy 一致）
t[0]           # 第 0 行
t[:, 0]        # 第 0 列
t[1:3, 2:4]    # 子矩阵
t[0, -1]       # 第 0 行最后一个元素

# 花式索引
t[[0, 2]]          # 取第 0、2 行
t[:, [1, 3]]       # 取第 1、3 列
t[torch.tensor([0, 2]), torch.tensor([1, 3])]  # 取 (0,1) 和 (2,3)

# 布尔索引
mask = t > 0
t[mask]            # 取所有大于 0 的元素
t[t > 0] = 1       # 将所有大于 0 的元素设为 1
```

> **注意**：索引返回的是原 Tensor 的**视图（view）**，修改会影响原 Tensor；切片同理。使用 `.clone()` 可以获得独立副本。

### Tensor 形状操作

```python
t = torch.randn(2, 3, 4)

# reshape / view：改变形状（要求内存连续）
t.reshape(6, 4)          # 推荐，不要求连续
t.view(6, 4)             # 要求内存连续，否则报错

# 转置与维度重排
t.T                      # 仅适用于 2D
t.permute(2, 0, 1)       # 任意维度重排
t.transpose(0, 1)        # 交换两个维度

# 增减维度
t.unsqueeze(0)           # 在第 0 维增加一个维度 → (1, 2, 3, 4)
t.squeeze()              # 去掉所有大小为 1 的维度

# 拼接与分割
torch.cat([t, t], dim=0)        # 沿第 0 维拼接
torch.stack([t, t], dim=0)      # 沿新维度堆叠
torch.split(t, 2, dim=0)        # 按大小分割
torch.chunk(t, 2, dim=0)        # 按数量分割

# 展平
t.flatten()              # 全部展平为 1D
t.flatten(start_dim=1)   # 从第 1 维开始展平 → (2, 12)
```

### Tensor 数学运算

```python
a = torch.randn(3, 4)
b = torch.randn(3, 4)

# 逐元素运算
a + b           # 加法
a - b           # 减法
a * b           # 逐元素乘法
a / b           # 逐元素除法
a ** 2          # 幂运算
a.sqrt()        # 开方
a.exp()         # 指数
a.log()         # 对数
a.abs()         # 绝对值
a.clamp(0, 1)   # 裁剪到 [0, 1]

# 归约运算
a.sum()         # 所有元素求和
a.mean()        # 均值
a.std()         # 标准差
a.min()         # 最小值
a.max()         # 最大值
a.argmax()      # 最大值索引（展平后）
a.argmax(dim=1) # 每行最大值的列索引

# 矩阵运算
a @ b.T         # 矩阵乘法 (3x4) @ (4x3) = (3x3)
torch.mm(a, b.T)    # 同上，仅限 2D
torch.matmul(a, b.T) # 同上，支持广播
torch.dot(a[0], b[0])  # 向量点积
torch.norm(a)     # 范数
```

### Tensor 与 NumPy 互转

```python
import numpy as np

# Tensor → NumPy（共享内存！）
t = torch.ones(3)
arr = t.numpy()        # 共享内存，修改一方会影响另一方
arr = t.detach().cpu().numpy()  # 如果 t 有梯度或在 GPU 上

# NumPy → Tensor（共享内存！）
arr = np.ones(3)
t = torch.from_numpy(arr)  # 共享内存
t = torch.tensor(arr)      # 复制数据，不共享
```

---

## Autograd（自动求导）

PyTorch 的自动求导机制是其核心特性之一，通过**计算图（Computational Graph）**自动计算梯度。

### 基本用法

```python
# requires_grad=True 告诉 PyTorch 需要追踪对该 Tensor 的所有操作
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1   # y = x² + 2x + 1

# 反向传播，计算 dy/dx
y.backward()

print(x.grad)  # tensor(8.)  即 dy/dx = 2x + 2 = 2*3 + 2 = 8
```

### 计算图

PyTorch 使用**动态计算图（Dynamic Computational Graph）**，每次前向传播都会构建新的图，这意味着：

- 可以在代码中使用任意 Python 控制流（if/for/while）
- 调试方便，可以随时 print 中间结果
- 每次前向传播后图会被释放（除非指定 `retain_graph=True`）

```python
x = torch.randn(3, requires_grad=True)

# 前向传播
y = x * 2
z = y.mean()

# 反向传播
z.backward()
print(x.grad)  # tensor([0.6667, 0.6667, 0.6667])  即 d(mean(2x))/dx = 2/3
```

### 梯度控制

```python
# 停止梯度追踪
with torch.no_grad():
    # 此上下文内的操作不会被追踪，节省内存
    y = x * 2

# 或者使用 detach() 从计算图中分离
y = x.detach()

# 累加梯度（PyTorch 默认累加）
x.grad.zero_()  # 手动清零梯度
```

> **重要**：在训练循环中，每次迭代前必须调用 `optimizer.zero_grad()`，否则梯度会累加。

### 链式法则

对于复合函数，PyTorch 自动应用链式法则：

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial w}$$

```python
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

y = w * x + b       # 线性函数
L = y ** 2           # 损失

L.backward()         # 反向传播

print(x.grad)  # dL/dx = 2y * w = 2*(3*2+1)*3 = 42
print(w.grad)  # dL/dw = 2y * x = 2*(3*2+1)*2 = 28
print(b.grad)  # dL/db = 2y * 1 = 2*(3*2+1)   = 14
```

---

## nn.Module（神经网络模块）

`torch.nn` 是 PyTorch 构建神经网络的核心模块。所有模型都应继承 `nn.Module`。

### 基本结构

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 在这里定义网络层
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 定义前向传播
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MyModel()
print(model)  # 打印模型结构
```

### 常用网络层

#### 线性层

```python
nn.Linear(in_features, out_features, bias=True)
# 参数量: in_features * out_features + out_features (bias)
```

#### 卷积层

```python
# 1D 卷积
nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0)

# 2D 卷积（图像处理最常用）
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
# 输入: (batch, in_channels, H, W)
# 输出: (batch, out_channels, H', W')
# H' = (H + 2*padding - kernel_size) / stride + 1
```

#### 池化层

```python
nn.MaxPool2d(kernel_size, stride=None, padding=0)  # 最大池化
nn.AvgPool2d(kernel_size, stride=None, padding=0)  # 平均池化
nn.AdaptiveAvgPool2d(output_size)                   # 自适应平均池化
```

#### 归一化层

```python
nn.BatchNorm1d(num_features)   # 批归一化 (1D)
nn.BatchNorm2d(num_features)   # 批归一化 (2D)
nn.LayerNorm(normalized_shape) # 层归一化
nn.GroupNorm(num_groups, num_channels)  # 组归一化
```

#### Dropout

```python
nn.Dropout(p=0.5)  # 训练时随机将元素置零，概率为 p
```

#### Embedding

```python
nn.Embedding(num_embeddings, embedding_dim)
# 输入: (batch, seq_len) 整数索引
# 输出: (batch, seq_len, embedding_dim)
```

#### RNN / Transformer

```python
# RNN 系列
nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)
nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)

# Transformer
nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
nn.TransformerEncoderLayer(d_model=512, nhead=8)
nn.MultiheadAttention(embed_dim, num_heads)
```

### 常用激活函数

```python
nn.ReLU()            # max(0, x)，最常用
nn.LeakyReLU(negative_slope=0.01)  # 带负斜率的 ReLU
nn.GELU()            # Transformer 中常用
nn.Sigmoid()         # 1/(1+e^(-x))，输出 (0,1)
nn.Tanh()            # 输出 (-1,1)
nn.Softmax(dim=-1)   # 多分类归一化
nn.SiLU()            # Swish 激活函数，x * sigmoid(x)
```

### Sequential 快速建模

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

### 模型参数操作

```python
model = MyModel()

# 查看参数
for name, param in model.named_parameters():
    print(name, param.shape)

# 参数初始化
nn.init.xavier_uniform_(model.fc1.weight)  # Xavier 均匀初始化
nn.init.kaiming_normal_(model.fc1.weight)   # Kaiming 正态初始化（ReLU 推荐）
nn.init.zeros_(model.fc1.bias)

# 参数数量统计
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

### 容器模块

```python
# nn.ModuleList：参数化的列表
self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])

# nn.ModuleDict：参数化的字典
self.layers = nn.ModuleDict({'conv': nn.Conv2d(1, 32, 3), 'fc': nn.Linear(32, 10)})

# nn.ParameterList / nn.ParameterDict：存放纯参数
```

---

## Loss Functions（损失函数）

### 回归

```python
nn.MSELoss()          # 均方误差 L = mean((y - ŷ)²)
nn.L1Loss()           # 平均绝对误差 L = mean(|y - ŷ|)
nn.SmoothL1Loss()     # Huber 损失，结合 MSE 和 L1，对异常值鲁棒
```

### 分类

```python
# 二分类
nn.BCELoss()              # 二元交叉熵，输入需经过 Sigmoid
nn.BCEWithLogitsLoss()    # 内置 Sigmoid，数值更稳定，推荐使用

# 多分类
nn.CrossEntropyLoss()     # 内置 Softmax，输入为 raw logits
# 等价于 LogSoftmax(dim=1) + NLLLoss()
```

> **重要**：`nn.CrossEntropyLoss()` 的输入是 **未经 Softmax 的 logits**，不是概率分布。标签是类别索引（int64），不是 one-hot。

### 其他

```python
nn.CosineEmbeddingLoss()     # 余弦相似度损失
nn.TripletMarginLoss()       # 三元组损失（度量学习）
nn.KLDivLoss()               # KL 散度（知识蒸馏常用）
nn.FocalLoss()               # 不内置，需要自定义（类别不平衡）
```

---

## Optimizers（优化器）

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

### 常用优化器对比

| 优化器 | 特点 | 适用场景 |
| --- | --- | --- |
| SGD | 简单，需要仔细调 lr 和 momentum | CV 任务经典选择 |
| Adam | 自适应学习率，收敛快 | 通用默认选择 |
| AdamW | Adam + 解耦权重衰减 | Transformer 训练标配 |
| AdaGrad | 学习率会单调递减 | 稀疏数据 |
| RMSProp | 修复 AdaGrad 学习率衰减过快的问题 | RNN 训练 |

### 学习率调度

```python
# StepLR：每隔 step_size 个 epoch，lr 乘以 gamma
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# CosineAnnealingLR：余弦退火
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# ReduceLROnPlateau：当指标停止改善时降低 lr
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1)

# OneCycleLR：一周期策略（1cycle policy）
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=1000)

# 在训练循环中
scheduler.step()              # StepLR / CosineAnnealingLR
scheduler.step(val_loss)      # ReduceLROnPlateau 需要传入监控指标
```

---

## Dataset 与 DataLoader

### 自定义 Dataset

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
```

### DataLoader

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,         # 训练集打乱
    num_workers=4,        # 多进程加载
    pin_memory=True,      # 加速 GPU 传输
    drop_last=True,       # 丢弃最后不完整的 batch
)

# 迭代
for batch_idx, (data, labels) in enumerate(train_loader):
    data = data.to(device)
    labels = labels.to(device)
    # 训练...
```

### 内置数据集

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
```

### 常用数据变换（torchvision.transforms）

```python
transforms.Compose([
    transforms.Resize(256),                    # 缩放
    transforms.CenterCrop(224),                # 中心裁剪
    transforms.RandomHorizontalFlip(p=0.5),    # 随机水平翻转
    transforms.RandomRotation(15),             # 随机旋转 ±15°
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
    transforms.ToTensor(),                     # 转为 Tensor，归一化到 [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 均值
                         std=[0.229, 0.224, 0.225]),   # ImageNet 标准差
])
```

---

## 训练循环模板

```python
import torch
import torch.nn as nn

# 1. 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 准备数据
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. 创建模型、损失函数、优化器
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4. 训练
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # 训练模式（启用 Dropout、BatchNorm）
    total_loss = 0

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)

        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # 5. 验证
    model.eval()  # 评估模式（禁用 Dropout，BatchNorm 使用全局统计）
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
```

### model.train() vs model.eval()

| 模式 | Dropout | BatchNorm | 用途 |
| --- | --- | --- | --- |
| `model.train()` | 随机丢弃 | 使用 batch 统计量 | 训练 |
| `model.eval()` | 不丢弃 | 使用全局统计量 | 推理/验证 |

---

## GPU 管理

```python
# 检查 GPU
torch.cuda.is_available()          # 是否可用
torch.cuda.device_count()          # GPU 数量
torch.cuda.current_device()        # 当前 GPU 编号
torch.cuda.get_device_name(0)      # GPU 名称

# Tensor 转移到 GPU
t = t.to('cuda')                   # 转到 GPU
t = t.cuda()                       # 同上
t = t.to('cuda:1')                 # 指定 GPU 编号
t = t.cpu()                        # 转回 CPU

# 模型转移到 GPU
model = model.to(device)

# 指定设备的通用写法
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 清理 GPU 缓存
torch.cuda.empty_cache()
```

### 多 GPU 训练

```python
# DataParallel（单机多卡，简单但效率一般）
model = nn.DataParallel(model)

# DistributedDataParallel（推荐，效率更高）
# 需要 torch.distributed.launch 或 torchrun 启动
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

---

## 模型保存与加载

```python
# 方法一：保存整个模型（简单但不灵活）
torch.save(model, 'model.pth')
model = torch.load('model.pth')

# 方法二：只保存参数（推荐）
torch.save(model.state_dict(), 'model.pth')
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 保存检查点（包含训练状态）
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# 加载检查点
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

---

## 常用技巧

### 混合精度训练（AMP）

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, labels in train_loader:
    optimizer.zero_grad()
    with autocast():                     # 自动选择 float16/float32
        outputs = model(data)
        loss = criterion(outputs, labels)
    scaler.scale(loss).backward()        # 缩放损失防止梯度下溢
    scaler.step(optimizer)
    scaler.update()
```

### 梯度裁剪

```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# 在 loss.backward() 之后、optimizer.step() 之前调用
```

### detach() 的常见用法

```python
# 从计算图中分离，不追踪梯度
# 用途 1：验证时不需要梯度
with torch.no_grad():  # 更推荐的方式
    output = model(input)

# 用途 2：截断梯度流（如 GAN 中固定判别器更新生成器）
loss = criterion(fake_output.detach(), real_labels)

# 用途 3：Tensor 转 NumPy
result = output.detach().cpu().numpy()
```

### 权重初始化

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

model.apply(init_weights)
```

### 参数分组

```python
# 不同参数组使用不同学习率
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-5},    # 预训练层低学习率
    {'params': model.head.parameters(), 'lr': 1e-3},        # 新层高学习率
], weight_decay=0.01)

# 冻结参数
for param in model.backbone.parameters():
    param.requires_grad = False
```

---

## 总结

1. **Tensor**：PyTorch 的基本数据结构，支持 GPU 加速和自动求导
2. **Autograd**：通过动态计算图自动计算梯度，支持任意 Python 控制流
3. **nn.Module**：构建神经网络的基类，提供各层、损失函数、激活函数等模块
4. **Optimizers**：SGD、Adam、AdamW 等优化器，配合学习率调度器使用
5. **Dataset & DataLoader**：高效的数据加载管道，支持多进程和批量处理
6. **训练循环**：forward → loss → backward → step 的标准流程
7. **GPU**：通过 `.to(device)` 统一管理 CPU/GPU 设备
8. **模型持久化**：推荐只保存 `state_dict()`，支持检查点恢复训练

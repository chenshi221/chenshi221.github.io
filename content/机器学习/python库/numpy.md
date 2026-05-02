# NumPy 官方文档总结

## numpy.array vs list

- **NumPy 数组**在创建时具有固定大小，而**Python 列表**可以动态增长。更改 `ndarray` 的大小会创建一个新数组并删除原始数组。
- **NumPy 数组**中的元素必须是相同的数据类型，这样内存中每个元素的大小相同。而**Python 列表**可以包含不同类型的元素。

## numpy 快速入门

### `ndarray` 对象的属性

- `ndim`：数组的维数。例如，一个二维数组的维数为 2。
- `shape`：数组的形状，表示每个维度的大小，返回一个元组。比如 `(3, 4)` 表示 3 行 4 列。
- `size`：数组的元素总数，等于所有维度大小的乘积。例如，`3*4 = 12`。
- `dtype`：数组的数据类型，返回数组元素的类型，如 `int32`、`float64` 等。
- `itemsize`：数组中每个元素的字节大小。例如，一个 `int32` 类型的元素占 4 字节。

### 数组创建

```python
numpy.array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)
```

- `object`：数据源，可以是数组或嵌套序列。
- `dtype`：数组元素的数据类型，默认为 `None`，`numpy` 会根据输入自动推断。
- `copy`：是否复制数据，默认为 `True`。若为 `False`，返回输入数组的引用。
- `order`：多维数组在内存中的存储顺序。`'C'` 表示按行存储（C-style），`'F'` 表示按列存储（Fortran-style），`'K'` 表示保持原顺序。
- `subok`：如果为 `True`，返回子类数组；若为 `False`，返回 `ndarray` 类型。
- `ndmin`：指定返回数组的最小维数。例如，`ndmin=2` 保证返回至少二维数组。

### 数组的基本操作

- **数组加法**：`array1 + array2`，对两个形状相同的数组元素逐一相加。
- **数组索引**：支持整数、切片、布尔索引等方式。
  - 整数索引：`array[0]` 返回第一个元素，`array[-1]` 返回最后一个元素。
  - 切片索引：`array[1:5]` 返回第 2 到第 5 个元素。
  - 布尔索引：`array[array > 5]` 返回数组中大于 5 的元素。
- **数组形状变换**：使用 `reshape()` 函数改变数组的形状，保持元素总数不变。
- **数组广播**：当两个数组的形状不同，`numpy` 会自动扩展其中一个数组，使它们形状相同。

### 操作注意

- 形如 `A = A + B` 的操作会创建一个新的空间，将 `A` 和 `B` 相加后将结果赋值给 `A`。
- 形如 `A += B` 的操作会直接在 `A` 上进行操作，不会创建新的空间。
- 操作不同类型的数组时，结果数组的类型会匹配更通用或更精确的数据类型。

### 其他常见功能

- `numpy.zeros(shape, dtype=float)`：创建一个全零数组。
- `numpy.ones(shape, dtype=float)`：创建一个全一数组。
- `numpy.arange([start,] stop[, step,])`：创建一个均匀间隔的数组，类似于 Python 的 `range` 函数。
- `numpy.linspace(start, stop, num=50)`：创建一个均匀分布的数值数组。
- `numpy.eye(N, M=None, k=0, dtype=float)`：创建一个对角线为 1 的单位矩阵。
- `numpy.random.random(size)`：返回一个指定大小的随机数组。

### 数学函数

- `numpy.sum(array)`：返回数组所有元素的和。
- `numpy.mean(array)`：返回数组元素的均值。
- `numpy.median(array)`：返回数组元素的中位数。
- `numpy.std(array)`：返回数组的标准差。
- `numpy.min(array)`：返回数组的最小值。
- `numpy.max(array)`：返回数组的最大值。

### 注意事项

- 简单赋值不会创建数组对象的副本，而是创建对相同数据的另一个引用。

### 广播规则

- 维度比较从最后一维开始，如果维度相同或者其中一个维度为 1，则继续比较前一维，直到一个数组的维度全部比较完毕。此时可以进行广播。
- **规则 1**：如果所有输入数组的维度数不同，会在较小数组的形状前添加“1”，直到所有数组具有相同的维度数。
- **规则 2**：如果某个维度大小为 1，那么该维度的元素将被“广播”至与最大维度相同的大小，值假定在该维度上是相同的。

---

## 进阶索引

### 花式索引（Fancy Indexing）

使用整数数组作为索引，可以一次性选取多个不连续的元素。

```python
import numpy as np

a = np.array([10, 20, 30, 40, 50])

# 整数数组索引
indices = [0, 2, 4]
print(a[indices])       # [10 30 50]

# 二维数组
b = np.arange(12).reshape(3, 4)
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]

print(b[[0, 2]])           # 取第 0、2 行
print(b[[0, 2], [1, 3]])   # 取 (0,1) 和 (2,3) 两个元素 → [1, 11]
```

### 布尔索引进阶

```python
a = np.array([1, 2, 3, 4, 5])

# 复合条件
mask = (a > 2) & (a < 5)       # AND
mask = (a < 2) | (a > 4)       # OR
mask = ~(a == 3)                # NOT

print(a[mask])  # [1 2 4 5]

# np.where：条件选择
result = np.where(a > 3, a, 0)   # 大于3保留，否则置0 → [0 0 0 4 5]

# np.select：多条件
conditions = [a < 2, a < 4, a >= 4]
choices = [0, 1, 2]
result = np.select(conditions, choices)  # → [0 1 1 2 2]
```

### ix_ 函数

用于构造开放网格的索引，常用于同时对多个维度进行花式索引。

```python
a = np.arange(12).reshape(3, 4)

# 选取第 0、2 行和第 1、3 列的交叉元素
rows = [0, 2]
cols = [1, 3]
print(a[np.ix_(rows, cols)])
# [[ 1,  3],
#  [ 9, 11]]
```

---

## 线性代数

NumPy 的 `linalg` 模块提供了丰富的线性代数操作。

### 基本矩阵运算

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵乘法
C = A @ B                # 推荐写法
C = np.dot(A, B)         # 等价
C = np.matmul(A, B)      # 等价

# 转置
A.T                      # 转置
np.transpose(A)          # 等价

# 逐元素乘法（Hadamard 积）
C = A * B
```

### linalg 模块

```python
# 行列式
np.linalg.det(A)

# 逆矩阵
np.linalg.inv(A)

# 伪逆（Moore-Penrose）
np.linalg.pinv(A)

# 特征值与特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

# 奇异值分解 (SVD)
U, S, Vt = np.linalg.svd(A)

# QR 分解
Q, R = np.linalg.qr(A)

# 解线性方程组 Ax = b
x = np.linalg.solve(A, b)

# 范数
np.linalg.norm(A)              # Frobenius 范数
np.linalg.norm(A, ord=2)       # 2-范数

# 矩阵的秩
np.linalg.matrix_rank(A)

# 迹（对角线元素之和）
np.trace(A)
```

### 常用矩阵构造

```python
np.eye(3)                      # 3x3 单位矩阵
np.diag([1, 2, 3])             # 对角矩阵
np.diag(A)                     # 提取对角线元素
np.triu(A)                     # 上三角矩阵
np.tril(A)                     # 下三角矩阵
```

---

## 随机数模块

### 基本随机数

```python
# 随机种子（可复现）
np.random.seed(42)

# 均匀分布
np.random.rand(3, 4)           # [0, 1) 均匀分布
np.random.uniform(0, 10, (3,)) # [0, 10)

# 正态分布
np.random.randn(3, 4)          # 标准正态 N(0,1)
np.random.normal(5, 2, (3,))   # N(5, 4)

# 整数随机
np.random.randint(0, 10, (3,)) # [0, 10)

# 随机选择
a = np.array([10, 20, 30, 40, 50])
np.random.choice(a, size=3, replace=False)  # 不放回抽样
np.random.choice(a, size=3, replace=True)   # 有放回抽样
```

### 打乱与排列

```python
a = np.arange(10)
np.random.shuffle(a)           # 原地打乱

# 返回打乱后的索引（不修改原数组）
perm = np.random.permutation(10)
b = a[perm]
```

### 新版随机 API（NumPy 1.17+ 推荐）

```python
rng = np.random.default_rng(42)

rng.random((3, 4))             # [0, 1) 均匀分布
rng.normal(0, 1, (3, 4))      # 正态分布
rng.integers(0, 10, (3,))     # 整数
rng.choice(a, size=3)         # 随机选择
rng.permutation(10)            # 随机排列
```

---

## 常用实用函数

### 聚合与统计

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

np.sum(a, axis=0)         # 按列求和 → [5, 7, 9]
np.sum(a, axis=1)         # 按行求和 → [6, 15]
np.cumsum(a, axis=0)      # 累积和
np.diff(a, axis=1)        # 差分
np.percentile(a, 50)      # 中位数
np.quantile(a, 0.75)      # 75% 分位数
```

### 数组操作

```python
a = np.array([[3, 1, 2], [6, 4, 5]])

# 排序
np.sort(a, axis=1)             # 每行排序
np.argsort(a, axis=1)          # 排序后的索引

# 去重
np.unique(a, return_counts=True)  # 返回唯一值和计数

# 搜索
np.where(a > 3)                # 满足条件的索引
np.argmax(a, axis=1)           # 每行最大值索引

# 集合运算
np.intersect1d([1,2,3], [2,3,4])  # 交集 → [2,3]
np.union1d([1,2,3], [2,3,4])      # 并集 → [1,2,3,4]
np.setdiff1d([1,2,3], [2,3,4])    # 差集 → [1]
```

### 数组拼接与拆分

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# 拼接
np.concatenate([a, b], axis=0)   # 垂直拼接
np.vstack([a, b])                # 等价
np.concatenate([a, b], axis=1)   # 水平拼接
np.hstack([a, b])                # 等价
np.stack([a, b], axis=0)         # 新维度堆叠

# 拆分
np.split(a, 2, axis=0)          # 均分为 2 份
np.array_split(a, 3, axis=0)    # 不均等拆分
```

---

## 性能优化技巧

```python
# ✓ 向量化替代循环
result = a ** 2                  # 远快于 for 循环

# ✓ 原地操作节省内存
a += 1                           # 不创建新数组
np.sqrt(a, out=a)                # 将结果写入已有数组

# ✓ 切片返回视图（共享内存，无复制）
b = a[0:3, :]

# ✓ 花式索引返回副本（独立内存）
b = a[[0, 2]]
```

---

## 常见陷阱

1. **视图 vs 副本**：切片返回视图，花式索引和布尔索引返回副本
2. **整数溢出**：`np.array([255], dtype=np.uint8) + 1` 溢出为 0
3. **浮点精度**：`np.float32` 约 7 位有效数字，`np.float64` 约 15 位
4. **axis 语义**：`axis=0` 沿行操作（跨行），`axis=1` 沿列操作（跨列）

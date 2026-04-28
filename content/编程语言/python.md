# Python

## ==列表==

1. `array.append(str)`
2. `insert(index,str)` 剩余的后移一位
3. `del array[i]`
4. `array.pop(i)` 删除并返回第 i 个列表值
5. `array.remove(str)` 查找并删除匹配值
6. `array.sort()` 按字典顺序排序，反向需传递 `reverse=True`
7. `sorted(array)` 返回 sort 后的列表，但不改变原列表
8. `array.reverse()` 倒序排列列表
9. `len(array)` 返回列表长度
10. 负数 index 表示倒数

---

## ==操作列表==

### 列表

1. `for x in array：` 遍历列表
2. `range(a,b,c)` 生成范围 `a<=x<b`，步长为 c，省略为 1
3. `list(range(a,b))` 生成顺序列表
4. `min(array),max(array),sum(array)` 统计
5. 列表解析：`squares = [value**2 for value in range(1,11)]`
6. 列表切片：`array[l:r]` 默认值为始末
7. 列表复制要使用切片的方式，否则是将两个变量指向同一个列表

### 元组

1. 不可变的列表，用圆括号来标识

---

## 字典

1. `zidian={'key' : value, ····}`
2. 用 key 作为索引
3. 添加键值对

   ```py
   zidian['new_key']=new_value
   ```

4. 删除键值对

   ```py
   del zidian['key']
   ```

5. 遍历字典

   ```py
   for k,v in zidian.keys():
   ```

6. 遍历字典键

   ```py
   for k in zidian.keys():
   ```

7. 获取字典的元素时，获取顺序是不可预测的
8. 创建集合 `set(zidian.values())` 每个元素皆不同
9. 列表和字典可以互相嵌套

---

## 输入

1. `mesage=input("information")` 读入为字符串

---

## 函数

1. `def func()`
2. `"""xianshi"""` 文档字符串（docstring）的注释
3. 所有提供了默认值的必须放在参数列表最后
4. 防止函数修改列表，传递切片
5. 传递任意数量的实参(列表)

   ```py
   def make_pizza(*toppings):
      """打印顾客点的所有配料"""
      print(toppings)

   make_pizza('pepperoni')
   make_pizza('mushrooms', 'green peppers', 'extra cheese')
   ```

6. 使用任意数量的关键字实参（字典）

   ```py
   def build_profile(first, last, **user_info):
      """创建一个字典，其中包含我们知道的有关用户的一切"""
      profile = {}
      profile['first_name'] = first
      profile['last_name'] = last
      for key, value in user_info.items():
   ```

7. 导入模块和函数

   ```py
   import pizza as p
   p.make_pizza()
   from pizza import make_pizza as mp
   mp()
   ```

8. 导入所有函数 `from modle import *`

---

## 类

### 创建和使用类

1. 构造函数 `__init__(self，val1，val2,···)`
2. 形参 `self` 类对象的引用，调用函数时不用传递它

### 继承

1. 创建子类时，父类必须包含在当前文件中，且位于子类前面。定义子类时，必须在括号内指定父类的名称。方法 __init__()接受创建子类实例所需的信息
2. super()是一个特殊函数，帮助 Python 将父类和子类关联起来，调用父类的方法 __init__()
3. 子类中可以重写父类函数，对子类对象会调用子类重写方法

### Python 标准库

1. 记录字典添加顺序

   ```py
   from collection import OrderedDict
   new_dict=OrderedFict()   
   ```

---

## 文件 I/O 与异常

### 文件读取

1. `with open('txt_flie\file_name.txt') as file` 打开文件
2. `for line in file` 按行读取
3. 使用关键字 with 时， open()返回的文件对象只在 with 代码块内可用
4. 方法 readlines()从文件中读取每一行

### 写入文件

1. 写入空文件

   ```py
   with open(filename, 'w') as file_object:
      file_object.write("I love programming.")
   ```

打开文件时，可指定读取模式（ 'r'）、 写入模式（ 'w'）、 附加模式（ 'a'）或让你能够读取和写入文件的模式（ 'r+'）。
如果你省略了模式实参， Python 将以默认的只读模式打开文件, 如果指定的文件已经存在， Python 将在返回文件对象前清空该文件。
Python 只能将字符串写入文本文件

### 异常

1. try-except 代码块

   ```py
   try:
      content
   except errortype:
   
   else: 
   ```

### 存储数据

1. 模块 json
2. `json.dump(array,flie)` 函数 json.dump()接受两个实参：要存储的数据以及可用于存储数据的文件对象。
3. `json.load(file)` 从 file 中读取信息

## CHAPTER 1 解释器

1. 启动解释器: `python`
2. 退出解释器: `exit() or Ctrl+D(Unix) or Ctrl+Z(Windows) or quit()`
3. 命令行参数: `python -c command [arg] ...`，读取的命令行参数存储在 sys.argv 列表中：
   1. 未给定命令行参数时，`sys.argv` 至少包含一个元素，sys.argv [0] 是脚本的名称'-'。
   2. 使用 `-c command` 选项时，`sys.argv[0]` 是'-c'。
   3. 使用 `-m module` 选项时，`sys.argv[0]` 是包含目录的模块名称。
4. 指定文件编码，第一行添加，默认为 utf-8

   ```py
   # -*- coding: encoding -*-
   ```

5. pip 安装模块

   ```py
   pip install module
   ```

## CHAPTER 2 python 速览

### Num

1. `/` 始终返回浮点数，`//` 返回整数(向下取整)
2. 混合类型运算时，整数转换为浮点数
3. `**` 乘方

### Str

1. str：使用单引号或双引号，三引号可换行，使用 `\` 可取消换行
2. `r'...'` 原始字符串，不转义
3. 使用 `+` 连接字符串，`*` 重复字符串
4. 连续字符串自动连接, 只适用于字面量

   ```py
   text = ('Put several strings within parentheses '
           'to have them joined together.')
   ```

5. ==字符串支持索引和切片==，索引从 0 开始，负数从-1 开始
6. 获取切片时，左闭右开: `s[1:3]`，省略时默认为 0 和最后一个，`s[:i]+s[i:]` 等于 s
7. 索引越界会报错，切片不会，切片超出范围时，超出部分被忽略
8. 字符串不可变，不能修改字符串的内容，只能创建新的字符串
9. 格式化字符串 f-string

   ```py
   name = 'Alice'
   f'Hello, {name}!'
   ```

10. 字符串方法，方括号表示此参数可选

* `s.find(sub[, start[, end]])` ，查找子串，返回最小索引，未找到返回-1
* `s.rfind(sub[, start[, end]])` ，从右边开始查找
* `s.count(sub[, start[, end]])` ，返回子串出现次数，sub 为空时返回 1+最大索引，即字符串长度
* `s.encode(encoding='utf-8', errors='strict')` , 编码字符串
* `s.endswith(suffix[, start[, end]])` ，判断是否以 suffix 结尾, 返回布尔值
* `s.replace(old, new[, count])` ，替换子串，count 表示替换次数

### List

1. 列表是可变的，可以修改列表的内容
2. 列表可以包含任意类型的对象，甚至可以包含其他列表
3. 列表的简单赋值是引用，修改引用的内容会改变原列表
4. 切片赋值是浅拷贝，浅拷贝创建一个新的对象，新对象中的元素是原对象中元素的引用。也就是说，新对象的第一层元素是复制的，但内部嵌套对象的引用仍然指向原始对象的内存地址。
5. 深拷贝，`copy.deepcopy()`，创建一个新的对象，递归地将原对象的内部对象复制到新对象中
6. 为切片赋值时，会改变原列表
7. 列表支持+和 *操作，+连接列表，* 重复列表
8. 使用 in 和 not in 检查列表中是否存在元素
9. 列表生成式：`[expr for var in iterable]`

   ```py
   squares = [x**2 for x in range(10)]
   ```

10. 列表方法

   * `list.append(x)` ，在列表末尾添加一个元素
   * `del list[i]` ，删除指定位置的元素
   * `max,min,cmp,len,list` ，返回最大值，最小值，比较，长度，列表
   * `list.count(x)` ，返回元素出现次数
   * `list.extend(iterable)` ，在列表末尾一次性追加另一个序列的多个值
   * `list.index(x[, start[, end]])` ，返回第一个匹配元素的索引
   * `list.insert(i, x)` ，在指定位置插入元素
   * `list.pop([i])` ，删除指定位置的元素并返回
   * `list.remove(x)` ，删除第一个匹配元素, 匹配方向从左到右
   * `list.reverse()` ，反转列表
   * `list.sort(cmp=None, key=None, reverse=False)` ，排序列表，cmp 为比较函数，key 为排序函数，reverse 为排序方向，默认升序
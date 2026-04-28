# Pandas 菜鸟教程

## Series

1. **创建 Series**

    ```python
    pandas.Series(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False)
    ```

    - `data`：数据源，可以是列表、字典、数组等。不提供时，创建一个空 Series。
    - `index`：索引，可以是列表、字典、数组等。不提供时，使用默认索引。
    - `dtype`：数据类型，可以是 `int`、`float`、`str` 等。不提供时，`pandas` 会自动推断。
    - `name`：Series 的名称。
    - `copy`：是否复制数据，默认为 `False`。若为 `True`，返回输入数据的副本。
    - `fastpath`：是否使用快速路径，默认为 `False`。若为 `True`，不会检查数据的类型。

2. **Series 基本操作**

    - **索引**：`series[index]`，返回索引对应的值。
    - **切片**：`series[start:end]`，返回指定范围的值。
    - **布尔索引**：`series[series > 5]`，返回满足条件的值。
    - **计算**：`series1 + series2`，对两个 Series 对象逐一相加。
    - **缺失值**：`None` 或 `NaN` 表示缺失值。

3. **Series 常用方法**

    - `index`：返回 Series 的索引。
    - `values`：返回 Series 的值，类型为 `numpy.ndarray`。
    - `head(n)`：返回前 `n` 行数据，默认为 5。
    - `tail(n)`：返回后 `n` 行数据，默认为 5。
    - `dtype`：返回 Series 的数据类型。
    - `shape`：返回 Series 的形状，即 `(n,)`。
    - `size`：返回 Series 的元素总数。
    - `count()`：返回非缺失值的数量。
    - `map(func)`：对 Series 的每个元素应用函数。
    - `apply(func)`：对 Series 的每个元素应用函数。

## DataFrame

1. **创建 DataFrame**

    ```python
    pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
    ```

    - `data`：数据源，可以是列表、字典、数组等。不提供时，创建一个空 DataFrame。
    - `index`：行索引，可以是列表、字典、数组等。不提供时，使用默认索引。
    - `columns`：列索引，可以是列表、字典、数组等。不提供时，使用默认索引。
    - `dtype`：数据类型，可以是 `int`、`float`、`str` 等。不提供时，`pandas` 会自动推断。
    - `copy`：是否复制数据，默认为 `False`。若为 `True`，返回输入数据的副本。

2. **DataFrame 基本操作**

    - **索引**：`df[column]`，返回列名对应的列。
    - **切片**：`df[start:end]`，返回指定范围的行。
    - **布尔索引**：`df[df['column'] > 5]`，返回满足条件的行。
    - **计算**：`df1 + df2`，对两个 DataFrame 对象逐一相加。
    - **缺失值**：`None` 或 `NaN` 表示缺失值。

3. **DataFrame 常用方法**

    - `index`：返回 DataFrame 的行索引。
    - `columns`：返回 DataFrame 的列索引。
    - `values`：返回 DataFrame 的值，类型为 `numpy.ndarray`。
    - `head(n)`：返回前 `n` 行数据，默认为 5。
    - `tail(n)`：返回后 `n` 行数据，默认为 5。
    - `dtypes`：返回 DataFrame 的数据类型。
    - `shape`：返回 DataFrame 的形状，即 `(n, m)`。
    - `size`：返回 DataFrame 的元素总数。
    - `count()`：返回非缺失值的数量。
    - `describe()`：返回 DataFrame 的统计信息。
    - `sort_values(by, ascending=True)`：按照指定列排序，默认升序。
    - `sort_index(axis=0, ascending=True)`：按照索引排序，默认升序。

## CSV 文件操作

1. **读取 CSV 文件**

    ```python
    pandas.read_csv(filepath_or_buffer, sep=',', header='infer', names=None, index_col=None)
    ```

    - `filepath_or_buffer`：文件路径或对象。
    - `sep`：分隔符，默认为逗号 `,`。
    - `header`：指定行数作为列名，默认为 `'infer'`。
    - `names`：列名列表，与 `header=None` 一起使用。
    - `index_col`：指定列作为行索引。

2. **写入 CSV 文件**

    ```python
    pandas.DataFrame.to_csv(path_or_buf, sep=',', na_rep='', columns=None, header=True, index=True)
    ```

    - `path_or_buf`：文件路径或对象。
    - `sep`：分隔符，默认为逗号 `,`。
    - `na_rep`：缺失值表示，默认为空字符串。
    - `columns`：要写入的列，默认为全部列。
    - `header`：是否写入列名，默认为 `True`。
    - `index`：是否写入行索引，默认为 `True`。
    - `mode`：写入模式，默认为 `'w'`。
    - `encoding`：编码格式，默认为 `'utf-8'`。

## Excel 文件操作

1. **读取 Excel 文件**

    ```python
    pandas.read_excel(io, sheet_name=0, *, header=0, names=None, index_col=None, usecols=None, dtype=None, engine=None, converters=None, true_values=None, false_values=None, skiprows=None, nrows=None, na_values=None, keep_default_na=True, na_filter=True, verbose=False, parse_dates=False, date_parser=<no_default>, date_format=None, thousands=None, decimal='.', comment=None, skipfooter=0, storage_options=None, dtype_backend=<no_default>, engine_kwargs=None)
    ```

    - **io**：这是必需的参数，指定了要读取的 Excel 文件的路径或文件对象。
    - **sheet_name=0**：指定要读取的工作表名称或索引，默认为 `0`，即第一个工作表。
    - **header=0**：指定用作列名的行，默认为 `0`，即第一行。
    - **names=None**：用于指定列名的列表。如果提供，将覆盖文件中的列名。
    - **index_col=None**：指定用作行索引的列，可以是列的名称或数字。
    - **usecols=None**：指定要读取的列，可以是列名的列表或列索引的列表。
    - **dtype=None**：指定列的数据类型，可以是字典格式，键为列名，值为数据类型。
    - **engine=None**：指定解析引擎，默认为 `None`，`pandas` 会自动选择。
    - **converters=None**：用于转换数据的函数字典。
    - **true_values=None**：指定应该被视为布尔值 `True` 的值。
    - **false_values=None**：指定应该被视为布尔值 `False` 的值。
    - **skiprows=None**：指定要跳过的行数或要跳过的行的列表。
    - **nrows=None**：指定要读取的行数。
    - **na_values=None**：指定应该被视为缺失值的值。
    - **keep_default_na=True**：指定是否要将默认的缺失值（例如 `NaN`）解析为 NA。
    - **na_filter=True**：指定是否要将数据转换为 NA。
    - **verbose=False**：指定是否要输出详细的进度信息。
    - **parse_dates=False**：指定是否要解析日期。
    - **date_parser=<no_default>**：用于解析日期的函数。
    - **date_format=None**：指定日期的格式。
    - **thousands=None**：指定千位分隔符。
    - **decimal='.'**：指定小数点字符。
    - **comment=None**：指定注释字符。
    - **skipfooter=0**：指定要跳过的文件末尾的行数。
    - **storage_options=None**：用于云存储的参数字典。
    - **dtype_backend=<no_default>**：指定数据类型后端。
    - **engine_kwargs=None**：传递给引擎的额外参数字典。

## 数据清洗

1. **缺失值处理**

    - **检测缺失值**：`df.isnull()` 或 `df.isna()`。

    - **删除缺失值**：`df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)`
       - `axis=0`：删除行，`axis=1`：删除列。
       - `how='any'`：只要有缺失值就删除，`how='all'`：全部为缺失值才删除。
       - `thresh=None`：保留至少有 `thresh` 个非缺失值的行或列。
       - `subset=None`：指定要考虑的列。
       - `inplace=False`：是否在原 DataFrame 上操作。

    - **替换缺失值**：`df.fillna(value=None, method=None, axis=None, inplace=False)`
       - `value=None`：要替换的值。
       - `method=None`：替换方法，`'ffill'`：用前一个值替换，`'bfill'`：用后一个值替换。
       - `axis=None`：替换方向，`axis=0`：按列替换，`axis=1`：按行替换。
       - `inplace=False`：是否在原 DataFrame 上操作。
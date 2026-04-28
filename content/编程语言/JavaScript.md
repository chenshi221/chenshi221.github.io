# JavaScript

## No.1 基本语法

1. JavaScript 是一种动态类型的弱类型语言，变量的类型在运行时可以被改变。
2. 语句以分号结尾，表达式不需要。
3. 变量提升：JavaScript 引擎在执行代码之前会先读取所有的变量声明，并将它们提升到作用域的顶部。==提升方式是将声明与初始化分开，声明会被提升，初始化不会被提升。==

    ```javascript
    if (true){
    var a=0;
        console.log(a);
    }
    var a=1
    console.log(a);
    ```

    ==上面的代码会输出 0 和 1，因为变量 a 被提升到了作用域的顶部。== 等价于：

    ```javascript
    var a;
    if (true){
        a=0;
        console.log(a);
    }
    a=1
    console.log(a);
    ```

4. var & let & const
   - var：声明的变量会被提升到函数作用域的顶部或全局作用域的顶部，但不会被初始化。
   - let：声明的变量不会被提升到作用域的顶部，而是在声明的位置初始化。
   - const：声明的变量必须被初始化，且不能被修改，不会被提升到作用域的顶部。

5. switch 语句，比较使用的是全等操作符（===）
6. if，for，while，do-while，switch 语法与 C 语言类似。

## No.2 数据类型

1. 基本数据类型：Number、String、Boolean、Null、Undefined、Symbol
2. 引用数据类型：Object、Array、Function
3. typeof 运算符可以返回一个变量的数据类型: `typeof x`
4. instanceof 运算符可以用来检测一个对象是否是一个类的实例：`object instanceof class`

### Null & Undefined & Boolean

1. null 是一个表示“空”的对象，转为数值时为 0；undefined 是一个表示”此处无定义”的原始值，转为数值时为 NaN。
2. 下列运算符会返回布尔值：

    ```javascript
    前置逻辑运算符： ! (Not)
    相等运算符：===，!==，==，!=
    比较运算符：>，>=，<，<=
    ```

    如果 JavaScript 预期某个位置应该是布尔值，会将该位置上现有的值自动转为布尔值。转换规则是除了下面六个值被转为 false，其他值都视为 true(包括空数组、空对象、空字符串等)：

    ```javascript
    undefined
    null
    false
    0
    NaN
    ""
    ''
    ```

### Number

1. JavaScript 只有一种数字类型，即 64 位双精度浮点数（IEEE 754 标准）。
2. 数值范围：`Number.MAX_VALUE`、`Number.MIN_VALUE`、`Number.POSITIVE_INFINITY`、`Number.NEGATIVE_INFINITY`、`Number.NaN`
3. NAN：`NaN` 是一个特殊的数值，表示“非数字”（Not-A-Number）。不等于任何值，包括它自己。
4. 数值转换：
    - parseInt()：将字符串转为整数，非字符串会先转为字符串，遇到非数字字符会停止转换，返回已经转换的部分。接收第二个参数，表示字符串按照什么进制转换。
    - parseFloat()：将字符串转为浮点数
    - isNaN()：判断一个值是否为 NaN

### String

1. 字符串是一种不可变的数据类型，一旦创建，它们的值就不能被改变。
2. 字符串的长度：`string.length`

### Object

1. 对象可以理解为 python 中的字典，键值对的集合。
2. 键名都是字符串，值可以是任意数据类型。
3. 添加属性：`object.key = value`
4. 删除属性：`delete object.key`， 不能删除继承来的属性
5. 对象间通过 `=` 赋值时，只是复制了引用，两个对象指向同一个内存地址。
6. 访问对象属性时，如果属性名是一个合法的变量名，可以直接用点运算符；否则用方括号：`object['key']`
7. `Object.keys()`：返回一个数组，包含对象自身的所有可枚举属性的键名。
8. `key in object`：判断对象是否包含某个属性，包括继承的属性。
9. for...in 循环有两个使用注意点。

    - 它遍历的是对象所有可遍历（enumerable）的属性，会跳过不可遍历的属性。
    - 它不仅遍历对象自身的属性，还遍历继承的属性。
    - 如果只想遍历对象自身的属性，不包括继承的属性，可以使用 `Object.keys()` 方法代替。或者使用 `object.hasOwnProperty(key)` 方法判断一个属性是对象自身的属性，还是继承的属性。
10. with 语句：将代码的作用域设置到一个特定的对象中，可以不用重复引用对象名。

    ```javascript
    var obj = {p: 1};
    with (obj) {
        p = 2;
    }
    ```

    ==with 不修改作用域链，所以在 with 代码块中定义的变量，还是在原来的作用域中。==

### Function

1. function 声明：`function name(parameters) { code }`
2. 函数表达式：`var name = function(parameters) { code }`
3. 函数声明提升：函数声明会被提升到作用域的顶部，可以在声明之前调用。
4. 使用 function 关键字声明的函数会被提升到作用域的顶部，而使用函数表达式声明的函数不会被提升。

5. 函数的属性与方法：

    - `name`：函数的名称
    - `length`：函数的参数个数
    - `toString()`：返回函数的源代码
    - `apply()`：调用函数，传递一个包含多个参数的数组
    - `call()`：调用函数，传递一个参数列表
    - `bind()`：创建一个新函数，绑定 this 对象，传递一个参数列表
    - `arguments`：函数内部的一个局部变量，包含调用函数时传递的参数列表

6. 作用域：JavaScript 只有两种作用域：全局作用域和函数作用域。

    - 全局作用域：在函数外部声明的变量，拥有全局作用域，可以在代码的任何地方被访问。
    - 函数作用域：在函数内部声明的变量，拥有函数作用域，可以在函数内部的任何地方被访问。

7. 参数传递：

    - 基本数据类型：传递的是值的副本，不会影响原始值。
    - 引用数据类型：传递的是对象地址，会改变原始值。

    ```javascript
    var obj = [1, 2, 3];
    function f(o) {
    o = [2, 3, 4];
    }
    f(obj);
    console.log(obj); // [1, 2, 3]
    ```

8. arguments 对象：包含函数调用时传递给函数的参数列表。

    - `arguments.length`：参数的个数
    - `arguments[0]`：第一个参数
    - `arguments.callee`：指向当前正在执行的函数
    - 将 arguments 对象转为数组：`var args = Array.prototype.slice.call(arguments)`

9. 闭包：java 使用链式作用域，即函数可以访问其外部作用域的变量，但是外部作用域不能访问函数内部的变量。即使其外部作用域被销毁，函数依然可以访问外部作用域的变量。

    ```javascript
    function f() {
    var n = 999;
    function g() {
        console.log(n);
    }
    return g;
    }
    var result = f();
    result(); // 999
    ```

    ==闭包的作用：1. 读取函数内部的变量；2. 让这些变量的值始终保持在内存中。==

    ```javascript
    function createIncrementor(start) {
    return function() {
        return start++;
    };
    }
    var inc = createIncrementor(5);
    console.log(inc()); // 5
    console.log(inc()); // 6
    ```

    inc 的存在依赖于 createIncrementor，所以 start 的值始终保持在内存中。

10. 闭包可以用于封装私有变量。

    ```javascript
    function Person(name) {
    var _name = name;
    function getName() {
        return _name;
    }
    return getName;
    }
    var p = Person('Tom');
    console.log(p.getName()); // Tom
    ```

11. IIFE：立即调用的函数表达式（Immediately-Invoked Function Expression）。

    ```javascript
    (function() {
    console.log('Hello World');
    })();
    ```

    ==IIFE 的作用：1. 不必为函数命名，避免污染全局变量；2. 立即执行，不必调用。==

### Array

1. 数组是一种特殊的对象，用来表示和操作一组数据。任何数据类型都可以放入数组。
2. 其键名是从 0 开始的整数，会被自动转为字符串。所以可以通过字符 0 访问第一个元素。
3. length 属性：
   - 会自动更新，值为最大整数属性名加 1，不一定等于数组的元素个数。
   - 可以手动设置，会删除大于等于 length 的键名。
   - 数组可以添加非整数键名，但不会计算在 length 属性中。
4. in 运算符：判断一个键名是否存在于数组中，包括继承的键名。
5. for...in 循环：会便利所有的键名，包括非整数键名和继承的键名。
6. 空位 & undefined：数组的空位不是 undefined，而是没有任何值。forEach 方法会跳过空位，但不会跳过 undefined。

7. forEach()：对数组的每个元素执行一次提供的函数。

    ```javascript
    var arr = [1, 2, 3];
    arr.forEach(function (value, index, array) {
    console.log(value, index, array);
    });
    ```

8. map()：对数组的每个元素执行一次提供的函数，返回一个新数组。

    ```javascript
    var arr = [1, 2, 3];
    var arr2 = arr.map(function (value, index, array) {
    return value * 2;
    });
    console.log(arr2); // [2, 4, 6]
    ```

9. filter()：对数组的每个元素执行一次提供的函数，返回一个新数组，只包含返回值为 true 的元素。

    ```javascript
    var arr = [1, 2, 3];
    var arr2 = arr.filter(function (value, index, array) {
    return value > 1;
    });
    console.log(arr2); // [2, 3]
    ```

10. reduce()：对数组的每个元素执行一次提供的函数，返回一个累计值。

    ```javascript
    var arr = [1, 2, 3];
    var sum = arr.reduce(function (prev, cur, index, array) {
    return prev + cur;
    });
    console.log(sum); // 6
    ```

11. some()：对数组的每个元素执行一次提供的函数，只要有一个返回值为 true，就返回 true。

    ```javascript
    var arr = [1, 2, 3];
    var result = arr.some(function (value, index, array) {
    return value > 2;
    });
    console.log(result); // true
    ```

## No.3 异步

1. JavaScript 是单线程的，同一时间只能做一件事。
2. 异步编程：通过回调函数、事件监听、Promise 等方式实现异步编程。
3. 回调函数：将函数作为参数传递给另一个函数，等到另一个函数执行完毕后再执行回调函数。

    ```javascript
    function f(callback) {
    //code
    callback();
    }
    f(function() {
    console.log('Hello World');
    });
    ```
4. 事件监听：通过事件监听函数，当事件发生时执行回调函数。

    ```javascript
    f1.on('done', f2);
    function f1() {
    setTimeout(function () {
        f1.trigger('done');
    }, 1000);
    }
    function f2() {
    console.log('Hello World');
    }
    ```
5. 发布/订阅模式：发布者发布事件，订阅者订阅事件，当事件发生时执行回调函数。

    ```javascript
    Jquery.subscribe('done', f2);
    function f1() {
    setTimeout(function () {
        Jquery.publish('done');
    }, 1000);
    }
# Java

## 3. 面向对象编程

### 3.1 面向对象基础

#### 3.1.1 方法

1. `objectname name = new objectname();` 创建对象, `name` 是对象的 ==引用==
2. public 类的文件名必须和类名相同
3. 字段(fild)和方法(method)是类的成员:
   1. private: 仅在类内部可见
   2. public: 从任何地方都可以访问
   3. protected: 和 private 类似, 但在子类中也可以访问
4. this: 即当前对象的引用
5. 可变参数: `public void add(int... a) { }` 可以传入任意个数的 int 参数, 传入后会被封装成数组
   传入 null 会被封装成长度为 0 的数组

6. 基本类参数和引用类参数的区别:
   1. 基本类参数传递的是值, 引用类参数传递的是引用
   2. 基本类参数的改变不会影响原来的值, 引用类参数的改变会影响原来的值

#### 3.1.2 构造方法

1. 构造方法的名字和类名相同，没有任何构造方法时，系统会自动创建一个无参构造方法
2. 构造方法没有返回值，未初始化的字段会被初始化为默认值(0, false, null)
3. 创建类对象的初始化顺序：
   1. Parent 静态初始化块（首次创建父类及其所有子类对象时）
   2. Child 静态初始化块（首次创建该类对象时）
   3. 创建 Child 对象
   4. Parent 实例初始化块
   5. Parent 构造函数
   6. Child 实例初始化块
   7. Child 构造函数

#### 3.1.3 方法重载(Overload)

1. 方法名相同，参数列表不同（参数个数、参数类型、参数顺序）

#### 3.1.4 继承

1. 子类继承父类的所有字段和方法，但不继承构造方法
2. 所有的类都继承自 Object 类，JAVA 中只支持单继承
3. 任何 class 的构造方法的第一行都是调用父类的构造方法，如果没有显示调用，会默认调用父类的无参构造方法

    ```java
    public class Child extends Parent {
        public Child() {
            super(); // 调用父类的构造方法
        }
    }
    ```

4. super: 调用父类的方法或字段
5. final: 修饰类，表示该类不能被继承；修饰方法，表示该方法不能被重写；修饰字段，表示该字段不能被修改
6. sealed: 修饰类，只能被指定的类继承，其他类不能继承

    ```java
    public sealed class Parent permits Child {
    }
    ```

7. 向上转型(Upcasting): 子类对象可以赋值给父类引用
8. instanceof: 判断对象是否是某个类的实例

#### 3.1.5 多态

1. 覆写(Override): 子类重写父类的方法

    ```java
    @Override
    public void method() {
        // 子类方法
    }
    ```

2. 覆写规则:
   1. 子类方法的访问权限不能小于父类方法的访问权限
   2. 子类方法的返回值类型必须和父类方法的返回值类型相同，或者是父类方法返回值类型的子类
   3. 子类方法的抛出异常不能大于父类方法的抛出异常

3. 运行时多态: 父类引用指向子类对象，调用方法时会调用子类的方法

    ```java
    Parent p = new Child();
    p.method(); // 调用子类的方法
    ```

4. 可以通过 super 调用父类的方法，包括覆写的方法

#### 3.1.6 抽象类

1. abstract: 用于修饰类和方法，抽象类不能被实例化，抽象方法没有方法体，必须在子类中实现

#### 3.1.7 接口

1. interface: 用于定义一组抽象方法，接口中的方法默认是 public abstract 的，不能定义实例字段
2. 类只能继承一个类，但可以实现多个接口
3. 实现接口的类必须实现接口中的所有方法，使用 implements 关键字

    ```java
    public class Child implements Interface {
        @Override
        public void method() {
            // 实现接口方法
        }
    }
    ```

4. default: 用于接口中的方法，可以有方法体，实现类可以不实现该方法，此方法 ==不能访问实例字段==

    ```java
    public interface Interface {
        default void method() {
            // 接口方法
        }
    }
    ```

#### 3.1.8 静态字段和方法

1. ==公共类的 main 方法必须是静态方法，因为在调用 main 方法时，类还没有实例化==

2. interface 中的字段默认且只能是 public static final，且必须初始化

#### 3.1.9 包

1. package: 用于定义类的命名空间，避免类名冲突，在类的第一行声明
2. 完整类名: 包名+类名
3. 包的文件结构有父子关系，但作用域没有父子关系
4. 位于同一个包中的类可以直接访问，不用 public、protected、private 修饰的字段和方法就是包作用域
5. import: 用于导入包，可以导入指定类，也可以导入整个包

    ```java
    import java.util.ArrayList;// 导入ArrayList类
    import java.util.*; // 导入java.util包下的所有类
    ```

6. class 查找顺序：
    1. 按照完整类名查找
    2. 查找当前 package 下的类
    3. 查找 import 导入的类
    4. 查找 java.lang 包下的类

#### 3.1.10 作用域

1. public: 任何类都可以访问，包括其他包中的类
2. private: 仅在类内部可见，包括此类的内部类
3. protected: 在同一个包内可见，其他包中的子类也可以访问
4. 包作用域: 没有用 public、protected、private 修饰的字段和方法，只能在同一个包中访问

#### 3.1.11 内部类

1. 内部类可以访问外部类的字段和方法，包括 private 字段和方法
2. 内部类实例必须依赖外部类实例，可以通过 `outer.new Inner()` 创建内部类实例

#### 3.1.12 classpath 和 jar

1. classpath: 用于指定类的搜索路径，可以是目录或 jar 文件，默认是当前目录
2. 使用命令行参数 `-classpath` 或 `-cp` 指定 classpath

    ```shell
    java -classpath .:/path/to/classes:/path/to/lib/* Main
    ```

3. 执行某个包内的 class 文件时，需要指定包名

    如执行当前目录下：`\abc\Main.class`，需要执行 `java abc.Main`

    ```shell
    java -cp . abc.Main
    ```

4. jar: Java 的压缩包，实际上是 zip 格式，可以通过压缩成 zip 文件，然后改后缀名为 jar

    ```shell
    java -cp ./hello.jar abc.xyz.Hello
    ```

#### 3.1.13 class 版本

1. class 文件的版本号：major_version.minor_version
2. 高版本的 JVM 可以运行低版本的 class 文件，但不能运行高版本的 class 文件
3. 指定编译输出

    ```shell
    javac --release 8 Main.java // 指定编译输出为JDK8
    javac --source 9 --target 11 Main.java // 指定java源码版本为JDK9，输出的class目标版本为JDK11
    ```

#### 3.1.14 模块

1. module: 用于定义一组包，描述模块信息，包括模块名、依赖的模块、导出的包等
2. module-info.java: 用于定义模块信息，必须在模块根目录下，模块名和目录名必须一致

    ```java
    module hello {
        requires java.base;
        exports abc;
    }
    ```

### 3.2 Java 核心类

#### 3.2.1 字符串和编码

1. String 是引用类型，==不可变==，==字符串相等比较用 equals 方法==
2. 不可变是通过内部 ==private final char [] value== 实现的
3. JVM 在编译器会将出现的字符串常量放入常量池，==相同的字符串常量只会在常量池中存储一份==
4. `contains()` 方法判断是否包含子串
5. `indexOf()` 方法查找子串的位置
6. `lastIndexOf()` 方法查找子串的最后位置
7. `startsWith()` 方法判断是否以子串开头
8. `endsWith()` 方法判断是否以子串结尾
9. `substring()` 方法截取子串
10. `trim()` 方法去除首尾空格
11. `replace()` 方法替换子串, `replaceAll()` 方法支持正则表达式
12. 类型转化

    ```java
    int n = 123;
    String s = String.valueOf(n); // "123"
    int x = Integer.parseInt(s); // 123
    ```

#### 3.2.2 StringBuilder

1. java 可以用 `+` 连接字符串，但是效率低，因为每次连接都会创建新的字符串对象
2. StringBuilder 是可变对象，==线程不安全==

    ```java
    StringBuilder sb = new StringBuilder();
    sb.append("Hello");
    sb.append(" ");
    sb.append("world");
    String s = sb.toString(); // "Hello world"
    ```

#### 3.2.3 StringJoiner

1. `StringJoiner(分割符, 前缀, 后缀)`

    ```java
    StringJoiner sj = new StringJoiner(", ", "[", "]");
    sj.add("A");
    sj.add("B");
    sj.add("C");
    String result = sj.toString(); // "[A, B, C]"
    ```

2. `String.join(分割符, 字符串数组)`

    ```java
    String[] array = new String[] { "A", "B", "C" };
    String result = String.join(", ", array); // "A, B, C"
    ```

#### 3.2.4 包装类型

1. 基本类型有对应的包装类型，==包装类型是引用类型==

    | 基本类型 | 包装类型 |
    | -------- | -------- |
    | byte     | java.lang.Byte     |
    | short    | java.lang.Short    |
    | int      | java.lang.Integer  |
    | long     | java.lang.Long     |
    | float    | java.lang.Float    |
    | double   | java.lang.Double   |
    | char     | java.lang.Character|
    | boolean  | java.lang.Boolean  |

2. 创建包装类型对象

    ```java
    Integer n = new Integer(123); // 123
    Integer n = Integer.valueOf(123); // 123
    ```

3. 自动装箱和拆箱

    ```java
    Integer n = 123; // 自动装箱
    int x = n; // 自动拆箱
    ```

4. 所有的包装类型都是不可变对象，内部的 value 都是 private final 的
5. 对于包装类型，应该使用 equals 方法比较，而不是 `==`
6. 为了节省内存，Integer.valueOf()对于较小的数，始终返回相同的实例

7. 静态工厂方法：能创建“新”对象，但不一定每次都创建新对象，会尽可能重用已有对象

#### 3.2.5 JavaBean

1. JavaBean 是一种符合命名规范的类，==必须有一个无参构造方法==，字段使用 private 修饰，提供 getter 和 setter 方法

    ```java
    public class Person {
        private String name;
        private int age;
    
        public Person() {
        }
    
        public String getName() {
            return name;
        }
    
        public void setName(String name) {
            this.name = name;
        }
    
        public int getAge() {
            return age;
        }
    
        public void setAge(int age) {
            this.age = age;
        }
    }
    ```

#### 3.2.6 枚举类

1. enum: 用于定义枚举类型，枚举类型是一种特殊的类，可以定义字段、方法，也可以实现接口

    ```java
    public enum Weekday {
        MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY;
    }
    ```

2. 枚举类的构造方法默认是 private，字段默认是 public static final

3. `name()` 方法返回枚举常量的名字，`ordinal()` 方法返回枚举常量的序号

#### 3.2.7 常用工具类

1. Math: 数学计算

    * `Math.abs()` 绝对值
    * `Math.max()` 最大值
    * `Math.min()` 最小值
    * `Math.pow()` 幂运算
    * `Math.sqrt()` 平方根
    * `Math.random()` 随机数，返回 0~1 之间的 double

        ```java
        double y = Math.random() * (max - min) + min; // [min, max)
        ```

2. Random: 伪随机数生成器

    * `nextInt()` 生成 int 随机数
    * `nextDouble()` 生成 double 随机数
    * `nextBoolean()` 生成 boolean 随机数
    * `nextBytes(byte[] bytes)` 生成随机字节数组

        ```java
        Random r = new Random();// 默认以当前时间为种子
        int n = r.nextInt(100); // [0, 100)
        ```

## 4. 异常处理

### 4.1 Java 异常

1. 异常继承关系

    ```mermaid
    graph LR
    Class --> Throwable
    Throwable --> Error
    Throwable --> Exception
    Error --> OutOfMemoryError
    Error --> StackOverflowError
    Exception --> RuntimeException
    RuntimeException --> NullPointerException
    RuntimeException --> IllegalArgumentException
    RuntimeException --> IndexOutOfBoundsException
    Exception --> IOException
    ```

2. Exception 及其子类是必须捕获的异常，但不包括 RuntimeException 及其子类

3. try-catch-finally

    ```java
    try {
        // 可能抛出异常的代码
    } catch (Exception e) {
        // 捕获异常
    } finally {
        // 无论是否发生异常都会执行
    }
    ```

4. 在方法定义的时候，可以声明可能抛出的异常，调用者必须处理这些异常

    ```java
    public void readFile(String file) throws IOException {
        // 读取文件
    }
    ```

5. 未捕获的异常会向上传递，直到 main 方法，如果 main 方法也没有捕获，==程序会终止==

### 4.2 捕获异常

1. 可以有多个 catch 块，按照 catch 块的顺序捕获异常，但只有一个 catch 块会被执行

2. catch 块可以捕获异常的子类，但必须先捕获子类异常，再捕获父类异常

3. catch 块可以捕获多个异常，用 `|` 分隔

    ```java
    try {
        // 可能抛出异常的代码
    } catch (FileNotFoundException | IOException e) {
        // 捕获异常
    }
    ```

### 4.3 抛出异常

1. 可以通过 `printStackTrace()` 方法打印异常堆栈信息
2. 可以通过 `throw` 关键字抛出异常

    ```java
    throw new IllegalArgumentException("Invalid argument");
    ```

3. 异常可以作为方法的参数传递，来保证调用栈的清晰

4. 在 try 块或者 catch 块中使用 throw 抛出异常，finally 块中的代码依然会执行，然后再抛出异常

5. 如果在 finally 块中抛出异常，会覆盖 try 块中的异常，导致 try 块中的异常被忽略

### 4.4 自定义异常

1. 继承 Exception 或 RuntimeException

    ```java
    public class BaseException extends RuntimeException {
        public BaseException() {
            super();
        }
    
        public BaseException(String message, Throwable cause) {
            super(message, cause);
        }
    
        public BaseException(String message) {
            super(message);
        }
    
        public BaseException(Throwable cause) {
            super(cause);
        }
    }
    ```

### 4.5 JDK Logging

1. java.util.logging: JDK 自带的日志库，使用 Logger 对象记录日志

    ```java
    import java.util.logging.Logger;
    
    public class Main {
        public static void main(String[] args) {
            Logger logger = Logger.getGlobal();
            logger.info("Start process...");
            try {
                int n = 100 / 0;
            } catch (Exception e) {
                logger.warning("Exception: " + e);
            }
        }
    }
    ```

## 5. 反射(Reflection)

1. 反射是指程序在运行期可以拿到一个对象的所有信息，包括字段、方法、构造方法等

### 5.1 Class 类

1. Class 类是反射的核心类，==一个类在 JVM 中只有一个 Class 实例==
2. Class 类由 JVM 在 ==首次== 加载类时自动创建，读取 `类名.class` 文件到内存中
3. Class 实例包含了类的所有信息：类名，包名，字段，方法，构造方法，父类，接口等
4. 获取 Class 实例的方法：

    ```java
    Class cls1 = String.class; // 直接通过类名获取
    Class cls2 = "hello".getClass(); // 通过实例获取
    Class cls3 = Class.forName("java.lang.String"); // 通过类名获取
    ```

5. ==数组类型== 的 Class 实例和 ==基本类型== 的 Class 实例不同
6. 可以通过 `cls.newInstance()` 创建实例，但是只能调用无参构造方法
7. class 是动态加载的

### 5.2 访问字段

1. Field: 通过 Class 实例可以获取字段信息，包括字段名，字段类型，字段值

    * `Field getField(String name)` 获取 public 字段，包括父类的 public 字段
    * `Field getDeclaredField(String name)` 获取本类的字段，包括 private 字段，但不包括父类的字段
    * `Field[] getFields()` 获取所有 public 字段，包括父类的 public 字段
    * `Field[] getDeclaredFields()` 获取本类的所有字段，包括 private 字段，但不包括父类的字段

2. Field 对象包含字段的所有信息

    * `getName()` 获取字段名
    * `getType()` 获取字段类型，返回 Class 实例
    * `getModifiers()` 获取字段修饰符，返回 int

3. 获取字段值

    * `Object get(Object instance)` 获取字段值

    ```java
    public static void main(String[] args) throws Exception {
        Object p = new Person("Xiao Ming");
        Class c = p.getClass();
        Field f = c.getDeclaredField("name");
        f.setAccessible(true); // 设置为可访问
        Object value = f.get(p);
        System.out.println(value); // "Xiao Ming"
    }
    ```

4. 设置字段值

    * `void set(Object instance, Object value)` 设置字段值

    ```java
    public static void main(String[] args) throws Exception {
        Object p = new Person("Xiao Ming");
        Class c = p.getClass();
        Field f = c.getDeclaredField("name");
        f.setAccessible(true); // 设置为可访问
        f.set(p, "Xiao Hong");
        System.out.println(p); // "Xiao Hong"
    }
    ```

### 5.3 调用方法

1. Method: 通过 Class 实例可以获取方法信息，包括方法名，参数类型，返回类型

    * `Method getMethod(String name, Class<?>... parameterTypes)` 获取 public 方法，包括父类的 public 方法
    * `Method getDeclaredMethod(String name, Class<?>... parameterTypes)` 获取本类的方法，包括 private 方法，但不包括父类的方法
    * `Method[] getMethods()` 获取所有 public 方法，包括父类的 public 方法
    * `Method[] getDeclaredMethods()` 获取本类的所有方法，包括 private 方法，但不包括父类的方法
  
2. Method 对象包含方法的所有信息

    * `getName()` 获取方法名
    * `getReturnType()` 获取返回类型，返回 Class 实例
    * `getParameterTypes()` 获取参数类型，返回 Class 数组
    * `getModifiers()` 获取方法修饰符，返回 int

3. `public Object invoke(Object instance, Object... args)` 调用方法，==通过 invoke 方法的返回值都是 Object 类型==

    ```java
    public static void main(String[] args) throws Exception {
        Object p = new Person("Xiao Ming");
        Class c = p.getClass();
        Method m = c.getDeclaredMethod("setName", String.class);
        m.invoke(p, "Xiao Hong");
        System.out.println(p); // "Xiao Hong"
    }
    ```

4. `invoke()` 调用静态方法时，==instance 传入 null==
5. `invoke()` 调用 private 方法时，==需要先调用 `setAccessible(true)`==
6. 使用反射调用方法时，仍然遵循多态原则：即总是调用实际类型的覆写方法（如果存在）

### 5.4 调用构造方法

1. `Class.newInstance(Object... args)` 调用构造方法，==但是只能调用无参构造方法==

2. Constructor: 通过 Class 实例可以获取构造方法信息，包括构造方法名，参数类型

    * `Constructor getConstructor(Class<?>... parameterTypes)` 获取 public 构造方法
    * `Constructor getDeclaredConstructor(Class<?>... parameterTypes)` 获取本类的构造方法，包括 private 构造方法
    * `Constructor[] getConstructors()` 获取所有 public 构造方法
    * `Constructor[] getDeclaredConstructors()` 获取本类的所有构造方法，包括 private 构造方法

### 5.5 获取继承关系

1. `Class getSuperclass()` 获取父类的 Class 实例
2. `Class[] getInterfaces()` 获取接口的 Class 实例数组，不包括父类实现的接口

### 5.6 动态代理

1. 在运行期动态创建一个 interface 实例的方法如下：

    1. 定义一个 InvocationHandler 实例，它负责实现接口的方法调用；
    2. 通过 Proxy.newProxyInstance()创建 interface 实例，它需要 3 个参数：
        1. 使用的 ClassLoader，通常就是接口类的 ClassLoader；
        2. 需要实现的接口数组，至少需要传入一个接口进去；
        3. 用来处理接口方法调用的 InvocationHandler 实例。
    3. 将返回的 Object 强制转型为接口。

    ```java
    public class Main {
        public static void main(String[] args) {
            InvocationHandler handler = new MyInvocationHandler(...);
            MyInterface myInterface = (MyInterface) Proxy.newProxyInstance(
                Main.class.getClassLoader(),
                new Class[] { MyInterface.class },
                handler);
            myInterface.method();
        }
    }
    ```

## 6. 注解(Annotation)

### 6.1 注解

1. 注解是一种用来为程序元素（类、方法、字段等）设置元数据的方法，会被打包到 class 文件中，可以通过反射读取
2. 注解的分类：

    1. 内置注解：如@Override、@Deprecated、@SuppressWarnings，不会编译到 class 文件中
    2. 程序运行期可以读取的注解：如@Retention(RetentionPolicy.RUNTIME)的注解，加载后会一直存在在 JVM 中

3. 注解参数必须是常量，包括基本类型、String、枚举、注解、以上类型的数组

### 6.2 定义注解

1. 使用@interface 定义注解

    ```java
    public @interface MyAnnotation {
        String name();
        int age() default 18;
    }
    ```

2. 元注解：用于修饰注解的注解

    1. @Retention: 用于指定注解的生命周期
        1. RetentionPolicy.SOURCE: 编译器直接丢弃这种注解
        2. RetentionPolicy.CLASS: 编译器把注解记录在 class 文件中，但是运行时无法获得
        3. RetentionPolicy.RUNTIME: 编译器把注解记录在 class 文件中，运行时可以通过反射获取

    2. @Target: 用于指定注解可以修饰的目标类型
        1. ElementType.TYPE: 类、接口、枚举
        2. ElementType.FIELD: 字段
        3. ElementType.METHOD: 方法
        4. ElementType.PARAMETER: 方法参数
        5. ElementType.CONSTRUCTOR: 构造方法
        6. ElementType.LOCAL_VARIABLE: 局部变量
        7. ElementType.ANNOTATION_TYPE: 注解
        8. ElementType.PACKAGE: 包

    3. @Repeatable: 用于指定注解可以重复修饰
    4. @Inherited: 用于指定子类是否继承父类的注解

### 6.3 处理注解

1. 注解继承自 `java.lang.annotation.Annotation` 接口，使用反射来读取注解

2. 判断某个注解是否存在于 Class、Field、Method 或 Constructor：

    * Class.isAnnotationPresent(Class)
    * Field.isAnnotationPresent(Class)
    * Method.isAnnotationPresent(Class)
    * Constructor.isAnnotationPresent(Class)

3. 读取注解：

    * Class.getAnnotation(Class)
    * Field.getAnnotation(Class)
    * Method.getAnnotation(Class)
    * Constructor.getAnnotation(Class)

4. 使用注解：

    ```java
    static void Check(Person person) throws IllegalArgumentException, IllegalAccessException {
        for (Field field : person.getClass().getDeclaredFields()) {
            Range range = field.getAnnotation(Range.class);
            if (range != null) {
                field.setAccessible(true); // 允许访问私有字段
                if (field.getName().equals("name")){
                    String name = (String) field.get(person); // 正确获取字段值
                    if (name.length() < range.min() || name.length() > range.max()) {
                        throw new IllegalArgumentException("Name is out of range: " + name);
                    }
                } else if (field.getType() == int.class) {
                    int age = (int) field.get(person); // 正确获取字段值
                    if (age < range.min() || age > range.max()) {
                        throw new IllegalArgumentException("Age is out of range: " + age);
                    }
                }
            }
        }
    }
    ```

## 7. 泛型(Generics)

### 7.1 泛型

1. 泛型是指定义类、接口、方法时使用类型参数，==在使用时传入具体的类型==

### 7.3 编写泛型

1. 泛型类

    ```java
    public class Pair<T> {
        private T first;
        private T second;

        public Pair(T first, T second) {
            this.first = first;
            this.second = second;
        }

        public T getFirst() {
            return first;
        }

        public T getSecond() {
            return second;
        }
    }
    ```

2. 泛型类型不能用于静态字段和静态方法，因为泛型类型在实例化时才确定，==而静态字段和静态方法在类加载时就已经存在==
3. 对于静态方法，可以定义为泛型方法

    ```java
    public class Pair<T> {
        private T first;
        private T second;
    
        public static <K> Pair<K> create(K first, K second) {
            return new Pair<>(first, second);
        }
    }
    ```

### 7.4 擦除法

1. java 实现泛型是通过 ==擦除法== 实现的，即在编译器处理泛型，JVM 对泛型一无所知，无法得到泛型 T 的类型等信息
2. 所有的泛型类型参数都会在编译时被替换成它们的上界（bound），如果没有明确指定上界，默认是 Object

    ```java
    //设置上界
    public class Pair<T extends Comparable> {
        private T first;
        private T second;
    }
    ```

3. ==Java 不支持基于泛型的方法重载==，因为擦除后都是上界类型
4. 泛型的缺点：
   1. 因为 Java 泛型的上界是 Object，所以 ==不能使用基本类型==，需要使用包装类型
   2. 无法获取泛型的实际类型
   3. 无法判断带泛型的类型，泛型类只有一个 Class 实例
   4. 不能实例化泛型类型，实例化需要 ==使用反射来实现== `Class.newInstance`，使用的时候需要传入 Class 实例

        ```java
        public class Pair<T> {
            private T first;
        
            public Pair(Class<T> clazz) throws Exception {
                this.first = clazz.newInstance();
            }
        }
        
        Pair<String> p = new Pair<>(String.class);
        ```

5. 泛型继承：类可以继承泛型类，但是需要指定泛型类型，并且子类能获取到父类的泛型类型

    ```java
    public class Child extends Pair<String> {
        public Child() {
            super(String.class);
        }
    }
    ```

### 7.5 extends 通配符

1. extends 的目的是允许参数为指定类型或其子类
2. Pair \<Integer\> 不是 Pair \<Number\> 的子类
3. 使用 extends 通配符可以解决这个问题

    ```java
    public static void print(Pair< ? extends Number> p) {
        Number first = p.getFirst();
        Number second = p.getSecond();
        System.out.println(first + ", " + second);
    }
    ```

4. 通配符 extends 只允许读取，不允许写入，==因为通配符类型是不确定的==，除了 null

    > 因 Java 中的泛型是在编译时执行类型检查，Java 会确保你不会对泛型执行不安全的操作。

5. extends 通配符作为参数可以保证该方法只会读取数据，不会修改数据，是安全的

6. extends 可以限制泛型的上界，super 可以限制泛型的下界

### 7.6 super 通配符

1. super 的目的是允许参数为指定类型或其父类

2. 因为向上转型是安全的，所以 super 通配符可以写入

    ```java
    public static void set(Pair<? super Integer> p) {
        p.setFirst(1);
        p.setSecond(2);
    }
    ```

3. super 通配符作为参数可以保证该方法只会写入数据，不会读取数据，因为可能会向下转型，是不安全的

4. 下面是 `Collections.copy()` 方法的实现，可以看到使用了 super 和 extends 通配符

    ```java
    public static <T> void copy(List<? super T> dest, List<? extends T> src) {
        for (int i = 0; i < src.size(); i++) {
            T t = src.get(i);
            dest.add(t);
        }
    }
    ```

    * dest 使用 super 通配符，表示对 dest 只能写入，不能读取
    * src 使用 extends 通配符，表示对 src 只能读取，不能写入，==保证了 copy 方法的安全性==
    * super+extends 通配符保证了 dest 一定是 src 的同类及父类，向上转型是安全的

5. PECS 原则：Producer Extends, Consumer Super

    * 如果需要返回 T，使用 extends 通配符
    * 如果需要写入 T，使用 super 通配符

6. 无限定通配符：`<?>`，表示任意类型，但是不能写入，也不能读取，只能用于 ==类型检查==，同时 `<?>` 是所有泛型类型的父类

    > 不能读取实际上能获取 Object 类型，但是不能强制转换为其他类型
    > 不能写入实际上能写入 null，因为 null 是任意类型的子类

### 7.7 泛型与反射

1. 我们可以声明带泛型的数组，但不能用 new 操作符创建带泛型的数组，必须使用强制转型

    ```java
    Pair<String>[] ps = (Pair<String>[]) new Pair[10];
    ```

2. 协变数组：数组是协变的，即子类数组可以赋值给父类数组，但是这是危险的，因为数组是可变的

    ```java
    Person[] ps = new Student[10];
    ps[0] = new Person(); // 运行时错误
    ```

## 8. 集合(Collection)

### 8.1 Collection 接口

1. Collection 是集合接口，定义了集合的基本操作，包括添加、删除、遍历等

    ```java
    Collection<String> list = new ArrayList<>();
    list.add("apple");
    list.add("banana");
    list.add("cherry");
    for (String s : list) {
        System.out.println(s);
    }
    ```

2. Collection 接口继承自 Iterable 接口，可以使用 foreach 循环遍历

3. Collection 接口定义了一些操作方法

    * `boolean add(E e)` 添加元素
    * `boolean remove(Object o)` 删除元素
    * `boolean contains(Object o)` 判断是否包含元素
    * `int size()` 获取元素个数
    * `void clear()` 清空所有元素
    * `boolean isEmpty()` 判断是否为空
    * `Object[] toArray()` 转换为数组
    * `boolean addAll(Collection<? extends E> c)` 添加多个元素
    * `boolean removeAll(Collection<?> c)` 删除多个元素
    * `boolean retainAll(Collection<?> c)` 保留多个元素
    * `boolean containsAll(Collection<?> c)` 判断是否包含多个元素

### 8.2 List 接口

1. List 是有序集合，可以重复，可以通过索引访问元素，内部按照 ==插入顺序== 存储

2. `ArrayList` 是动态数组，内部使用数组存储，支持随机访问，==插入和删除元素效率低==

3. `LinkedList` 是双向链表，插入和删除元素效率高，但是随机访问效率低

4. 创建 list

    ```java
    List<String> list = new ArrayList<>();
    List<String> list = new LinkedList<>();
    List<Integer> list = List.of(1, 2, 3);  //不支持传入null，不可变集合
    ```

5. List 接口定义了一些操作方法

    * `void add(int index, E element)` 在指定位置插入元素
    * `void add(E element)` 在末尾插入元素
    * `E get(int index)` 获取指定位置的元素
    * `int indexOf(Object o)` 查找元素的位置，返回第一次出现的位置
    * `int lastIndexOf(Object o)` 查找元素的最后位置
    * `E remove(int index)` 删除指定位置的元素
    * `E set(int index, E element)` 替换指定位置的元素
    * `boolean contains(Object o)` 判断是否包含元素

6. 遍历 List，建议使用 Iterator，因为 Iterator 总是有最高的性能。

    * `boolean hasNext()` 判断是否有下一个元素
    * `E next()` 获取下一个元素

    ```java
    List<String> list = new ArrayList<>();
    list.add("apple");
    list.add("banana");
    list.add("cherry");
    for (Iterator<String> it = list.iterator(); it.hasNext(); ) {
        String s = it.next();
        System.out.println(s);
    }
    ```

7. 也可以使用 foreach 循环遍历，但不能在过程中删除或添加元素，也无法获取当前元素的索引

    ```java
    for (String s : list) {
        System.out.println(s);
    }
    ```

8. List 转换为数组

    * `Object[] toArray()` 转换为 Object 数组，但会丢失类型信息
    * `T[] toArray(T[] a)` 转换为指定类型的数组，如果数组长度不够，会创建一个新数组
    * `T[] toArray(IntFunction<T[]> generator)` 转换为指定类型的数组，如果数组长度不够，会调用 generator 创建一个新数组

    ```java
    List<String> list = List.of("apple", "banana", "cherry");
    Object[] array1 = list.toArray();
    String[] array2 = list.toArray(new String[3]);
    String[] array3 = list.toArray(String[]::new);
    ```

9. 数组转换为 List

    * `List.of(T... elements)` 创建一个不可变的 List
    * `Arrays.asList(T... a)` 创建一个可变的 List，但是不能添加或删除元素
    * `Collections.addAll(Collection<? super T> c, T... elements)` 添加元素到 List
    * `stream(array).collect(Collectors.toList())` 使用 Stream 转换为 List

    ```java
    String[] array = new String[] { "apple", "banana", "cherry" };
    List<String> list1 = List.of(array);
    List<String> list2 = new ArrayList<>(Arrays.asList(array));
    List<String> list3 = new ArrayList<>();
    Collections.addAll(list3, array);
    List<String> list4 = Arrays.stream(array).collect(Collectors.toList());
    ```

### 8.3 编写 equals 方法

1. List 内部使用 `equals()` 方法判断两个元素是否相等，对于基本类型，Java 标准库提供了 `equals()` 方法
2. 对应自定义类型，需要重写 `equals()` 方法，要求满足如下要求

    1. 自反性：对于任意非 null 的引用值 x，x.equals(x)必须返回 true
    2. 对称性：对于任意非 null 的引用值 x 和 y，当且仅当 y.equals(x)返回 true 时，x.equals(y)必须返回 true
    3. 传递性：对于任意非 null 的引用值 x、y 和 z，如果 x.equals(y)返回 true，并且 y.equals(z)返回 true，那么 x.equals(z)必顋返回 true
    4. 一致性：对于任意非 null 的引用值 x 和 y，只要 equals 的比较操作在对象中所用的信息没有被修改，多次调用 x.equals(y)就会一致地返回 true，或者一致地返回 false
    5. 对于任意非 null 的引用值 x，x.equals(null)必须返回 false

3. 事实上我们只需要对 ==引用类型== 使用 `Objects.equals()` 方法，==基本类型== 使用 `==` 比较即可

    ```java
    public class Person {
        private String name;
        private int age;
    
        @Override
        public boolean equals(Object o) {
            if (o instanceof Person) {
                Person p = (Person) o;
                return Objects.equals(name, p.name) && age == p.age;
            }
            return false;
        }
    }
    ```

### 8.4 Map 接口

1. Map 是键值对集合，==键不能重复==，值可以重复，内部按照 ==键的哈希值== 存储

2. Map 提供的常用方法

    * `V put(K key, V value)` 添加键值对，如果 key 已经存在，会替换 value，并返回旧的 value，否则返回 null
    * `V get(Object key)` 获取 key 对应的 value，如果 key 不存在，返回 null
    * `boolean containsKey(Object key)` 判断是否包含 key
    * `boolean containsValue(Object value)` 判断是否包含 value
    * `V remove(Object key)` 删除 key 对应的 value，返回被删除的 value
    * `int size()` 获取元素个数
    * `void clear()` 清空所有元素
    * `boolean isEmpty()` 判断是否为空
    * `Set<K> keySet()` 获取所有 key 的集合
    * `Collection<V> values()` 获取所有 value 的集合
    * `Set<Map.Entry<K, V>> entrySet()` 获取所有键值对的集合
    * `V getOrDefault(Object key, V defaultValue)` 获取 key 对应的 value，如果 key 不存在，返回 defaultValue
    * `V putIfAbsent(K key, V value)` 添加键值对，如果 key 已经存在，不替换 value

3. Map 遍历

    * 遍历 key

        ```java
        Map<String, Integer> map = new HashMap<>();
        map.put("apple", 1);
        map.put("banana", 2);
        map.put("cherry", 3);
        for (String key : map.keySet()) {
            System.out.println(key + ": " + map.get(key));
        }
        ```

    * 遍历 value

        ```java
        for (Integer value : map.values()) {
            System.out.println(value);
        }
        ```

    * 遍历键值对

        ```java
        for (Map.Entry<String, Integer> entry : map.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
        ```

### 8.5 编写 hashcode 方法

1. hashMap 的步骤：

    1. 计算 key 的 hashCode
    2. 根据 hashCode 计算存储位置
    3. 如果位置为空，直接存储
    4. 如果位置不为空，==判断 key 是否相等==，如果相等，替换 value，否则，==使用链表或红黑树解决冲突==

2. 因为上面的步骤，==key 的 hashCode 和 equals 方法必须正确实现==，否则会导致 key 无法正确存储和查找

3. 编写 hashCode 方法使用 Objects.hash()方法:

    ```java
    @Override
    public int hashCode() {
        return Objects.hash(name, age);
    }
    ```

4. HashMap 的默认容量是 16，负载因子是 0.75，当元素个数超过容量*负载因子时，会自动扩容

### 8.5 EnumMap

1. EnumMap 是一种特殊的 Map，==key 是枚举类型==，内部使用数组存储，无需计算 hashCode

    ```java
    public enum Weekday {
        MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY;
    }
    
    Map<Weekday, String> map = new EnumMap<>(Weekday.class);
    map.put(Weekday.MONDAY, "星期一");
    map.put(Weekday.TUESDAY, "星期二");
    map.put(Weekday.WEDNESDAY, "星期三");
    ```

### 8.6 TreeMap

1. TreeMap 是一种有序的 Map，接口是 `SortedMap`，实现类是 `TreeMap`
2. TreeMap 内部使用 ==红黑树== 存储，==key 必须实现 Comparable 接口==，或者在创建 TreeMap 时传入 Comparator

    ```java
    Map<String, Integer> map = new TreeMap<>();
    map.put("apple", 1);
    map.put("banana", 2);
    map.put("cherry", 3);
    ```

    ```java
    Map<Person, Integer> map = new TreeMap<>(new Comparator<Person>() {
            public int compare(Person p1, Person p2) {
                return p1.name.compareTo(p2.name);
            }
        });
    ```

3. 使用 compareTo 方法比较两个对象，==返回负数表示小于，0 表示等于，正数表示大于==，要能保证相等的对象返回 0

### 8.7 Properties

1. Properties 是一种特殊的 Map，==key 和 value 都是 String 类型==，通常用于读取配置文件

2. Java 配置文件以 `.properties` 为扩展名，格式是 `key=value`，注释以 `#` 开头

    ```java
    # config.properties
    url=jdbc:mysql://localhost:3306/test
    username=root
    password=123456
    ```

3. 使用 Properties 读取配置文件

    ```java
    String configFile = "config.properties";
    Properties props = new Properties();
    try (InputStream input = new FileInputStream(configFile)) {
        props.load(input);
        String url = props.getProperty("url");
        String username = props.getProperty("username");
        String password = props.getProperty("password");
    }
    ```

4. 使用 Properties 写入配置文件

    ```java
    String configFile = "config.properties";
    Properties props = new Properties();
    props.setProperty("url", "jdbc:mysql://localhost:3306/test");
    props.setProperty("username", "root");
    props.setProperty("password", "123456");
    ```

### 8.8 Set 接口

1. Set 是一种不允许重复元素的集合，内部使用 ==哈希表== 存储

2. Set 接口定义了一些操作方法

    * `boolean add(E e)` 添加元素
    * `boolean remove(Object o)` 删除元素
    * `boolean contains(Object o)` 判断是否包含元素
    * `int size()` 获取元素个数
    * `void clear()` 清空所有元素
    * `boolean isEmpty()` 判断是否为空
    * `Object[] toArray()` 转换为数组
    * `boolean addAll(Collection<? extends E> c)` 添加多个元素
    * `boolean removeAll(Collection<?> c)` 删除多个元素
    * `boolean retainAll(Collection<?> c)` 保留多个元素
    * `boolean containsAll(Collection<?> c)` 判断是否包含多个元素

3. Set 的实现类有 `HashSet` 和 `TreeSet`

    1. HashSet 是无序 Set 的一种实现，内部使用 ==HashMap== 存储，==key 是元素，value 是一个固定对象==

        ```java
        public class HashSet<E> implements Set<E> {
            // 持有一个HashMap:
            private HashMap<E, Object> map = new HashMap<>();

            // 放入HashMap的value:
            private static final Object PRESENT = new Object();

            public boolean add(E e) {
                return map.put(e, PRESENT) == null;
            }

            public boolean contains(Object o) {
                return map.containsKey(o);
            }

            public boolean remove(Object o) {
                return map.remove(o) == PRESENT;
            }
        }
        ```

    2. TreeSet 是有序 Set 的一种实现，内部使用 ==TreeMap== 存储，==key 是元素，value 是一个固定对象==

### 8.9 Queue 接口

1. Queue 是一种先进先出的队列，内部使用 ==链表== 存储

2. Queue 接口定义了一些操作方法

    | 功能 | 抛出异常 | 返回特殊值 |
    | --- | --- | --- |
    | 添加元素 | add(e) | offer(e) |
    | 删除元素 | remove() | poll() |
    | 获取头元素 | element() | peek() |

### 8.10 PriorityQueue

1. PriorityQueue 是一种优先队列，内部使用 ==堆== 存储，==元素必须实现 Comparable 接口==，或者在创建 PriorityQueue 时传入 Comparator

2. PriorityQueue 的元素按照 ==优先级== 排序，==优先级高的元素先出队==，可以通过传入 Comparator 来自定义优先级

    ```java
    Queue<Integer> q = new PriorityQueue<>(new Comparator<Integer>() {
        public int compare(Integer o1, Integer o2) {
            return o2 - o1;
        }
    });
    ```

### 8.11 Deque 接口

1. Deque 是一种双端队列，可以在队列头部和尾部添加、删除元素

2. Deque 接口提供的方法：

    | 功能 | 方法 |
    | --- | --- |
    | 添加到队尾 | addLast(E e) / offerLast(E e) |
    | 添加到队头 | addFirst(E e) / offerFirst(E e) |
    | 取队尾并删除 | removeLast() / pollLast() |
    | 取队头并删除 | removeFirst() / pollFirst() |
    | 取队尾不删除 | getLast() / peekLast() |
    | 取队头不删除 | getFirst() / peekFirst() |

### 8.12 Stack

1. 在 Java 中是用 Deque 接口实现的 Stack，==推荐使用 Deque==

    ```java
    Deque<String> stack = new ArrayDeque<>();
    stack.push("apple");
    stack.push("banana");
    stack.push("cherry");
    while (!stack.isEmpty()) {
        System.out.println(stack.pop());
    }
    ```

### 8.13 Iterator

1. Iterator 对象是集合对象自己在内部创建的，它自己知道如何高效遍历内部的数据集合

### 8.14 Collections

1. Collections 是一个工具类，提供了一系列静态方法，可以方便地操作各种集合

2. 创建空集合，==不可变集合，不支持添加和删除==

    * `List<T> emptyList()` 创建一个空 List
    * `Set<T> emptySet()` 创建一个空 Set
    * `Map<K, V> emptyMap()` 创建一个空 Map

3. 创建单元素集合，==不可变集合，不支持添加和删除==

    * `List<T> singletonList(T o)` 创建一个单元素 List
    * `Set<T> singletonSet(T o)` 创建一个单元素 Set
    * `Map<K, V> singletonMap(K key, V value)` 创建一个单元素 Map

4. 排序，必须传入可变集合，==会直接修改传入的集合==

    * `void sort(List<T> list)` 对 List 进行排序
    * `void shuffle(List<?> list)` 对 List 进行随机排序
    * `void reverse(List<?> list)` 对 List 进行反转
    * `void swap(List<?> list, int i, int j)` 对 List 进行交换
    * `void rotate(List<?> list, int distance)` 对 List 进行旋转

5. 封装可变集合为不可变集合

    * `List<T> unmodifiableList(List<? extends T> list)` 封装 List 为不可变 List
    * `Set<T> unmodifiableSet(Set<? extends T> set)` 封装 Set 为不可变 Set
    * `Map<K, V> unmodifiableMap(Map<? extends K, ? extends V> map)` 封装 Map 为不可变 Map

    > 这种封装实际上是通过创建一个代理对象，拦截掉所有修改方法实现的，==但是原始集合的修改仍然会影响到封装后的集合==
    > 所以应当在创建不可变集合后，丢弃原始集合的引用

6. 线程安全集合

    * `List<T> synchronizedList(List<T> list)` 创建一个线程安全的 List
    * `Set<T> synchronizedSet(Set<T> set)` 创建一个线程安全的 Set
    * `Map<K, V> synchronizedMap(Map<K, V> map)` 创建一个线程安全的 Map

    > 现在一般用更高效的并发集合类，上述的线程安全集合类已经过时

## 9. IO

1. IO 流是一种顺序读写数据的方式，以字节为最小单位

### 9.1 File 对象

1. File 对象应当导入 `java.io.File`

2. File 对象表示 ==文件或目录==，传入的文件和目录可能不存在

3. 创建 File 对象

    ```java
    File f1 = new File("C:\\Windows");
    File f2 = new File("C:\\Windows", "notepad.exe");
    File f3 = new File(f1, "notepad.exe");
    ```

4. File 对象的常用方法

    * `boolean exists()` 判断文件或目录是否存在
    * `boolean isFile()` 判断是否是文件
    * `boolean isDirectory()` 判断是否是目录
    * `String getName()` 获取文件或目录名
    * `String getPath()` 返回构造方法传入的路径
    * `String getAbsolutePath()` 返回绝对路径
    * `String getCannonicalPath()` 返回规范路径
    * `long length()` 返回文件大小，单位是字节
    * `boolean canRead()` 判断是否可读
    * `boolean canWrite()` 判断是否可写
    * `boolean canExecute()` 判断是否可执行

5. 创建和删除文件

    * `boolean createNewFile()` 创建文件，如果文件已经存在，返回 false
    * `boolean delete()` 删除文件或目录，如果文件不存在，返回 false
    * `boolean mkdir()` 创建目录，如果目录已经存在，返回 false
    * `boolean mkdirs()` 创建目录，并创建不存在的父目录

6. 遍历文件和目录

    * `String[] list()` 返回目录下的所有文件和目录名
    * `File[] listFiles()` 返回目录下的所有文件和目录 File 对象

7. listFiles 可以传入 FilenameFilter 和 FileFilter，用于过滤文件和目录

    ```java
    File[] files = f.listFiles(new FilenameFilter() {
        public boolean accept(File dir, String name) {
            return name.endsWith(".exe");
        }
    });
    ```

### 9.2 InputStream

1. InputStream 是所有输入流的父类，是一个抽象类，定义了读取字节的方法
2. InputStream 定义了 `int read()` 方法，返回下一个字节(0~255)，如果已经读到末尾，返回-1
3. FileInputStream 是 InputStream 的子类，用于读取文件

    ```java
    try (InputStream input = new FileInputStream("test.txt")) {
        for (;;) {
            int n = input.read();
            if (n == -1) {
                break;
            }
            System.out.println(n);
        }
    }
    ```

4. IO 流需要关闭，可以使用 `try{} finally{}` 来关闭流, 或者使用 `try-with-resources` 语法自动关闭流
5. 缓冲：

    * `int read(byte[] b)` 读取多个字节到 byte 数组，返回读取的字节数
    * `int read(byte[] b, int off, int len)` 读取多个字节到 byte 数组的指定位置，返回读取的字节数

6. InputStream 的 read 方法是阻塞的，==如果没有数据，会一直等待==

### 9.3 OutputStream

1. OutputStream 是所有输出流的父类，是一个抽象类，定义了写入字节的方法
2. OutputStream 定义了 `void write(int b)` 方法，写入一个字节，==即 int 的后 8 位==
3. `flush()` 方法用于强制将缓冲区的数据输出到目的地

    ```java
    try (OutputStream output = new FileOutputStream("test.txt")) {
        output.write(72); // H
        output.write("Hello, world!".getBytes("UTF-8"));
    }
    ```

### 9.4 Filter 模式

1. Filter 模式是一种常用的设计模式，==通过组合的方式，对现有的类进行功能增强==

### 9.5 操作 Zip

1. Zip 是一种常见的压缩文件格式，Java 标准库提供了 `java.util.zip` 包，可以读写 Zip 文件
2. ZipInputStream 用于读取 Zip 文件，ZipOutputStream 用于写入 Zip 文件

    ```java
    try (ZipInputStream zip = new ZipInputStream(new FileInputStream("test.zip"))) {
        ZipEntry entry;
        while ((entry = zip.getNextEntry()) != null) {
            String name = entry.getName();
            long size = entry.getSize();
            System.out.println(name + ": " + size + " bytes");
        }
    }
    ```

    ```java
    try (ZipOutputStream zip = new ZipOutputStream(new FileOutputStream("test.zip"))) {
        zip.putNextEntry(new ZipEntry("test/"));
        zip.putNextEntry(new ZipEntry("test/1.txt"));
        zip.write("Hello, world!".getBytes("UTF-8"));
    }
    ```

### 9.6 操作 Classpath

1. Classpath 是 Java 程序运行时查找 class 文件的路径，可以是目录，也可以是 jar 文件
2. Classpath 是由环境变量 `CLASSPATH` 指定的，==可以通过 `System.getProperty("java.class.path")` 获取==

    ```java
    String classpath = System.getProperty("java.class.path");
    ```

### 9.7 序列化

1. 序列化是指将对象转换为字节序列，以便存储到文件或网络传输
2. 反序列化是指将字节序列恢复为对象
3. 一个 Java 对象要能序列化，必须实现 `Serializable` 接口

    ```java
    public interface Serializable {
    }
    ```

    这个接口没有任何方法，==只是一个标记接口==，==只有实现了这个接口的类才能被序列化==

4. 对象序列化使用 `ObjectOutputStream`，对象反序列化使用 `ObjectInputStream`

    ```java
    ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    try (ObjectOutputStream output = new ObjectOutputStream(buffer)) {
        output.writeObject(new Person("Xiao Ming", 15));
    }
    ```

    ```java
    try (ObjectInputStream input = new ObjectInputStream(new ByteArrayInputStream(buffer.toByteArray()))) {
        Person p = (Person) input.readObject();
    }
    ```

### 9.8 Files

1. Java 标准库提供了 `java.nio.Files` 类，用于操作文件和目录

    * `Path toPath()` 返回 Path 对象
    * `InputStream newInputStream()` 返回 InputStream 对象
    * `OutputStream newOutputStream()` 返回 OutputStream 对象
    * `byte[] readAllBytes()` 读取所有字节到 byte 数组
    * `List<String> readAllLines()` 读取所有行到 List
    * `void write(byte[])` 写入 byte 数组
    * `void write(Iterable<? extends CharSequence> lines)` 写入多行
    * `void copy(InputStream in, OutputStream out)` 复制流
    * `void move(Path source, Path target)` 移动文件或目录
    * `void delete(Path path)` 删除文件或目录

## 11. 单元测试

### 11.1 JUnit

1. JUnit 是一个单元测试框架，==用于编写和运行单元测试==
2. Assert 类提供了一系列断言方法，用于判断测试结果是否符合预期

    * `assertEquals(expected, actual)` 判断两个对象是否相等
    * `assertTrue(boolean condition)` 判断条件是否为 true
    * `assertFalse(boolean condition)` 判断条件是否为 false
    * `assertNull(Object object)` 判断对象是否为 null
    * `assertNotNull(Object object)` 判断对象是否不为 null
    * `assertSame(expected, actual)` 判断两个对象是否 ==引用同一个对象==
    * `assertNotSame(expected, actual)` 判断两个对象是否 ==引用不同的对象==
    * `assertArrayEquals(expectedArray, resultArray)` 判断两个数组是否相等

3. JUnit 的测试类必须是 public 类，==方法必须是 public 方法==，==方法必须返回 void==，==方法不能有参数==

    ```java
    public class MyTest {
        @Test
        public void test() {
            assertEquals(2, 1 + 1);
        }
    }
    ```

### 11.2 Fixture

1. Fixture 是指测试环境的搭建，==在每个测试方法运行前后都会执行==

    * `@BeforeClass` 在所有测试方法运行前执行一次
    * `@AfterClass` 在所有测试方法运行后执行一次
    * `@Before` 在每个测试方法运行前执行
    * `@After` 在每个测试方法运行后执行

    ```java
    public class MyTest {
        @BeforeClass
        public static void setUp() {
            System.out.println("setUp");
        }

        @AfterClass
        public static void tearDown() {
            System.out.println("tearDown");
        }

        @Before
        public void init() {
            System.out.println("init");
        }

        @After
        public void close() {
            System.out.println("close");
        }

        @Test
        public void test1() {
            System.out.println("test1");
        }

        @Test
        public void test2() {
            System.out.println("test2");
        }
    }
    ```

### 11.3 异常测试

1. 异常测试是指测试方法会抛出指定的异常

    ```java
    @Test
    public void test() {
        assertThrows(IllegalArgumentException.class, () -> {
            Float.parseFloat("one");
        });

### 11.4 条件测试

1. 条件测试是指测试方法只有满足条件才会执行

2. `@Disabled` 注解用于禁用测试方法

    ```java
    @Test
    @Disabled
    public void test() {
        System.out.println("test");
    }
    ```

### 11.5 参数化测试

1. 参数化测试是指测试方法可以传入不同的参数进行测试

2. `@ParameterizedTest` 注解用于指定参数化测试方法

    ```java
    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3})
    public void test(int x) {
        System.out.println(x);
    }
    ```

3. `@MethodSource` 注解用于指定参数化测试方法

    ```java
    @ParameterizedTest
    @MethodSource("stringProvider")
    public void test(String s) {
        System.out.println(s);
    }

## 12. 正则表达式

1. 正则表达式实质上是字符串，其中的特殊字符需要转义，如 `.` 需要写成 `\.`

2. 匹配规则：

    | 字符 | 匹配规则 |
    | --- | --- |
    | `.` | 匹配任意字符 |
    | `\d` | 匹配数字，等价于 `[0-9]` |
    | `\D` | 匹配非数字，等价于 `[^0-9]` |
    | `\s` | 匹配空白字符，包括空格、制表符、换页符等 |
    | `\S` | 匹配非空白字符 |
    | `\w` | 匹配单词字符，包括 `0-9`、`a-z`、`A-Z`、`_` |
    | `\W` | 匹配非单词字符 |
    | `*` | 匹配任意个字符(包括 0 个) |
    | `+` | 匹配至少一个字符 |
    | `?` | 匹配 0 个或 1 个字符 |
    | `{n}` | 匹配 n 个字符 |
    | `{n,}` | 匹配至少 n 个字符 |
    | `{n,m}` | 匹配 n~m 个字符 |
    | `[]` | 匹配指定范围内的字符 |
    | `()` | 分组，可以提取匹配的字符串 |
    | `|` | 或，匹配多个字符串中的一个 |

3. 匹配多行文本：使用 `^` 和 `$` 匹配行的开头和结尾

    ```java
    Pattern p = Pattern.compile("^\\d{3,4}$");
    Matcher m = p.matcher("1234");
    ```

4. 提取分组：使用 `()` 进行分组，可以提取匹配的字符串

    ```java
    Pattern p = Pattern.compile("(\\d{3,4})-(\\d{7,8})");
    Matcher m = p.matcher("010-12345678");
    if (m.matches()) {
        String whole = m.group(0);
        String area = m.group(1);
        String number = m.group(2);
    }
    ```

5. 非贪婪匹配：默认是贪婪匹配，即尽可能多地匹配，可以使用 `?` 进行非贪婪匹配，即尽可能少地匹配

    ```java
    Pattern p = Pattern.compile("(\\d+?)(0*)");
    Matcher m = p.matcher("123000");
    if (m.matches()) {
        String digits = m.group(1);
        String zeros = m.group(2);
    }
    ```

6. 分割字符串：使用 `split()` 方法，传入正则表达式

    ```java
    String[] ss = "A,B,C".split(",");
    ```

7. 替换字符串：使用 `replaceAll()` 方法，传入正则表达式

    ```java
    String s = "A1B2C3".replaceAll("\\d", "#");
    ```

8. 反向引用：使用匹配到的字符串

    ```java
    String s = "the quick brown fox jumps over the lazy dog.";
    String r = s.replaceAll("\\s([a-z]{4})\\s", " <b>$1</b> ");
    System.out.println(r);
    ```

## 14. 多线程

### 14.1 多线程基础

1. 一个 Java 程序就是一个 JVM 进程，JVM 进程用一个主线程来执行 main()方法，==在 main()方法中启动其他线程==

### 14.2 创建线程

1. Java 标准库提供了 `java.lang.Thread` 类，可以直接创建线程

    ```java
    Thread t = new Thread();
    t.start();
    ```

2. 线程过程：start 方法启动线程，run 方法执行线程任务，线程任务执行完毕，线程终止

3. 让线程执行任务有 2 种方法：

    * 从 Thread 派生一个子类，重写 run()方法

        ```java
        public class MyThread extends Thread {
            public void run() {
                System.out.println("start new thread!");
            }
        }

        Thread t = new MyThread();
        t.start();
        ```

    * 实现 Runnable 接口，传入 Thread 构造方法

        ```java
        public class MyRunnable implements Runnable {
            public void run() {
                System.out.println("start new thread!");
            }
        }
        
        Thread t = new Thread(new MyRunnable());
        t.start();
        ```

        ```java
        // 使用lambda表达式
        Thread t = new Thread(() -> {
            System.out.println("start new thread!");
        });
        t.start();
        ```

4. 设置线程优先级，范围是 1~10，默认是 5，但不能保证高优先级的线程一定会先执行

    * `int getPriority()` 获取线程优先级
    * `void setPriority(int priority)` 设置线程优先级

### 14.3 线程状态

1. 线程状态是 Thread.State 枚举定义的，包括：

    * NEW：新创建的线程，尚未执行
    * RUNNABLE：运行中的线程
    * BLOCKED：运行中的线程，因为某些操作被阻塞而等待
    * WAITING：运行中的线程，因为某些操作在等待中
    * TIMED_WAITING：运行中的线程，因为执行 sleep()方法正在计时等待
    * TERMINATED：线程已终止

    ```mermaid
    graph LR
        NEW --> xyz
        subgraph xyz
            RUNNABLE
            BLOCKED
            WAITING
            TIMED_WAITING
        end
        xyz --> TERMINATED
    ```

2. 线程终止原因：

    * run()方法执行完毕，到达 return 语句
    * run()方法抛出异常
    * 线程调用了 stop()方法，强制终止线程

3. join 方法：阻塞当前线程，直到目标线程终止

    * `void join()` 等待线程终止
    * `void join(long millis)` 等待线程终止，最长等待时间为 millis 毫秒

### 14.4 中断线程

1. 通过 `interrupt()` 方法中断线程，==并不会立即终止线程==，只是给线程发送一个中断信号

    * `boolean isInterrupted()` 判断线程是否被中断
    * `static boolean interrupted()` 判断当前线程是否被中断，并清除当前线程的中断标志

    ```java
    Thread t = new Thread(() -> {
        for (;;) {
            if (Thread.currentThread().isInterrupted()) {
                System.out.println("interrupted!");
                break;
            }
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                System.out.println("interrupted when sleep");
                Thread.currentThread().interrupt();
            }
        }
    });

    t.start();
    Thread.sleep(3000);
    t.interrupt();
    ```

2. 如果线程处于等待状态，==调用 interrupt()方法会抛出 InterruptedException 异常==。

3. 也可以使用标志位来中断线程，如使用 `public volatile boolean running = true;` 来控制线程终止

    > Java 内部将变量保存在主内存中，每个线程有各自的工作内存，当线程访问变量时，它会先获取一个副本，并保存在自己的工作内存中。如果线程修改了变量的值，虚拟机会在某个时刻把修改后的值回写到主内存，但是，这个时间是不确定的！

    > 使用 `volatile` 关键字:
    > 1. 每次读取变量时，都从主内存读取
    > 2. 每次修改变量时，立刻回写到主内存

### 14.5 守护线程(Daemon Thread)

1. 守护线程是指为其他线程服务的线程。在 JVM 中，所有非守护线程都执行完毕后，无论有没有守护线程，虚拟机都会自动退出。
2. 编写代码要注意：守护线程不能持有任何需要关闭的资源，如打开文件等。

    ```java
    Thread t = new Thread(() -> {
        try {
            for (;;) {
                System.out.println("守护线程");
                Thread.sleep(1000);
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    });
    t.setDaemon(true);
    t.start();
    ```

### 14.6 线程同步(Synchronized)

1. 临界区(Critical Section)是指一段代码，只能有一个线程进入执行，其他线程必顿等待

2. Java 提供了 synchronized 关键字，可以 ==保证临界区内的代码只能有一个线程执行==

    ```java
    synchronized (this) {
        // 临界区
    }
    ```

    > 同一实例的 synchronized 方法对应的是同一个锁

3. ==线程安全==：多个线程能够正确地处理共享变量

4. 用 synchornized 修饰方法，==相当于对 this 加锁==, 此方法的所有代码都是临界区

    ```java
    public synchronized void add(int n) {
        count += n;
    }
    ```

5. 可重入锁：==一个线程可以多次获得同一个锁==，即可以在持有锁的情况下再次获取锁，并且会增加锁的计数

6. `wait()`: 使当前线程进入等待状态，并释放锁，当唤醒时，重新获得锁，此方法属于 `Object类` 的 `native` 方法，只能在 `synchronized` 块中调用

    ```java
    synchronized (this) {
        this.wait();
    }
    ```

7. `notify()`: 唤醒一个等待的线程，有多个线程等待时，唤醒哪个是不确定的

        ```java
        synchronized (this) {
            this.notify();
        }
        ```

8. `notifyAll()`: 唤醒所有等待的线程，但只有一个线程能获得锁，其他的线程会继续等待，所有用 `notifyAll()` 时，应当在 `while` 循环中判断条件

        ```java
        synchronized (this) {
            this.notifyAll();
        }
        ```

9. `wait()` 和 `notify()` 必须在 `synchronized` 块中调用，==否则会抛出 IllegalMonitorStateException 异常==

10. `ReentrantLock` 是一个可重入锁，==可以替代 synchronized==

    ```java
    import java.util.concurrent.locks.ReentrantLock;

    public class Counter {
        private final ReentrantLock lock = new ReentrantLock();
        private int count;

        public void add(int n) {
            lock.lock();
            try {
                count += n;
            } finally {
                lock.unlock();
            }
        }
    }
    ```

11. `ReentrantLock` 可以替代 `synchronized`，并且可以设置 ==超时时间==，避免死锁

    ```java
    if (lock.tryLock(1, TimeUnit.SECONDS)) {
        try {
            // ...
        } finally {
            lock.unlock();
        }
    }
    ```

12. `Condition` 是 ==用来替代 `wait()` 和 `notify()`== 的，可以 ==精确唤醒== 某个线程

    ```java
    import java.util.concurrent.locks.Condition;
    import java.util.concurrent.locks.ReentrantLock;

    public class Counter {
        private final ReentrantLock lock = new ReentrantLock();
        private final Condition condition = lock.newCondition();
        private int count;

        public void add(int n) {
            lock.lock();
            try {
                count += n;
                condition.signalAll();
            } finally {
                lock.unlock();
            }
        }

        public void get() throws InterruptedException {
            lock.lock();
            try {
                while (count == 0) {
                    condition.await();
                }
                count--;
            } finally {
                lock.unlock();
            }
        }
    }
    ```

13. `ReentrantLock` 和 `synchronized` 的区别：

    * `ReentrantLock` 是显示锁，==需要手动获取和释放==，而 `synchronized` 是隐式锁，==由 JVM 自动管理==
    * `ReentrantLock` 可以 ==设置超时时间==，避免死锁
    * `ReentrantLock` 可以 ==设置公平锁==，避免饥饿

        > 公平锁：多个线程按照申请锁的顺序来获取锁

    * `ReentrantLock` 可以 ==设置多个条件==，精确唤醒线程

14. `ReadWriteLock` 是一种特殊的锁，==允许多个线程同时读，但只允许一个线程写==

    ```java
    import java.util.concurrent.locks.ReadWriteLock;
    import java.util.concurrent.locks.ReentrantReadWriteLock;

    public class Counter {
        private final ReadWriteLock readWriteLock = new ReentrantReadWriteLock();
        private final Lock readLock = readWriteLock.readLock();
        private final Lock writeLock = readWriteLock.writeLock();
        private int count;

        public void add(int n) {
            writeLock.lock();
            try {
                count += n;
            } finally {
                writeLock.unlock();
            }
        }

        public int get() {
            readLock.lock();
            try {
                return count;
            } finally {
                readLock.unlock();
            }
        }
    }
    ```

15. `StampedLock` 是一种特殊的锁，==可以替代 ReadWriteLock==，==提供了乐观读锁==

    > 悲观读锁：读锁会阻塞写锁和其他读锁
    > 乐观读锁：读锁不会阻塞写锁，只是在读取时，如果发现有写锁，则重试

    ```java
    import java.util.concurrent.locks.StampedLock;

    public class Point {
        private double x, y;
        private final StampedLock stampedLock = new StampedLock();

        double distanceFromOrigin() {
            long stamp = stampedLock.tryOptimisticRead();
            double currentX = x, currentY = y;
            if (!stampedLock.validate(stamp)) {
                stamp = stampedLock.readLock();
                try {
                    currentX = x;
                    currentY = y;
                } finally {
                    stampedLock.unlockRead(stamp);
                }
            }
            return Math.sqrt(currentX * currentX + currentY * currentY);
        }
    }
    ```

16. `Semaphore` 是一种计数信号量，==用来控制同时访问特定资源的线程数量==

    ```java
    import java.util.concurrent.Semaphore;

    public class SemaphoreDemo {
        public static void main(String[] args) {
            Semaphore semaphore = new Semaphore(3);     // 允许3个线程同时执行
            for (int i = 0; i < 10; i++) {
                new Thread(() -> {
                    try {
                        semaphore.acquire();
                        System.out.println(Thread.currentThread() + " acquired semaphore");
                        Thread.sleep(1000);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    } finally {
                        semaphore.release();
                        System.out.println(Thread.currentThread() + " released semaphore");
                    }
                }).start();
            }
        }
    }
    ```

17. `Concurrent` 包提供了一组线程安全的集合类:

    |interface|non-thread-safe|thread-safe|
    |---|---|---|
    |List|ArrayList|CopyOnWriteArrayList|
    |Map|HashMap|ConcurrentHashMap|
    |Set|HashSet|CopyOnWriteArraySet|
    |Queue|ArrayDeque/LinkedList|ArrayBlockingQueue/LinkedBlockingQueue|
    |Deque|ArrayDeque/LinkedList|LinkedBlockingDeque|

18. `Atomic` 包提供了一组原子操作的基本类型，==采用的是无锁的方式，CAS 方法==

### 14.7 线程池

1. 线程池是一种多线程处理形式，==线程池中有多个线程，可以并发执行多个任务==

2. 线程池的好处：

    * 重用线程
    * 控制最大并发数
    * 管理线程

3. JAVA 标准库提供了 `java.util.concurrent` 包，==用于实现线程池==

    * `FixedThreadPool`：固定大小的线程池
    * `CachedThreadPool`：可变大小的线程池
    * `SingleThreadPool`：单线程的线程池

    创建这些线程池的方法都被封装到 Executors 这个类中

    ```java
    import java.util.concurrent.*;

    public class Main {
        public static void main(String[] args) {
            // 创建一个固定大小的线程池:
            ExecutorService es = Executors.newFixedThreadPool(4);
            for (int i = 0; i < 6; i++) {
                es.submit(new Task("" + i));
            }
            // 关闭线程池:
            es.shutdown();
        }
    }

    class Task implements Runnable {
        private final String name;

        public Task(String name) {
            this.name = name;
        }

        @Override
        public void run() {
            System.out.println("start task " + name);
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
            }
            System.out.println("end task " + name);
        }
    }
    ```

4. `ScheduledThreadPool`：定时任务线程池

    * `schedule(Runnable command, long delay, TimeUnit unit)`：延迟执行任务
    * `scheduleAtFixedRate(Runnable command, long initialDelay, long period, TimeUnit unit)`：固定延迟执行任务，如果上个任务没有执行完，下个任务会等待
    * `scheduleWithFixedDelay(Runnable command, long initialDelay, long delay, TimeUnit unit)`：固定间隔执行任务，上个任务执行完后，再等待固定时间

### 14.8 Future

1. `Callable`接口类似于`Runnable`，但是它可以返回一个结果：

    ```java
    class Task implements Callable <String> {
        public String call() throws Exception {
            return "result";
        }
    }
    ```

2. `ExecutorService`的`submit()`方法可以提交一个`Callable`任务，并获得一个`Future`对象

    ``` java
    ExecutorService es = Executors.newFixedThreadPool(4);
    Future <String> future = es.submit(new Task());
    String result = future.get(); // 获取结果
    ```

3. `Future`接口表示一个未来可能会返回的结果，在调用get()时，如果异步任务已经完成，我们就直接获得结果。如果异步任务还没有完成，那么get()会阻塞，直到任务完成后才返回结果

4. `Future`接口提供了几个方法：

    * `get()`：获取结果
    * `get(long timeout, TimeUnit unit)`：获取结果，但是最多等待指定的时间
    * `cancel(boolean mayInterruptIfRunning)`：取消任务
    * `isDone()`：判断任务是否已完成

### 14.9 CompletableFuture

1. `CompletableFuture`是Java 8引入的一个新的异步编程机制，可以传入回调对象，当异步任务完成或者发生异常时，自动调用回调对象的回调方法

    > 回调：本质上就是把一个函数或方法作为参数传递给另一个函数或方法，并在适当的时候由后者调用

    ``` java
    public class Main {
        public static void main(String [] args) {
            CompletableFuture <Double> cf = CompletableFuture.supplyAsync(Main:: fetchPrice);
            //成功时调用
            cf.thenAccept(result -> {
                System.out.println("price: " + result);
            });

            //异常时调用
            cf.exceptionally(e -> {
                e.printStackTrace();
                return null;
            });
            // 主线程不要立刻结束，否则 CompletableFuture 默认使用的线程池会立刻关闭:
            try {
                Thread.sleep(2000);
            } catch (InterruptedException e) {
            }

        }

        static Double fetchPrice() {
        }
    }
    ```

2. `CompletableFuture`的串行执行：

    ``` java
    public class Main {
    public static void main(String [] args) throws Exception {
        // 第一个任务:
        CompletableFuture <String> cfQuery = CompletableFuture.supplyAsync(() -> {
            return queryCode("中国石油");
        });
        // cfQuery 成功后继续执行下一个任务:
        CompletableFuture <Double> cfFetch = cfQuery.thenApplyAsync((code) -> {
            return fetchPrice(code);
        });
        // cfFetch 成功后打印结果:
        cfFetch.thenAccept((result) -> {
            System.out.println("price: " + result);
        });
        // 主线程不要立刻结束，否则 CompletableFuture 默认使用的线程池会立刻关闭:
        Thread.sleep(2000);
    }
    }
    ```


    > `(code) -> { return fetchPrice(code); }`是Lambda表达式，等价于`new Function<String, Double>() { public Double apply(String code) { return fetchPrice(code); } }`

3. `anyOf`: 只要有一个CompletableFuture成功返回，就会执行指定的回调

    ``` java
    CompletableFuture <Object> cfQuery = CompletableFuture.anyOf(cfQueryFromSina, cfQueryFrom163);
    ```

4. `allOf`: 所有CompletableFuture都成功返回后，才会执行指定的回调

    ``` java
    CompletableFuture <Void> cfQuery = CompletableFuture.allOf(cfQueryFromSina, cfQueryFrom163);
    ```

### 14.11 使用ThreadLocal

1. `ThreadLocal`是一个本地线程副本变量工具类，==主要用于将私有线程和该线程存放的副本对象做一个映射==，==各个线程之间的变量互不干扰==

    ``` java
    public class Main {
        static ThreadLocal <Integer> threadLocal = new ThreadLocal <>();
    
        public static void main(String [] args) {
            threadLocal.set(1);
            System.out.println(threadLocal.get());
        }
    }
    ```

### 14.12 使用虚拟线程

1. 虚拟线程是一种==比较轻量级的线程==，==不需要操作系统线程==，==可以大量创建==，目的是处理大量的IO密集型任务

2. 使用虚拟线程：

    * 直接创建虚拟线程并启动

        ``` java
        Thread t = Thread.startVirtualThread(() -> {
            // ...
        });
        ```

    * 创建虚拟线程再使用start()方法启动

        ``` java
        Thread t = Thread.ofVirtual().unstarted(() -> {
            // ...
        });
        t.start();
        ```

    * 通过ThreadFactory创建虚拟线程，再使用start()方法启动

        ``` java
        ThreadFactory factory = Thread.ofVirtual().factory();
        Thread t = factory.newThread(() -> {
            // ...
        });
        t.start();
        ```

    * 通过ExecutorService创建虚拟线程

        ``` java
        // 创建调度器:
        ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
        // 创建大量虚拟线程并调度:
        ThreadFactory tf = Thread.ofVirtual().factory();
        for (int i = 0; i < 100000; i++) {
            Thread vt = tf.newThread(() -> { ... });
            executor.submit(vt);
            // 也可以直接传入 Runnable 或 Callable:
            executor.submit(() -> {
                System.out.println("Start virtual thread...");
                Thread.sleep(1000);
                System.out.println("End virtual thread.");
                return true;
            });
        }
        ```

3. 虚拟线程在执行到IO操作或Blocking操作时，会自动切换到其他虚拟线程执行，从而避免当前线程等待，能最大化线程的执行效率；

## 15. Maven基础

1. Maven 是一个项目管理工具，可以自动化构建项目，管理依赖，发布项目等

### 15.1 Maven的介绍

1. Maven 项目结构

    a-maven-project
    ├── pom.xml
    ├── src
    │   ├── main
    │   │   ├── java
    │   │   └── resources
    │   └── test
    │       ├── java
    │       └── resources
    └── target

    * `pom.xml`：Maven 项目的配置文件
    * `src/main/java`：Java 源码目录
    * `src/main/resources`：资源文件目录
    * `src/test/java`：测试代码目录
    * `src/test/resources`：测试资源目录
    * `target`：编译输出目录

2. 一个Maven工程就是由groupId，artifactId和version作为唯一标识的

    * `groupId`：组织标识
    * `artifactId`：项目标识
    * `version`：版本号

3. Maven: pom.xml

    * project: 项目配置
        * xmlns: XML 命名空间
        * xmlns:xsi: XML 命名空间
        * xsi:schemaLocation: XML 命名空间
    * modelVersion: Maven 模型版本
    * groupId: 组织标识
    * artifactId: 项目标识
    * version: 版本号
    * dependencies: 依赖列表
    * dependency: 依赖
        * groupId: 依赖组织
        * artifactId: 依赖项目
        * version: 依赖版本
        * scope: 依赖范围

    ``` xml
    <project xmlns = "http://maven.apache.org/POM/4.0.0"
        xmlns: xsi = "http://www.w3.org/2001/XMLSchema-instance"
        xsi: schemaLocation = "http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd" >
        <modelVersion> 4.0.0 </modelVersion>
        <groupId> com.itranswarp.learnjava </groupId>
        <artifactId> hello-maven </artifactId>
        <version> 1.0.0 </version>
        <dependencies>
            <dependency>
                <groupId> org.apache.commons </groupId>
                <artifactId> commons-lang3 </artifactId>
                <version> 3.12.0 </version>
            </dependency>
        </dependencies>
    </project>
    ```

### 15.2 依赖管理

1. Maven 通过 `pom.xml` 文件来管理依赖

    ``` xml
    <dependencies>
        <dependency>
            <groupId> org.apache.commons </groupId>
            <artifactId> commons-lang3 </artifactId>
            <version> 3.12.0 </version>
        </dependency>
    </dependencies>
    ```

2. 依赖关系：

    * `compile`：编译依赖，默认为 compile
    * `test`：编译测试代码时使用
    * `runtime`：运行时使用，不参与编译
    * `provided`：编译时使用，但不打包到最终的 jar 文件中

3. Maven 会自动下载依赖，并放到本地仓库中

4. 搜索依赖：[Maven Repository](https://mvnrepository.com/)

5. 命令行编译：`mvn clean compile`

### 15.3 构建流程

1. Maven 的构建流程：

    * `clean`：清理
    * `compile`：编译
    * `test`：测试
    * `package`：打包
    * `install`：安装
    * `deploy`：发布

2. 常用命令：

    * `mvn clean`：清理所有生成的class和jar
    * `mvn compile`：编译
    * `mvn clean compile`：清理并编译
    * `mvn test`：运行测试
    * `mvn package`：打包

3. Maven 的生命周期：使用Maven构建项目就是执行lifecycle，执行到指定的phase为止。每个phase会执行自己默认的一个或多个goal。goal是最小任务单元。

4. 插件：Maven 通过插件来实现各种功能，如编译、测试、打包等

    ```xml
    <build>
        <plugins>
            <plugin>
                <groupId> org.apache.maven.plugins </groupId>
                <artifactId> maven-compiler-plugin </artifactId>
                <version> 3.8.1 </version>
                <configuration>
                    <source> 1.8 </source>
                    <target> 1.8 </target>
                </configuration>
            </plugin>
        </plugins>
    </build>
    ```

## 18. JDBC

### 18.1 JDBC基础

1. JDBC 是 Java Database Connectivity 的缩写，==是 Java 语言操作数据库的标准接口==

### 18.2 JDBC查询

1. 添加MySQL驱动，此依赖是runtime依赖，即运行时需要

    ``` xml
    <dependency>
        <groupId> com.mysql </groupId>
        <artifactId> mysql-connector-j </artifactId>
        <version> 9.1.0 </version>
    </dependency>
    ```

2. 查询步骤：

    1. 通过DriverManager获取数据库连接：

        ``` java
        String jdbcUrl = "jdbc: mysql://localhost: 3306/learnjdbc";
        String jdbcUsername = "learn";
        String jdbcPassword = "learnpassword";
        Connection conn = DriverManager.getConnection(jdbcUrl, jdbcUsername, jdbcPassword);
        ```

    2. 通过Connection创建Statement，用于执行SQL：

        ``` java
        Statement stmt = conn.createStatement();
        ```

    3. 通过Statement执行SQL，获取ResultSet：

        ``` java
        ResultSet rs = stmt.executeQuery("SELECT id, grade, name FROM students");
        ```

    4. 通过ResultSet读取数据：

        ``` java
        while (rs.next()) {
            long id = rs.getLong("id");
            int grade = rs.getInt("grade");
            String name = rs.getString("name");
        }
        ```

3. `PreparedStatement`：预编译的 SQL 语句，可以防止 SQL 注入

    ``` java
    String sql = "SELECT id, grade, name FROM students WHERE grade = ?";
    PreparedStatement ps = conn.prepareStatement(sql);
    ps.setObject(1, 3);
    ResultSet rs = ps.executeQuery();
    ```

### 18.3 JDBC更新

1. 插入数据：`executeUpdate()`方法来执行Insert、Update、Delete等操作，返回更新的行数

    ``` java
    PreparedStatement ps = conn.prepareStatement("INSERT INTO students (id, grade, name) VALUES (?, ?, ?)"); 
    ps.setObject(1, 1001);
    ps.setObject(2, 3);
    ps.setObject(3, "Bob");
    int n = ps.executeUpdate();
    ```

2. 插入并获取主键：在创建`PreparedStatement`时，传入`Statement.RETURN_GENERATED_KEYS`，并在执行后通过`getGeneratedKeys()`获取主键

    ``` java
    PreparedStatement ps = conn.prepareStatement("INSERT INTO students (grade, name) VALUES (?, ?)", Statement.RETURN_GENERATED_KEYS);
    ps.setObject(1, 3);
    ps.setObject(2, "Alice");
    int n = ps.executeUpdate();
    try (ResultSet rs = ps.getGeneratedKeys()) {
        if (rs.next()) {
            long id = rs.getLong(1);
        }
    }
    ```

### 18.4 JDBC事务

1. 事务是一组SQL操作，要么全部成功，要么全部失败

2. JDBC事务代码：

    ``` java
    Connection conn = DriverManager.getConnection(jdbcUrl, jdbcUsername, jdbcPassword);
    try {
        conn.setAutoCommit(false); // 关闭自动提交
        // 执行多条 SQL 语句
        conn.commit(); // 提交事务
    } catch (SQLException e) {
        conn.rollback(); // 回滚事务
    } finally {
        conn.setAutoCommit(true); // 恢复现场
        conn.close();
    }
    ```

### 18.5 JDBC批处理

1. 批处理是指一次性执行多条SQL语句，可以大大提高执行效率

2. JDBC批处理代码：

    ``` java
    try (PreparedStatement ps = conn.prepareStatement("INSERT INTO students (name, gender, grade, score) VALUES (?, ?, ?, ?)")) {
        // 对同一个 PreparedStatement 反复设置参数并调用 addBatch():
        for (Student s : students) {
            ps.setString(1, s.name);
            ps.setBoolean(2, s.gender);
            ps.setInt(3, s.grade);
            ps.setInt(4, s.score);
            ps.addBatch(); // 添加到 batch
        }
        // 执行 batch:
        int [] ns = ps.executeBatch();
        for (int n : ns) {
            System.out.println(n + " inserted."); // batch 中每个 SQL 执行的结果数量
        }
    }
    ```

### 18.6 JDBC连接池

1. 连接池是一种数据库连接管理技术，==可以避免频繁打开和关闭数据库连接==，从而提高数据库操作性能

2. 使用连接池 HikariCP :

    ``` java
    HikariConfig config = new HikariConfig();
    config.setJdbcUrl("jdbc: mysql://localhost: 3306/test");
    config.setUsername("root");
    config.setPassword("password");
    config.addDataSourceProperty("connectionTimeout", "1000"); // 连接超时：1 秒
    config.addDataSourceProperty("idleTimeout", "60000"); // 空闲超时：60 秒
    config.addDataSourceProperty("maximumPoolSize", "10"); // 最大连接数：10
    DataSource ds = new HikariDataSource(config);
    ```

    >注意创建DataSource也是一个非常昂贵的操作，所以通常DataSource实例总是作为一个全局变量存储，并贯穿整个应用程序的生命周期。

3. 使用连接池：

    ``` java
    try (Connection conn = ds.getConnection()) {
        // 使用 conn
    }
    ```

## 19. Lambda 表达式

1. Lambda 表达式是一种 ==匿名函数==，可以理解为一段可以传递的代码，格式如下：

    ``` java
    (参数列表) -> {代码块}
    ```

2. Lambda 表达式的参数列表可以省略类型，但是必须要有括号，如果只有一个参数，括号也可以省略

    ``` java
    (int a, int b) -> {return a + b;}
    (a, b) -> {return a + b;}
    ```

3. Lambda 与匿名内部类的区别：

    * 匿名内部类仍然是一个类，只是不需要程序员显示指定类名，编译器会自动为该类取名，编译器仍然会为该类生成一个 class 文件
    * Lambda 表达式通过 invokedynamic 指令动态绑定，会被封装成主类的私有方法，不会生成额外的 class 文件

4. Lambda 表达式中的 this 指向外部类，而匿名内部类中的 this 指向内部类：

    ``` java
    public class Main {
        Runnable r1 = () -> {
            System.out.println(this); // Main
        };

        Runnable r2 = new Runnable() {
            @Override
            public void run() {
                System.out.println(this); // Main$1
            }
        };

        public static void main(String [] args) {
            new Main().r1.run();
            new Main().r2.run();
        }
    }
    ```

    ``` shell
    Main@3a71f4dd
    Main$1@7adf9f5f
    ```

    解释：`Main@3a71f4dd` 表示 Main 类的实例，`Main$1@7adf9f5f` 表示 Main 类的匿名内部类的实例

5. Lambda 表达式中的变量必须是 final 的，或者是等同于 final 的，即只能赋值一次，不能再次赋值

    ``` java
    public class Main {
        public static void main(String [] args) {
            int n = 100;
            Runnable r = () -> {
                System.out.println(n);
            };
            n = 200; // 报错
        }
    }
    ```

   解释：
    * Lambda 表达式获取外部变量时是基于闭包的当 Lambda 表达式被定义时，它并不是立即执行的，而是可以在某个时刻由不同的线程执行。由于局部变量存储在栈内存中，而 Lambda 表达式可能会在该局部变量的作用域之外执行，确保这些变量是不可变的，能够保证 Lambda 的行为是可预期的。
    * 由于局部变量是存在栈上的，而 Lambda 表达式可能在栈帧弹出之后才被调用，为了保证 Lambda 内部引用的是正确的值，Java 会复制这些局部变量，并将其作为匿名类的属性传递

### Lambda + Collection

1. `forEach(Consumer<? super T> action)` 方法接收一个 Consumer 接口，可以使用 Lambda 表达式

    > Consumer 接口是一个函数式接口，表示一个接受单个输入参数并且没有返回值的操作，只有一个 accept 方法

    ``` java
    // 使用 Lambda 表达式遍历 List, 保留长度大于 5 的字符串
    ArrayList <String> list = new ArrayList <>(Arrays.asList("Java", "Python", "C", "JavaScript"));
    list.forEach(s -> {
        if (s.length() > 5) {
            System.out.println(s);
        }
    });
    ```

2. `removeIf(Predicate<? super E> filter)` 方法接收一个 Predicate 接口，可以使用 Lambda 表达式

    > Predicate 接口是一个函数式接口，表示一个接受单个输入参数并返回布尔值的函数，只有一个 test 方法

    ``` java
    // 使用 Lambda 表达式删除 List 中长度小于 5 的字符串
    ArrayList <String> list = new ArrayList <>(Arrays.asList("Java", "Python", "C", "JavaScript"));
    list.removeIf(s -> s.length() < 5);
    ```

3. `replaceAll(UnaryOperator<E> operator)` 方法接收一个 UnaryOperator 接口，可以使用 Lambda 表达式

    > UnaryOperator 接口是一个函数式接口，表示一个接受单个输入参数并返回单个结果的函数，只有一个 apply 方法

    ``` java
    // 使用 Lambda 表达式将 List 中的字符串转换为大写
    ArrayList <String> list = new ArrayList <>(Arrays.asList("Java", "Python", "C", "JavaScript"));
    list.replaceAll(s -> s.toUpperCase());
    ```

4. `sort(Comparator<? super E> c)` 方法接收一个 Comparator 接口，可以使用 Lambda 表达式

    > Comparator 接口是一个函数式接口，表示一个比较器，返回负数表示小于，0 表示等于，正数表示大于，只有一个 compare 方法

    ``` java
    // 使用 Lambda 表达式对 List 进行排序
    ArrayList <String> list = new ArrayList <>(Arrays.asList("Java", "Python", "C", "JavaScript"));
    list.sort((s1, s2) -> s1.length() - s2.length());
    ```

5. `Spliterator<E> spliterator()` 方法返回一个容器的 Spliterator，可以使用 Lambda 表达式

    > Spliterator 是一个函数式接口，表示一个可分割的迭代器，只有一个 tryAdvance 方法
    > 1. Spliterator 既可以像 Iterator 那样逐个迭代，也可以批量迭代。批量迭代可以降低迭代的开销。
    > 2. Spliterator 是可拆分的，一个 Spliterator 可以通过调用 Spliterator \<T\> trySplit()方法来尝试分成两个。一个是 this，另一个是新返回的那个，这两个迭代器代表的元素没有重叠。

    * `tryAdvance(Consumer<? super T> action)` 方法接收一个 Consumer 接口，有元素时执行，返回 true，否则返回 false

        ``` java
        Spliterator <String> spliterator = list.spliterator();
        while(spliterator.tryAdvance((name) -> System.out.println(name)));
        
        //重新遍历，要重新获取 Spliterator
        spliterator = list.spliterator();
        ```

    * `trySplit()` 方法尝试将元素分成两个，返回新的 Spliterator, 两个 Spliterator 代表的元素没有重叠且可以并行处理

        ``` java
        Spliterator <String> spliterator1 = list.spliterator();
        Spliterator <String> spliterator2 = spliterator.trySplit();
        ```

### Lambda + Map

1. 相比 Collection，Map 中加入了更多的方法，我们以 HashMap 为例来逐一探讨：

2. `forEach(BiConsumer<? super K,? super V> action)` 方法接收一个 BiConsumer 接口

    > BiConsumer 接口是一个函数式接口，表示一个接受两个输入参数并且没有返回值的操作，只有一个 accept 方法

    ``` java
    // 使用 Lambda 表达式遍历 Map
    map.forEach((k, v) -> System.out.println(k + ": " + v));
    ```

3. `getOrDefault(Object key, V defaultValue)` 方法接收一个 key，如果 key 存在则返回对应的 value，否则返回 defaultValue

4. `putIfAbsent(K key, V value)` 方法接收一个 key 和 value，如果 key 不存在则添加，否则不添加

5. `remove(Object key, Object value)` 方法接收一个 key 和 value，如果 key 对应的 value 等于 value 则删除，否则不删除

6. `replace()` 方法:

    * `replace(K key, V value)` 方法接收一个 key 和 value，如果 key 存在则替换 value，否则不替换
    * `replace(K key, V oldValue, V newValue)` 方法接收一个 key，oldValue 和 newValue，如果 key 对应的 value 等于 oldValue 则替换为 newValue，否则不替换

7. `replaceAll(BiFunction<? super K,? super V,? extends V> function)` 方法接收一个 BiFunction 接口

    > BiFunction 接口是一个函数式接口，表示一个接受两个输入参数并返回单个结果的函数，只有一个 apply 方法

    ``` java
    // 使用 Lambda 表达式将 Map 中的 value 转换为大写
    map.replaceAll((k, v) -> v.toUpperCase());
    ```

8. `merge(K key, V value, BiFunction<? super V,? super V,? extends V> remappingFunction)` 方法接收一个 key，value 和 BiFunction 接口

    * 如果 key 对应的映射不存在或为 null，则将 value（由第二个参数提供）关联到 key
    * 否则执行 remappingFunction，将 key 对应的 value 和传入的 value 进行合并，合并后的值关联到 key，如果合并后的值为 null，则删除 key

    ``` java
    // 使用 Lambda 表达式将 Map 中的 value 转换为大写
    map.merge("Java", "Java", (v1, v2) -> v1.toUpperCase());
    ```

9. `compute(K key, BiFunction<? super K,? super V,? extends V> remappingFunction)` 方法接收一个 key 和 BiFunction 接口

    * 作用是把 remappingFunction 的计算结果关联到 key 上，如果计算结果为 null，则在 Map 中删除 key 的映射

    ``` java
    // 使用 Lambda 表达式将 Map 中的 value 转换为大写
    map.compute("Java", (k, v) -> v.toUpperCase());
    ```

10. `computeIfAbsent(K key, Function<? super K,? extends V> mappingFunction)` 方法接收一个 key 和 Function 接口

    * 作用是如果 key 不存在或对应的 value 为 null，则计算新的 value 并关联到 key 上，否则不做任何操作，用于对 Map 中不存在的 key 进行初始化

    ``` java
    map.computeIfAbsent("Java", v -> "Java".toUpperCase());
    ```

11. `computeIfPresent(K key, BiFunction<? super K,? super V,? extends V> remappingFunction)` 方法接收一个 key 和 BiFunction 接口

    * 作用是如果 key 存在且对应的 value 不为 null，则计算新的 value 并关联到 key 上，否则不做任何操作

    ``` java
    map.computeIfPresent("Java", (k, v) -> v.toUpperCase());
    ```

### Stream API

1. Stream 是 Java8 引入的一种处理集合的 API，可以对集合进行过滤、映射、排序等操作，类似于 SQL 语句

    ``` mermaid
    graph TB
    subgraph Stream 继承关系
    A[BaseStream] 
    B[IntStream] --> A
    C[LongStream] --> A
    D[DoubleStream] --> A
    E[Stream] --> A
    end
    ```

2. 虽然大部分情况下 stream 是容器调用 Collection.stream()方法得到的，但 stream 和 collections 有以下不同：

    * 无存储。stream 不是一种数据结构，它只是某种数据源的一个视图，数据源可以是一个数组，Java 容器或 I/O channel 等。
    * 为函数式编程而生。对 stream 的任何修改都不会修改背后的数据源，比如对 stream 执行过滤操作并不会删除被过滤的元素，而是会产生一个不包含被过滤元素的新 stream。
    * 惰式执行。stream 上的操作并不会立即执行，只有等到用户真正需要结果的时候才会执行。
    * 可消费性。stream 只能被“消费”一次，一旦遍历过就会失效，就像容器的迭代器那样，想要再次遍历必须重新生成

3. 对 stream 的操作分为为两类，中间操作(intermediate operations)和结束操作(terminal operations)，二者特点是：

    * 中间操作总是会惰式执行，调用中间操作只会生成一个标记了该操作的新 stream，仅此而已。
    * 结束操作会触发实际计算，计算发生时会把所有中间操作积攒的操作以 pipeline 的方式执行，这样可以减少迭代次数。计算完成之后 stream 就会失效。

4. Stream 接口的方法：

    |操作类型|接口方法|
    |--------|--------|
    |中间操作|concat() distinct() filter() flatMap() limit() map() peek()skip() sorted() parallel() sequential() unordered()|
    |结束操作|allMatch() anyMatch() collect() count() findAny() findFirst() forEach() forEachOrdered() max() min() noneMatch() reduce() toArray()|

    区分中间操作和结束操作最简单的方法，就是看方法的返回值，返回值为 stream 的大都是中间操作

5. `forEach(Consumer<? super T> action)` 方法:

    ``` java
    Stream <String> stream = Stream.of("Java", "Python", "C", "JavaScript");
    stream.forEach(s -> System.out.println(s));
    ```

6. `filter(Predicate<? super T> predicate)` 方法，返回一个满足条件的新 stream

    ``` java
    Stream.filter(s -> s.length() > 5);
    ```

7. `distinct()` 方法，返回一个去重的新 stream

    > 去重是通过 equals 方法判断的

    ``` java
    Stream.distinct();
    ```

8. `sorted()` 方法，返回一个排序的新 stream

    * 默认是自然排序：`Stream.sorted()`
    * 自定义排序：`Stream.sorted((s1, s2) -> s1.length() - s2.length())`

9. `map(Function<? super T, ? extends R> mapper)` 方法，对所有的元素进行映射，执行 mapper 函数

    ``` java
    Stream.map(s -> s.toUpperCase());
    ```

10. `flatMap(Function<? super T, ? extends Stream<? extends R>> mapper)` 方法，对所有的元素进行映射，执行 mapper 函数，返回一个新的 stream

    ``` java
    Stream.flatMap(s -> Stream.of(s.split("")));
    ```

### Stream II

1. `reduce()` 方法，对所有元素进行归约操作，有三个重载方法：

    * `Optional<T> reduce(BinaryOperator<T> accumulator)`

    > Optional 是一个容器对象，可能包含也可能不包含一个非 null 的值，如果包含则 isPresent()方法返回 true，get()方法返回值

    ``` java
    Stream.reduce((s1, s2) -> s1 + s2);
    ```

    * `T reduce(T identity, BinaryOperator<T> accumulator)`

    > identity 是初始值，如果 stream 为空则返回 identity

    ``` java
    Stream.reduce("", (s1, s2) -> s1 + s2);
    ```

    * `<U> U reduce(U identity, BiFunction<U, ? super T, U> accumulator, BinaryOperator<U> combiner)`

    > combiner 是并行计算时的合并函数

    ``` java
    Stream <String> stream = Stream.of("I", "love", "you", "too");
    Integer lengthSum = stream.reduce(0,　// 初始值　// (1)
            (sum, str) -> sum+str.length(), // 累加器 // (2)
            (a, b) -> a+b);　// 部分和拼接器，并行执行时才会用到 // (3)
    // int lengthSum = stream.mapToInt(str -> str.length()).sum();
    System.out.println(lengthSum);
    ```

2. `collect()` 方法定义为 `<R> R collect(Supplier<R> supplier, BiConsumer<R,? super T> accumulator, BiConsumer<R,R> combiner)`

    > `Function.identity()` 是 `Function` 接口的一个静态方法，返回一个执行恒等转换的 Lambda 表达式，即返回输入的参数

    > 方法引用：形如 `String::toUpperCase`，表示调用 toUpperCase 方法
    >
    > | 方法引用类别 | 举例 |
    > | ------------ | ---- |
    > | 引用静态方法 | `Integer::parseInt` |
    > | 引用某个对象的方法 | `list::add` |
    > | 引用某个类的方法 | `String::toUpperCase` |
    > | 引用构造方法 | `String::new` |

3. 使用 `collect()` 方法生成 Collection：

    * 不指定实现类：`stream.collect(Collectors.toList())`
    * 指定实现类：`stream.collect(Collectors.toCollection(ArrayList::new))`

4. 使用 `collect()` 方法生成 Map：

    * 使用 `Collectors.toMap()` 方法，第一个参数是 key 的提取函数，第二个参数是 value 的提取函数，第三个参数是 key 冲突时的处理函数

    ``` java
    ArrayList <String> students = new ArrayList <>(Arrays.asList("Tom", "Jerry", "Mike"));
    Map <String, Integer> map = 
        students.stream().collect(Collectors.toMap(Function.identity(), 
                                    String:: length, (s1, s2) -> s1));
    ```

    * 使用 `Collectors.groupingBy()` 方法，按照指定的规则分组

    ``` java
    ArrayList <String> students = new ArrayList <>(Arrays.asList("Tom", "Jerry", "Mike"));
    Map <Integer, List<String> > map = 
        students.stream().collect(Collectors.groupingBy(String:: length));
    ```

    * 使用 `Collectors.partitioningBy()` 方法，按照指定的规则二元分组

    ``` java
    ArrayList <String> students = new ArrayList <>(Arrays.asList("Tom", "Jerry", "Mike"));
    Map <Boolean, List<String> > map = 
        students.stream().collect(Collectors.partitioningBy(s -> s.length() > 3));
    ```

    * 下游收集器：`Collectors.counting()`、`Collectors.summingInt()`、`Collectors.averagingInt()`、`Collectors.maxBy()`、`Collectors.minBy()`、`Collectors.joining()`

    ``` java
    ArrayList <String> students = new ArrayList <>(Arrays.asList("Tom", "Jerry", "Mike"));
    Map <Integer, Long> map = 
        students.stream().collect(Collectors.groupingBy(String:: length, Collectors.counting()));
    ```

5. 使用 `collect()` 方法做字符串拼接：

    * 使用 `Collectors.joining()` 方法，可以指定分隔符、前缀和后缀

    ``` java
    ArrayList <String> students = new ArrayList <>(Arrays.asList("Tom", "Jerry", "Mike"));
    String result = students.stream().collect(Collectors.joining(", ", "[", "]"));
    ```

### Stream Pipelines

1. Stream 操作分类

| 操作类型                               | 无状态 (Stateless)                                                                                                                            | 有状态 (Stateful)                                           |
|:--------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------:|
| **中间操作 (Intermediate operations)** | `unordered()` <br> `filter()` <br> `map()` <br> `mapToInt()` <br> `mapToLong()` <br> `mapToDouble()` <br> `flatMap()` <br> `flatMapToInt()` <br> `flatMapToLong()` <br> `flatMapToDouble()` <br> `peek()` | `distinct()` <br> `sorted()` <br> `limit()` <br> `skip()`     |

| 操作类型                               | 非短路操作 (Non-short-circuiting)                                                                                                              | 短路操作 (Short-circuiting)                                  |
|:--------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------:|
| **结束操作 (Terminal operations)**     | `forEach()` <br> `forEachOrdered()` <br> `toArray()` <br> `reduce()` <br> `collect()` <br> `max()` <br> `min()` <br> `count()`                 | `anyMatch()` <br> `allMatch()` <br> `noneMatch()` <br> `findFirst()` <br> `findAny()` |

Stream 上的所有操作分为两类：中间操作和结束操作，中间操作只是一种标记，只有结束操作才会触发实际计算。中间操作又可以分为无状态的(Stateless)和有状态的(Stateful)，无状态中间操作是指元素的处理不受前面元素的影响，而有状态的中间操作必须等到所有元素处理之后才知道最终结果，比如排序是有状态操作，在读取所有元素之前并不能确定排序结果；结束操作又可以分为短路操作和非短路操作，短路操作是指不用处理全部元素就可以返回结果，比如找到第一个满足条件的元素。之所以要进行如此精细的划分，是因为底层对每一种情况的处理方式不同。

## 20. 设计模式

### 20.1 创建型模式

1. 创建型模式是用来创建对象的设计模式，目的是将对象的创建和使用分离

#### 20.1.1 工厂方法

1. 工厂方法是指定义工厂接口和产品接口，但如何创建实际工厂和实际产品被推迟到子类实现，从而使调用方只和抽象工厂与抽象产品打交道。

    ``` java
    public interface Factory {
        Product createProduct();
    }

    public class ConcreteFactory implements Factory {
        public Product createProduct() {
            return new ConcreteProduct();
        }
    }
    ```

2. 静态 Factory 方法：

    ``` java
    public class Factory {
        public static Product createProduct() {
            return new ConcreteProduct();
        }
    }
    ```

### 20.1.2 抽象工厂

1. 抽象工厂是提供一个创建一系列相关或相互依赖对象的接口，而无需指定它们具体的类。

2. 抽象工厂是工厂方法的升级版：

    ``` java
    public interface AbstractFactory {
        public static AbstractFactory createFactory(String name) {
            if (name.equalsIgnoreCase("fast")) {
                return new FastFactory();
            } else if (name.equalsIgnoreCase("good")) {
                return new GoodFactory();
            } else {
                throw new IllegalArgumentException("Invalid factory name");
            }
        }
    }
    ```

### 20.1.3 生成器

1. Builder模式是为了创建一个复杂的对象，需要多个步骤完成创建，或者需要多个零件组装的场景，且创建过程中可以灵活调用不同的步骤或组件。

### 20.1.4 原型

1. 原型模式是指创建一个原型对象，通过复制这个原型对象来创建更多同类型的对象。

### 20.1.5 单例

1. 单例模式：目的是为了保证在一个进程中，某个类有且仅有一个实例。

2. 实现方法1：private static final 类型的静态变量

    ``` java
    public class Singleton {
        private static final Singleton INSTANCE = new Singleton();
        private Singleton() {}
        public static Singleton getInstance() {
            return INSTANCE;
        }
    }
    ```

3. 实现方法2：enum 类型，java会保证enum类型的构造方法只被调用一次

    ``` java
    public enum Singleton {
        INSTANCE;
    }
    ```
# Makefile

1. 基本语法：

    ```Makefile
    target: dependencies
        command
    ```

    1. 目标即要生成的文件。如果目标文件的更新时间晚于依赖文件更新时间，则说明依赖文件没有改动，目标文件不需要重新编译。否则会进行重新编译并更新目标文件。

    2. 默认情况下 Makefile 的第一个目标为终极目标。
    3. 注意每条命令之前必须有一个 tab 保持缩进

2. 变量：

    1. `$` 表示引用变量的值
    2. `$@` 表示目标文件
    3. `$^` 表示所有依赖文件
    4. `$<` 表示第一个依赖文件

3. 运算符：

    1. `:=` 表示变量的值现在就展开
    2. `=` 表示变量的值在 makefile 文件中被引用时才会展开
    3. `?=` 表示如果变量到目前为止还没有被定义过，则定义它
    4. `+=` 表示追加变量的值, 就是 c++中的 `a+=b` 等价于 `a=a+b`

4. 函数：

   1. `$(shell command)` 执行 shell 命令

        ```Makefile
        # 获取当前时间
        TIME = $(shell date)
        ```

   2. `$(wildcard pattern)` 展开通配符

        ```Makefile
        # 获取当前目录中的所有.c文件，并将它们的文件名列表赋值给SRC
        SRC = $(wildcard *.c)
           
        # 将所有.c文件的后缀替换为.o，生成目标文件的列表并赋值给OBJ
        OBJ = $(patsubst %.c, %.o, $(SRC))
           
        # 定义默认目标，这通常是Makefile运行时的默认构建任务
        ALL: hello.out
           
        # 规则用于生成可执行文件hello.out，它依赖于所有的目标文件（即$(OBJ)）
        hello.out: $(OBJ)
            gcc $(OBJ) -o hello.out  # 使用GCC将目标文件链接为可执行文件
           
        # 通用规则，用于将.c文件编译为.o文件
        %.o: %.c
            gcc -c $< -o $@  # 使用GCC将源文件编译为目标文件，$<代表输入文件，$@代表输出文件
        ```

5. 伪目标：

    1. 伪目标是指不对应任何实际文件的目标，只是一个标签，用于执行一系列命令。

        ```Makefile
        .PHONY: clean  # 声明clean为伪目标
        
        clean:  # clean伪目标的规则
            rm -f *.o hello.out  # 删除所有.o文件和可执行文件
        ```

        ```shell
        make clean  # 执行clean伪目标的规则
        ```

6. 嵌套 Makefile：

    1. 可以在 Makefile 中嵌套调用其他 Makefile。

        ```Makefile
        subsystem:
            cd subdir && $(MAKE)
        # 其等价于：
        subsystem:
            $(MAKE) -C subdir
        ```

7. 指定头文件，库文件路径：

    ```Makefile
    # 指定头文件路径
    CFLAGS = -I/usr/local/include
    
    # 指定库文件路径
    LDFLAGS = -L/usr/local/lib
    ```
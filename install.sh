#!/bin/bash

# 变量设定为0表示初始状态
SUCCESS=0

# 使用while循环，直到命令成功退出
while [ $SUCCESS -ne 1 ]; do
    # 尝试安装poppler
    brew install poppler
    
    # 检查最后一个命令的退出状态码
    if [ $? -eq 0 ]; then
        echo "Poppler 安装成功"
        SUCCESS=1
    else
        echo "Poppler 安装失败，正在重试..."
        sleep 1 # 等待5秒后重试，可以根据需要调整时间
    fi
done


# 编译命令（适配Python 3）
# -fPIC：生成位置无关代码（共享库必需）
# -shared：生成共享库
# -I$(python3 -m sysconfig | grep include | cut -d'=' -f2)：自动获取Python头文件路径
# -o simple.so：输出的扩展库文件名（必须是 模块名.so）
INCLUDE_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
gcc -fPIC -shared -o simple.so simple.c -I$INCLUDE_PATH

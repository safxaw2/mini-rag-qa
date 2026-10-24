# Linux 常用命令速查

## 一、文件与目录操作

```bash
ls              # 列出目录内容
ls -la          # 列出所有文件（含隐藏），详细信息
cd /path        # 切换目录
cd ..           # 返回上级目录
cd ~            # 回到家目录
pwd             # 显示当前路径

mkdir dir       # 创建目录
mkdir -p a/b/c  # 递归创建多级目录
rmdir dir       # 删除空目录
rm file         # 删除文件
rm -rf dir      # 递归强制删除（慎用！）

cp file1 file2  # 复制文件
cp -r dir1 dir2 # 递归复制目录
mv file1 file2  # 移动/重命名

touch file      # 创建空文件 / 更新时间戳
cat file        # 查看文件内容
less file       # 分页查看（可上下翻）
head -n 10 file # 查看前10行
tail -n 10 file # 查看后10行
tail -f log     # 实时跟踪日志
```



## 二、查找与搜索

```bash
find /path -name "*.log"        # 按名称查找
find . -type f -mtime -1        # 查找1天内修改的文件
grep "keyword" file             # 在文件中搜索
grep -r "keyword" dir           # 递归搜索目录
grep -i "keyword" file          # 忽略大小写
grep -n "keyword" file          # 显示行号

which command                   # 查找命令路径
whereis command                 # 查找命令相关文件
```



## 三、权限管理

```bash
chmod 755 file       # 设置权限（rwxr-xr-x）
chmod +x script.sh   # 添加执行权限
chown user:group file # 修改所有者
chown -R user dir    # 递归修改所有者

# 权限数字含义：r=4 w=2 x=1
```



## 四、进程管理

```bash
ps aux              # 查看所有进程
ps -ef | grep java  # 查找特定进程
top                 # 实时查看系统资源
htop                # 更友好的 top（需安装）
kill PID            # 终止进程
kill -9 PID         # 强制终止
pkill -f name       # 按名称终止进程
jobs                # 查看后台任务
bg / fg             # 后台/前台切换任务
```



## 五、系统信息

```bash
uname -a            # 系统内核信息
cat /etc/os-release # 系统版本
df -h               # 磁盘使用情况
du -sh dir          # 目录大小
free -h             # 内存使用
uptime              # 运行时间与负载
whoami              # 当前用户
date                # 当前时间
```



## 六、压缩与解压

```bash
tar -czvf a.tar.gz dir   # 打包压缩
tar -xzvf a.tar.gz       # 解压
zip -r a.zip dir         # 压缩为 zip
unzip a.zip              # 解压 zip
```



## 七、网络相关

```bash
ping host              # 测试连通性
curl url               # 发送 HTTP 请求
wget url               # 下载文件
ip addr                # 查看 IP 地址
netstat -tunlp         # 查看端口占用
ss -tunlp              # 同上（更现代）
ssh user@host          # 远程登录
scp file user@host:/path # 远程复制
```



## 八、软件包管理

```bash
# Debian/Ubuntu
apt update && apt upgrade   # 更新
apt install pkg             # 安装
apt remove pkg              # 卸载

# CentOS/RHEL
yum install pkg
dnf install pkg
```



## 九、实用技巧

```bash
command &           # 后台运行
nohup cmd &         # 后台运行且退出终端不中断
cmd1 && cmd2        # 前一个成功才执行后一个
cmd > file 2>&1     # 标准输出和错误都重定向
history             # 查看历史命令
Ctrl+R              # 搜索历史命令
Ctrl+C              # 终止当前命令
Ctrl+Z              # 挂起当前命令
man command         # 查看命令手册
```



## 十、常用组合示例

```bash
# 统计目录下文件数量
ls -l | wc -l

# 查找并删除 7 天前的日志
find /var/log -name "*.log" -mtime +7 -delete

# 查看端口 8080 被哪个进程占用
lsof -i:8080  或  netstat -tunlp | grep 8080

# 实时监控日志中的错误
tail -f app.log | grep -i error
```

## Shell 脚本核心语法

Shell 脚本的本质是把一系列命令写进文件，交给解释器（通常是 Bash）执行。

**1. 脚本结构与执行**
脚本第一行通常写 `#!/bin/bash`，这叫 Shebang，用来指定解释器。执行前需要赋予执行权限：

```bash
chmod +x script.sh
./script.sh          # 作为可执行程序运行
bash script.sh       # 或者直接用解释器运行（无需+x）[citation:2]
```



**2. 变量与引号（关键区别）**
Shell 中变量赋值 `=` 两边不能有空格。引用变量时加 `$`。

```bash
name="World"
echo "Hello, $name"   # 双引号：允许变量替换 -> Hello, World
echo 'Hello, $name'   # 单引号：原样输出 -> Hello, $name[citation:1]
```

如果需要把变量传给子进程（比如脚本里调用的其他命令），需要 `export` 它。

**3. 特殊参数（脚本自带变量）**
编写脚本时经常会用到外部传进来的参数：

- `$0`：脚本名称
- `$1` ~ `$9`：第 1 到第 9 个参数
- `$#`：参数的总个数
- `$@`：所有参数列表

**4. 流程控制**

```
# if 判断（注意空格）
if [ -f "$file" ]; then
    echo "文件存在"
elif [ -d "$file" ]; then
    echo "这是目录"
else
    echo "不存在"
fi

# for 循环（遍历文件比解析 ls 更安全）
for f in *.txt; do
    [ -e "$f" ] || continue  # 防止无匹配文件时报错
    echo "处理 $f"
done

# while 循环
i=1
while [ $i -le 5 ]; do
    echo $i
    ((i++))
done
```
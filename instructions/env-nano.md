# Jetson Nano 学习文档

# 初入 Jetson Nano

## 镜像

参考教程：[Jetson Nano入门教程：硬件篇+视频学习教程_jetson nano开发教程-CSDN博客](https://blog.csdn.net/ting_qifengl/article/details/111246874)

首先需要准备一张sd卡（和树莓派很像）；下载官方的镜像，[下载地址]([Jetson Download Center | NVIDIA Developer](https://developer.nvidia.com/embedded/downloads#?tx=$product,jetson_nano))

在Filter中勾选Jetson Nano，选择**Jetson Nano Developer Kit SD Card Image**

按照教程需要准备两个应用

1. SD Card Formatter: 用于先将sd卡进行格式化，[下载地址](https://www.sdcard.org/downloads/formatter/sd-memory-card-formatter-for-windows-download/)
2. banlenaEtcher：烧录镜像

**Notes**：如果无法识别sd卡，可以在windows笔记本上使用**win+X**打开**磁盘管理**，利用磁盘管理对sd卡进行一个初始化。

## 激活

Jetson Nano中安装的是Ubuntu系统；按照正常的Ubuntu激活即可；

可能需要网线、电源线、显示器、HDMI线

**Notes**：请注意电源方式，这关系到引脚是否需要连接

# Jetpack 版本查询

在此之前需要查询自己所对应的JetPack版本，查询指令

```bash
dpkg-query --show nvidia-l4t-core
```

这个命令会返回一个与JetPack版本相关联的版本号，根据版本号与NVIDIA官方文档进行比照。

我的版本号是：`nvidia-l4t-core 32.7.4-20230608212426`，对应的是**JetPack 4.6**

或者采取更直接的方法，输入指令：

```bash
sudo apt show nvidia-jetpack
```

# conda的安装

参考教程：[jetson nano配置conda、cuda、torch、torchvision环境_jetson nano 安装cuda-CSDN博客](https://blog.csdn.net/zhang1009_/article/details/130308534?ops_request_misc=%7B%22request%5Fid%22%3A%22170608422616800215093943%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fall.%22%7D&request_id=170608422616800215093943&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-8-130308534-null-null.142^v99^pc_search_result_base5&utm_term=jetsonnano安装miniconda&spm=1018.2226.3001.4187)

首先需要配置cuda，打开`./bashrc`文件

在最后添加代码

```
export CUDA_HOME=/usr/local/cuda-10.2
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.2/bin:$PATH
```

最后需要激活

```bash
source ~/.bashrc
```

验证是否成功

```bash
nvcc -V
```

下载archiconda，对于Jetson Nano是aarch64，在安装最新版本的miniconda的时候会报错

```bash
wget https://github.com/Archiconda/build-tools/releases/download/0.2.3/Archiconda3-0.2.3-Linux-aarch64.sh
```

使用指令bash

```bash
bash Archiconda3-0.2.3-Linux-aarch64.sh
```

安装完成以后配置环境变量，在文件`/.bashrc`中添加：

```
export PATH=~/archiconda3/bin:$PATH
```

之后再进行激活

```bash
source ~/.bashrc
```

添加conda源

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
```

# torch安装

对于Nano来说，安装torch需要使用[英伟达官方](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)提供的wheel；也可以使用[torch.kmtea.eu/whl/stable.html](https://torch.kmtea.eu/whl/stable.html)所提供的镜像。

选择合适的版本进行安装。

在安装之前一定要注意，为了避免后续一个错误的发生，请运行一个指令：

```bash
sudo apt-get install libopenblas-base libopenmpi-dev
```

这是为了防止下方的错误出现：

```
OSError: libmpi_cxx.so.20: cannot open shared object file: No such file or directory
```

## torch 1.4.0

下载wheel 文件以后，运行指令：

```bash
pip install torch-1.4.0-cp36-cp36m-linux_aarch64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

进行`import torch`测试的时候会发生以下的错误，是关于numpy的

```
>>> import torch
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/hao/archiconda3/envs/fl2/lib/python3.6/site-packages/torch/__init__.py", line 81, in <module>
    from torch._C import *
ImportError: numpy.core.multiarray failed to import
```

这是由于在安装上述torch wheel文件的时候numpy并未安装，因此需要额外安装一下numpy

鉴于`python=3.6`以及`torch=1.4.0`，我选择的numpy版本是`numpy 1.17`，安装指令

```bash
pip install numpy~=1.17.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

这样`import torch`就没有什么问题了。

另外测试一下jetson nano是否可以使用gpu：

```python
import torch
print(torch.cuda.is_available())
```

## torch 1.10.0

同样是下载wheel文件，运行指令

```bash
pip install torch-1.10.0-cp38-cp38-linux_aarch64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

进行`import torch`的操作以后会发生相似错误，关于numpy：

```python
>>> import torch
/home/hao/archiconda3/envs/dpl/lib/python3.8/site-packages/torch/package/_directory_reader.py:17: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at  /opt/pytorch/torch/torch/csrc/utils/tensor_numpy.cpp:68.)
  _dtype_to_storage = {data_type(0).dtype: data_type for data_type in _storages}
```

原因也是缺少numpy，安装指令

```bash
pip install numpy~=1.17.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

这样测试并没有问题`import torch`

测试是否可以使用cuda：

```python
import torch
print(torch.cuda.is_available())
```

Note：**非常可惜的事情是并不支持cuda**

# pysyft 0.2.4 安装

首先单独安装pysyft：

```bash
pip install syft==0.2.4  --no-dependencies  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

然后安装其dependency

```bash
pip install lz4~=3.0.2 msgpack~=1.0.0 phe~=1.4.0 scipy~=1.5.1 syft-proto~=0.2.5.a1 tblib~=1.6.0 websocket-client~=0.57.0 websockets~=8.1.0 zstd~=1.4.4.0 Flask~=1.1.1 tornado==4.5.3 flask-socketio~=4.2.1 lz4~=3.0.2 requests~=2.22.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中一个dependency `Pillow`存在问题，无法进行安装；但是测试`import syft`没有问题；并且使用代码测试也没有问题，暂时就先摆烂了。

# 迁移

参考教程：[Jetson 系列——jetson nano制作SD卡备份镜像以及还原_jetson nano sd卡-CSDN博客](https://blog.csdn.net/weixin_42264234/article/details/119977617)

准备工作：

1. Linux 一台
2. sd卡两张
   1. 32G：已经配置好的sd
   2. 64G：空白的sd卡
3. 读卡器

## 备份

连接sd卡到主机，对于Ubuntu而言，可以直接挂载；查看是否挂载成功可以使用`df -h`或者`lsblk`查看挂载的**sd卡名称。**

下一步是很关键的一步就是，对sd卡进行备份，指令如下：

```bash
sudo dd if=/dev/sdb conv=sync,noerror bs=4M | gzip -c > backup_image.img.gz
```

`if`为指定输入，`bs`控制读写的速度。

一般来说是无法查看到进程的，但是可以通过打开一个新的terminal，运行下方指令可以查看

```bash
sudo pkill -USR1 -n -x dd
```

## 烧录

将装好的sd卡取下来，插入空白的sd卡，执行恢复的命令；同时需要注意的是需要选择正确的设备`/dev/sdb`

```
sudo su
gunzip -c backup_image.img.gz | dd of=/dev/sdb bs=4M
```

## 扩展

将32G的镜像烧录到64G的SD卡中，需要执行扩展的操作。

那么就需要在nano中安装`gparted`这个应用来对系统的分区进行重建。

安装指令：

```bash
sudo apt-get update
sudo apt-get install gparted
```

运行`gparted`:

```bash
sudo gparted
```

这是一个图形化的界面，只要将分区滑动到想要的部分就可以了。

使用gparted扩展很有可能出现不成功的问题，会有分区unallocated；解决方案如下：

将烧录好的sd卡，插入Ubuntu中，首先执行命令，卸载分区

```bash
sudo umount /dev/sdb1
```

然后运行命令

```bash
sudo e2fsck -f /dev/sdb1
```

最后使用命令进行分区重组

```bash
sudo resize2fs /dev/sdb1
```


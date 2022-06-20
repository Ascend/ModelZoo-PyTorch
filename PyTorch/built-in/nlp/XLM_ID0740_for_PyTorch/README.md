# XLM模型使用说明

## Requirements
* NPU配套的run包安装
* Python 3.7.5
* numpy 
* PyTorch(NPU版本)
* apex(NPU版本)
* (可选)参考《Pytorch 网络模型移植&训练指南》6.4.2章节，配置cpu为性能模式，以达到模型最佳性能；不开启不影响功能。

## Dataset Prepare
1. 下载Wikipedia单语数据，本模型训练时使用en，zh两种单语数据集，执行下载单语数据命令：
```
./get-data-wiki.sh en  # 下载英文单语数据
./get-data-wiki.sh zh  # 下载中文单语数据
```

## 安装语言处理工具
方法一：
1. 进入tools路径
```
cd tools/
```
2. 安装摩西标记器 
```
git clone https://github.com/moses-smt/mosesdecoder
```
3. 安装 中文斯坦福分词器
```
wget https://nlp.stanford.edu/software/stanford-segmenter-2018-10-16.zip
unzip stanford-segmenter-2018-10-16.zip
```
4. 安装fastBPE
```
git clone https://github.com/glample/fastBPE
cd fastBPE
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
```
方法二：
直接使用install-tools.sh脚本进行安装
```
./install-tools.sh
```

##处理数据集en,zh
执行脚本tokenize_en_zh.sh
```
bash tokenize_en_zh.sh
```
就会在data/processed/XLM_en_zh/50k路径下生成处理好的en,zh数据集


## Train MODEL

### 单卡（由于XLM模型在单卡训练时，loss不收敛，故不采用单卡训练）
       bash ./test/train_full_1p.sh  --data_path=数据集路径                 # 精度训练
       bash ./test/train_performance_1p.sh  --data_path=数据集路径     # 性能训练
        [ 数据集路径写到XLM_en_zh这一级 ]

### 8卡
       bash ./test/train_full_8p.sh  --data_path=数据集路径           # 精度训练
       bash ./test/train_performance_8p.sh  --data_path=数据集路径     # 性能训练
        [ 数据集路径写到XLM_en_zh这一级 ]

## 单卡训练时，如何指定使用第几张卡进行训练
1. 修改 xlm/slurm.py脚本
 将168行，torch.npu.set_device(params.local_rank) 注释掉
 同时在其后添加如下一行
 torch.npu.set_device("npu:id") # id可以设置为自己想指定的卡


## 由于XLM模型在docker中训练需要占用比较大的内存，建议在开启docker时，将shm-size设置大些
建议设置shm-size为100G，已经在docker_start.sh脚本中添加，如果宿主机内存不足100G，可以适当减小，
修改docker_start.sh脚本中的shm-size参数配置， 可设置为10G左右。

## 注意: XLM模型在八卡训练的编译阶段，使用的内存最大能达到315G左右，
   建议测试服务器要保证有大于320G的可用内存空间，才能拉起模型训练。

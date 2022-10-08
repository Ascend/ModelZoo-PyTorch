# ESPnet Dynamic for PyTorch
- [ESPnet Dynamic for PyTorch](#espnet-dynamic-for-pytorch)
- [概述](#概述)
- [准备训练环境](#准备训练环境)
  - [准备环境](#准备环境)
  - [准备数据集](#准备数据集)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)
  - [变更](#变更)
  - [已知问题](#已知问题)

# 概述
ESPNet是一套基于E2E的开源工具包，可进行语音识别等任务。从另一个角度来说，ESPNet和HTK、Kaldi是一个性质的东西，都是开源的NLP工具；引用论文作者的话：ESPnet是基于一个基于Attention的编码器-解码器网络，另包含部分CTC组件

- 参考实现：

  ```
  url=https://github.com/espnet/espnet/tree/v.0.10.5
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/audio/ESPnet_Dynamic_for_PyTorch
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动 | [22.0.2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.1.RC1](https://www.hiascend.com/software/cann/commercial?version=5.1.RC1) |
  | PyTorch    | [1.5](https://gitee.com/ascend/pytorch/tree/master/)

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 1.安装依赖（根据模型需求，按需添加所需依赖）。

  ```
  pip3 install -r requirements.txt
  ```
- 2.安装ESPnet

  1）安装好相应的cann包、pytorch和apex包，并设置好pytorch运行的环境变量；

  2）基于espnet官方的安装说明进行安装： [Installation — ESPnet 202205 documentation](https://espnet.github.io/espnet/installation.html) 

  安装过程比较复杂，需注意以下几点：

  - 安装依赖的软件包时，当前模型可以只安装cmake/sox/sndfile ；

  - 安装kaldi时，当前模型调测选择了OpenBLAS作为BLAS库，在compile kaldi & install阶段，使用如下命令安装：

    ```
    $ cd <kaldi-root>/src
    $ ./configure --openblas-root=../tools/OpenBLAS/install --use-cuda=no
    $ make -j clean depend; make -j <NUM-CPU>
    ```

  - 安装espnet时，步骤1中的git clone ESPnet代码替换为下载本modelzoo中ESPnet的代码；步骤3中设置python环境，若当前已有可用的python环境，可以选择D选项执行；步骤4中进入tools目录后，直接使用make命令进行安装，不需要指定PyTorch版本;

  - custom tool installation这一步可以选择不安装。最后通过check installation步骤检查安装结果；

  3）运行模型前，还需安装：

  - boost: ubuntu上可使用 apt install libboost-all-dev命令安装，其它系统请选择合适命令安装
  - kenlm：进入<espnet-root>/tools目录，执行make kenlm.done
  
  4）(option)更新软连接：

  - cd <espnet-root>/egs/aishell/asr1
    - rm -f utils steps
    - ln -s ../../../tools/kaldi/egs/wsj/s5/utils utils
    - ln -s ../../../tools/kaldi/egs/wsj/s5/steps steps
  - cd <espnet-root>/egs/aishell/asr1/conf
    - rm -f train.yaml decode.yaml
    - ln -s tuning/train_pytorch_conformer_kernel15.yaml train.yaml
    - ln -s tuning/decode_pytorch_transformer.yaml decode.yaml


## 准备数据集

1. 获取数据集。

   2017 年，北京希尔贝壳科技有限责任公司 (Beijing shellshell Technology Co., Ltd.) 发布了当时最大规模的用于语音识别研究和构建语音识别系统的中文普通话数据集 aishell-1[3]，包含由 400 位说话人录制的超过 170 小时的语音。 aishell-1 是 500 小时多通道普通话数据集 aishell-asr0009 的子集，采样率 16kHz，量化精度 16 比特。数据集目录结构参考如下所示。

   ```
    /export/a05/xna/data/
                    ├── data_aishell.tgz
                    |
                    └── resource_aishell.tgz

2. 数据预处理（按需处理所需要的数据集）。

## 训练

### 1.原模型训练方法

进入egs/aishell/asr1目录，执行以下命令进行训练：

```
bash run.sh
```

常用参数：

--stage <-1 ~ 5>、 --stop_stage <-1 ~ 5>：控制模型训练的起始、终止阶段。模型包含-1 ~ 5个训练阶段，其中-1 ~ 2为数据下载、准备、特征生成等阶段，3为LM训练，4为ASR训练，5为decoding。首次运行时请从-1开始，-1 ~ 2阶段执行过一次之后，后续可以从stage 3开始训练。LM和ASR是在NPU上运行的，其余都在CPU上运行。

--ngpu <1 or 8>： 控制模型进行1P or 8P训练。

### 2.执行test目录下脚本进行训练

单卡训练

```
bash ./test/train_full_1p.sh --stage=起始stage --data_path=数据集路径
```

多卡训练

```
bash ./test/train_full_8p.sh --stage=起始stage --data_path=数据集路径
```

注：

--stage为可选参数，默认为-1，即从数据下载开始。若之前数据下载、准备、特征生成等阶段已完成，可从stage 3开始训练。

--data_path为必选参数。

    
   

# 训练结果展示

**表 2**  训练结果展示表

| 模型            | GPU错误率                 | NPU错误率                 | GPU 1P(iters/sec) | NPU 1P(iters/sec) | GPU 8P(iters/sec) | NPU 8P(iters/sec) |
|---------------|------------------------|------------------------|-------------------|-------------------|-------------------|-------------------|
| ESPnet动态shape | test数据集：6.1 dev数据集：5.6 | test数据集：6.1 dev数据集：5.5 | LM：24.154         | LM：7.3368         | ASR：1.1966        | ASR：0.77794       |


# 版本说明

## 变更

2022.08.17：首次发布。

## 已知问题

**_当前发行版本中存在的问题描述。_**

无。








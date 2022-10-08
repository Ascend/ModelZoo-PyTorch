## ESPnet模型训练方法

### 0.简介

模型代码基于 [GitHub - espnet/espnet at v.0.10.5](https://github.com/espnet/espnet/tree/v.0.10.5) ，对其中的子集egs/aishell/asr1进行了NPU的适配优化，相应的配置文件为 [espnet/train_pytorch_conformer_kernel15.yaml at v.0.10.5 · espnet/espnet · GitHub](https://github.com/espnet/espnet/blob/v.0.10.5/egs/aishell/asr1/conf/tuning/train_pytorch_conformer_kernel15.yaml) 。



### 1.安装依赖

```
pip3 install -r requirements.txt
```



### 2.安装ESPnet

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



### 3.训练

#### 原模型训练方法

进入egs/aishell/asr1目录，执行以下命令进行训练：

```
bash run.sh
```

常用参数：

--stage <-1 ~ 5>、 --stop_stage <-1 ~ 5>：控制模型训练的起始、终止阶段。模型包含-1 ~ 5个训练阶段，其中-1 ~ 2为数据下载、准备、特征生成等阶段，3为LM训练，4为ASR训练，5为decoding。首次运行时请从-1开始，-1 ~ 2阶段执行过一次之后，后续可以从stage 3开始训练。LM和ASR是在NPU上运行的，其余都在CPU上运行。

--ngpu <1 or 8>： 控制模型进行1P or 8P训练。

#### 执行test目录下脚本进行训练

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



### Q&A

1. arm环境上运行时遇到加载so报错，ImportError: /.../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block，可以通过加载环境变量LD_PRELOAD解决：

```
export LD_PRELOAD=/.../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
```
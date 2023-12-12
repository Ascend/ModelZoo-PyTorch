# RoBERTa模型-推理指导

- [Roberta模型-推理指导](#Roberta模型-推理指导)
- [概述](#概述)
  - [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
  - [安装CANN包](#安装cann包)
  - [安装Ascend-cann-aie](#安装ascend-cann-aie)
  - [安装Ascend-cann-torch-aie](#安装ascend-cann-torch-aie)
- [快速上手](#快速上手)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理性能\&精度](#模型推理性能精度)

# 概述

`RoBERTa` 属于BERT的强化版本，也是BERT模型更为精细的调优版本。RoBERTa 模型是BERT 的改进版(从其名字来看，A Robustly Optimized BERT，即简单粗暴称为强力优化的BERT方法)。主要在在模型规模、算力和数据上，进行了一些改进。

- 参考实现：

  ```
  url=https://github.com/pytorch/fairseq.git
  mode_name=RoBERTa
  hash=c1624b27
  ```



## 输入输出数据

- 输入数据

  说明：原仓默认的seq_length为70

  | 输入数据   | 数据类型 | 大小                      | 数据排布格式 |
  | --------   | -------- | ------------------------- | ------------ |
  | src_tokens | INT64    | batchsize x seq_len       | ND           |

- 输出数据

  | 输出数据 | 数据类型 | 大小                  | 数据排布格式 |
  | -------- | -------- | --------              | ------------ |
  | output   | FLOAT32  | batchsize x num_class | ND           |



# 推理环境准备

- 该模型需要以下依赖

  **表 1**  版本配套表


| 配套                      | 版本       |
| ------------------------- | --------------- |
| CANN                      | 7.0.RC1.alpha003   |
| Python                    | 3.9        | -                          |              |
| PyTorch （cuda）                   | 2.0.1      |
| torchAudio   (cuda)           | 2.0.1       |
| Ascend-cann-torch-aie | 6.3.T200           
| Ascend-cann-aie       | 6.3.T200        
| 芯片类型                  | Ascend310P3     | 


## 安装CANN包

 ```
 chmod +x Ascend-cann-toolkit_7.0.RC1.alpha003_linux-x86_64.run
./Ascend-cann-toolkit_7.0.RC1.alpha003_linux-x86_64.run --install
 ```

## 安装Ascend-cann-aie
1. 安装
```
chmod +x ./Ascend-cann-aie_${version}_linux-${arch}.run
./Ascend-cann-aie_${version}_linux-${arch}.run --check
# 默认路径安装
./Ascend-cann-aie_${version}_linux-${arch}.run --install
# 指定路径安装
./Ascend-cann-aie_${version}_linux-${arch}.run --install-path=${AieInstallPath}
```
2. 设置环境变量
```
cd ${AieInstallPath}
source set_env.sh
```
## 安装Ascend-cann-torch-aie
1. 安装
 ```
# 安装依赖
conda create -n py39_pt2.0 python=3.9.0 -c pytorch -y
conda install decorator -y
pip install attrs
pip install scipy
pip install synr==0.5.0
pip install tornado
pip install psutil
pip install cloudpickle
wget https://download.pytorch.org/whl/cpu/torch-2.0.1%2Bcpu-cp39-cp39-linux_x86_64.whl
pip install torch-2.0.1+cpu-cp39-cp39-linux_x86_64.whl

# 解压
tar -xvf Ascend-cann-torch-aie-${version}-linux-${arch}.tar.gz
cd Ascend-cann-torch-aie-${version}-linux-${arch}

# C++运行模式
chmod +x Ascend-cann-torch-aie_${version}_linux-${arch}.run
# 默认路径安装
./Ascend-cann-torch-aie_${version}_linux-${arch}.run --install
# 指定路径安装
./Ascend-cann-torch-aie_${version}_linux-${arch}.run --install-path=${TorchAIEInstallPath}

# python运行模式
pip install torch_aie-${version}-cp{pyVersion}-linux_x86_64.whl
 ```
 > 说明：如果用户环境是[libtorch1.11](https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-1.11.0%2Bcu113.zip)，需要使用配套的torch 1.11-cpu生成torchscript，再配套使用torch-aie-torch1.11的包。

2. 设置环境变量
```
cd ${TorchAIEInstallPath}
source set_env.sh
```



# 快速上手

1. 获取源码

   通过Git获取对应版本的代码并安装的方法如下：
   ```bash
   git clone https://github.com/huggingface/transformers.git    # 克隆仓库的代码
   cd transformers                                              # 切换到模型的代码仓目录
   git checkout v4.20.0                                         # 切换到对应版本
   git reset --hard 39b4aba54d349f35e2f0bd4addbe21847d037e9e    # 将暂存区与工作区都回到上一次版本
   pip3 install ./                                              # 通过源码进行安装
   cd ..
   ```

2. 安装依赖

   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
   ```

   ```
   pip3 install -r requirements.txt
   ```

  安装模型依赖:

   ```
   git clone https://github.com/pytorch/fairseq.git fairseq_workspace
   cd fairseq_workspace
   git checkout c1624b27
   git apply ../roberta-infer.patch
   pip3 install --editable ./
   ```




## 准备数据集

本模型使用 [SST-2官方数据集](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip)，解压到 `./data` 目录，如 `./data/SST-2` ,目录结构如下：

    ```
    ├── data
    |   ├── SST-2
    |   |    ├── test.tsv
    │   |    ├── dev.tsv
    │   |    ├── train.tsv
    │   |    ├── original/
    ```

   执行预处理脚本:

   ```
   # 使用代码仓自带脚本下载&&完成部分前置处理工作
   bash fairseq_workspace/examples/roberta/preprocess_GLUE_tasks.sh data/ SST-2
   若提示下载失败，则修改preprocess_GLUE_tasks.sh中wget部分代码
   将：
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
   修改为：
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json' --no-check-certificate
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe' --no-check-certificate
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt' --no-check-certificate
    

   # 生成预处理数据， 请按需修改batch_size和pad_length
    python3 RoBERTa_preprocess.py --data_path ./data/SST-2-bin --pad_length 70 --batch_size 4 
   ```

   - 参数说明：

        - --data_path: 数据集所在路径，输出数据保存在其下层目录下

     - --pad_length: 模型输入seq长度

  对于指定的pad_length, 每一个batch将生成的对应的预处理数据作为一个bin文件。 例如pad_len = 70, bs = 1 对应的输出数据在`./data/SST-2-bin/roberta_base_bin_bs1_pad70`目录下



## 模型推理
1. 获取权重文件：

    [RoBERTa模型pth权重文件](https://pan.baidu.com/s/1GZnnpz8fek2w7ARsZ0ujnA)，密码: x248。

   解压后将checkpoint.pt文件放至 `./checkpoints` 目录下(如没有则新建该目录)。



2. 生成torchscript模型

    ```
     # 以bs1， pad70为例
     python3 torch2ts.py --checkpoint_path checkpoints/ --checkpoint_file checkpoint.pt --data_name_or_path ./data/SST-2-bin --batch_size 1 --pad_length 70
    ```

    - 参数说明：
        - --checkpoint_path：权重文件所在目录

        - --checkpoint_file：权重文件名
           
        - --data_name_or_path: 数据集路径
           
        - --batch_size: 模型batchsize
           
        - --pad_length: 模型输入seq长度

    获得roberta_traced.pth 文件。





3. 执行模型编译和推理脚本（包含推理性能测试）

   ```bash
   # 执行compile.py后对应batch_size的模型性能会打印到终端
   python compile.py --torchscript_path ./roberta_traced.pth --input_path ./data/SST-2-bin/ --output_path ./data/SST-2-bin/output/ --pad_length 70 --batch_size 4 
   ```
      - 参数说明：
         - --input_path：推理前输入数据路径。
         - --output_path ：推理后数据输出路径
         - --torchscript_path: trace 之后保存的torchscript模型路径。



4. 模型推理的精度验证

   此脚本只支持bs=1
   ```bash
   # 执行RoBERTa_postprocess.py后对应batch_size的模型精度会打印到终端
    python3 RoBERTa_postprocess.py --res_path=./data/SST-2-bin/output/bs1_pad70 --data_path=./data/SST-2-bin
   ```
   - 参数说明：
      - --res_path：推理后的输出数据路径, 请按需修改pad
      - --data_path : 数据集路径




# 模型静态推理性能&精度

    
   | NPU芯片型号 | Batch Size |  数据集   |  精度 | 性能|
   | :-------:  | :--------: | :---------: | :-----: | :----: | 
   |Ascend310P3 |      1     | SST-2 |   Acc: 91.8% |  356   |     
   |Ascend310P3 |      4     | SST-2 |    |   764  |     
   |Ascend310P3 |      8     | SST-2 |    |   942  |     
   |Ascend310P3 |      16    | SST-2 |    |   1027  |     
   |Ascend310P3 |      32    | SST-2 |    |    985 |     
   |Ascend310P3 |      64    | SST-2 |    |    958 |     

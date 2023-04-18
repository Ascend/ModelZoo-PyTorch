# mBART for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

MBART 是一种序列到序列去噪自动编码器，使用 BART 目标在多种语言的大规模单语语料库上进行预训练。mBART 是通过对多种语言的全文进行去噪来预训练完整序列到序列模型的首批方法之一，而以前的方法只关注编码器、解码器或重建文本的一部分。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/fairseq/tree/v0.10.2/examples/mbart
  commit_id=de859692ff39cff1ecfd65e8e6860c621fb0e58a
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/nlp
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |
  | PyTorch 1.11   | - |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  在工程根目录下执行 
  pip3.7 install -e ./ 
  pip3.7 install -r requirements.txt
  git clone https://github.com/moses-smt/mosesdecoder.git
  git clone https://github.com/rsennrich/wmt16-scripts.git
  pip3.7 install sacrebleu==1.5.1
  ```


## 准备数据集

## 方法一. 下载已预处理好的数据集
1. 下载train_data.tar
2. tar -xvf train_data.tar
3. 将数据集放于工程根目录下，其目录结构如下:
```bash
train_data
    | -- en_ro
        | -- preprocess.log
        | -- dict.en_XX.txt
        | -- dict.ro_RO.txt
        | -- test.en_XX-ro_RO.ro_RO.bin
        | -- test.en_XX-ro_RO.ro_RO.idx
        | -- test.en_XX-ro_RO.en_XX.bin
        | -- test.en_XX-ro_RO.en_XX.idx
        | -- train.en_XX-ro_RO.ro_RO.bin
        | -- train.en_XX-ro_RO.ro_RO.idx
        | -- train.en_XX-ro_RO.en_XX.bin
        | -- train.en_XX-ro_RO.en_XX.idx
        | -- valid.en_XX-ro_RO.ro_RO.bin
        | -- valid.en_XX-ro_RO.ro_RO.idx
        | -- valid.en_XX-ro_RO.en_XX.bin
        | -- valid.en_XX-ro_RO.en_XX.idx
    | -- en_de
        | -- preprocess.log
        | -- dict.en_XX.txt
        | -- dict.de_DE.txt
        | -- test.en_XX-de_DE.de_DE.bin
        | -- test.en_XX-de_DE.de_DE.idx
        | -- test.en_XX-de_DE.en_XX.bin
        | -- test.en_XX-de_DE.en_XX.idx
        | -- train.en_XX-de_DE.de_DE.bin
        | -- train.en_XX-de_DE.de_DE.idx
        | -- train.en_XX-de_DE.en_XX.bin
        | -- train.en_XX-de_DE.en_XX.idx
        | -- valid.en_XX-de_DE.de_DE.bin
        | -- valid.en_XX-de_DE.de_DE.idx
        | -- valid.en_XX-de_DE.en_XX.bin
        | -- valid.en_XX-de_DE.en_XX.idx

```
> **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

## 方法二. 下载数据集并自行处理
### 1. 分词处理
1. 下载原始数据集并放于在源码包根目录下新建的“src_data/”目录下，以en_ro数据集为例。
2. 下载并安装SPM
```bash
SPM=/path/to/sentencepiece/build/src/spm_encode
MODEL=sentence.bpe.model
DATA=path_2_data
SRC=en_XX
TGT=ro_RO
TRAIN=train
VALID=valid
TEST=test
${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${SRC} > ${DATA}/${TRAIN}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${TGT} > ${DATA}/${TRAIN}.spm.${TGT} &
${SPM} --model=${MODEL} < ${DATA}/${VALID}.${SRC} > ${DATA}/${VALID}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${VALID}.${TGT} > ${DATA}/${VALID}.spm.${TGT} &
${SPM} --model=${MODEL} < ${DATA}/${TEST}.${SRC} > ${DATA}/${TEST}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${TEST}.${TGT} > ${DATA}/${TEST}.spm.${TGT} &
```


### 2. 数据预处理

```bash
DICT=dict.txt
DATA=/path/data/   //用户可以根据实际情况进行修改， 如 DATA=./src_data/ （原始数据集所在路径）
DEST=/path/dest/   //用户可以根据实际情况进行修改， 如 DEST=./train_data/ （处理后的数据集所在路径）
NAME=en_ro
TRAIN=train
TEST=test
SRC=en_XX
TGT=ro_RO
VALID=valid
TEST=test
fairseq-preprocess \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref ${DATA}/${TRAIN}.spm \
  --validpref ${DATA}/${VALID}.spm \
  --testpref ${DATA}/${TEST}.spm \
  --destdir ${DEST}/${NAME} \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict ${DICT} \
  --tgtdict ${DICT} \
  --workers 70

```
## 获取预训练模型

1. 下载mbart.CC25.tar.gz
   
2. tar -xzvf mbart.CC25.tar.gz
3. 将模型放于工程根目录下，其目录结构如下:
```
mbart.cc25
    | -- model.pt
    | -- dict.txt
    | -- sentence.bpe.model
```

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```
     data_path为数据集路径，若训练en_ro数据集，路径写到en_ro；若训练en_de数据集，路径写到en_de ，同时需要将训练脚本中dropout的参数设置为0.1，target-lang设置为de_DE
     
   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```
     
     data_path为数据集路径，若训练en_ro数据集，路径写到en_ro；若训练en_de数据集，路径写到en_de ，同时需要将训练脚本中dropout的参数设置为0.1，total-num-update与max-update设置为300000，target-lang设置为de_DE

    
   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                              //数据集路径
   --npu-id                            //使用的NPU的id
   --max-update                        //训练步数
   --max-epoch                         //重复训练次数
   --token_size                        //批大小
   --lr                                //学习率
   --seed                              //使用随机数种子
   --optimizer                         //使用的优化器
   --distributed-world-size            //是否使用多进程在多GPU节点上进行分布式训练
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  en_ro数据集训练结果展示表
| NAME  | Acc@1  | FPS  | Epochs | AMP_Type | Torch_Version |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 8p-竞品V  | - | 39281.96 | - | - | 1.8 |
| 8p-NPU   | 37.4 | 36171.24 | - | - | 1.8 |

**表 3**  en_de数据集训练结果展示表
| NAME  | Acc@1  | FPS  | Epochs | AMP_Type | Torch_Version |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 8p-竞品V  | - | 38365.15 |- | - | 1.8 |
| 8p-NPU   | 32.5  | 35320.3   |- | - | 1.8 |

> **说明：** 
   >由于该模型默认开启二进制，所以在性能测试时，需要安装二进制包

# 版本说明

## 变更

2022.12.14：首次发布。

## FAQ

无。

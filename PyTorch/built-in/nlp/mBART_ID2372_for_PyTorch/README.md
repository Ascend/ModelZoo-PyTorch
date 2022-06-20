# MBART: Multilingual Denoising Pre-training for Neural Machine Translation

# 安装fairseq

```bash
在工程根目录下执行pip3.7 install -e ./ 
```
# 下载预处理模型
1. 下载mbart.CC25.tar.gz
2. tar -xzvf mbart.CC25.tar.gz
3. 将模型放于工程根目录下，其目录结构如下:
```bash
mbart.cc25
    | -- model.pt
    | -- dict.txt
    | -- sentence.bpe.model
```

# 数据集
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

## 方法二. 下载数据集并自行处理
### 1. 分词处理
1. 下载数据集并放于工程根目录下，以en_ro数据集为例
2. 下载并安装SPM [here](https://github.com/google/sentencepiece)
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
DATA=/path/data/
DEST=/path/dest/
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

# 模型训练
单卡训练
    执行 bash ./test/train_performance_1p.sh --data_path=./train_data/en_ro   
[ data_path为数据集路径，若训练en_ro数据集，路径写到en_ro；若训练en_de数据集，路径写到en_de ，同时需要将train_performance_1p.sh中dropout的参数设置为0.1，target-lang设置为de_DE ]
 [ 若传参 --ckpt，则ckpt 为 model.pt的路径 ]
```

多卡训练
1 、下载依赖评估包
```bash  
git clone https://github.com/moses-smt/mosesdecoder.git
git clone https://github.com/rsennrich/wmt16-scripts.git
pip3.7 install sacrebleu==1.5.1
```
2 、执行 bash ./test/train_full_8p.sh --data_path=./train_data/en_ro  
[ data_path为数据集路径，若训练en_ro数据集，路径写到en_ro；若训练en_de数据集，路径写到en_de ，同时需要将train_full_8p.sh中dropout的参数设置为0.1，total-num-update与max-update设置为300000，target-lang设置为de_DE ]
[ 若传参 --ckpt，则ckpt 为 model.pt的路径 ]

# Docker容器训练

1.导入镜像二进制包

```bash
docker import ubuntuarmpytorch.tar pytorch:b***
```

2.执行docker_start.sh

```
./docker_start.sh pytorch:b*** /path/data /path/mbart
```

3.执行正常安装及训练步骤

# Jasper Onnx模型端到端推理指导

- [Jasper Onnx模型端到端推理指导](#jasper-onnx模型端到端推理指导)
  - [1. 模型概述](#1-模型概述)
    - [1.1 论文地址](#11-论文地址)
    - [1.2 代码地址](#12-代码地址)
  - [2. 环境准备](#2-环境准备)
    - [2.1 文件说明](#21-文件说明)
    - [2.2 环境依赖准备](#22-环境依赖准备)
  - [3. 模型转换](#3-模型转换)
    - [3.1 pth转onnx模型](#31-pth转onnx模型)
    - [3.2 onnx转om模型](#32-onnx转om模型)
  - [4. 端到端推理及验证](#4-端到端推理及验证)
    - [4.1 离线推理](#41-离线推理)
    - [4.2 精度验证](#42-精度验证)
    - [4.3 性能验证](#43-性能验证)



## 1. 模型概述

Jasper是应用于自动语音识别（ASR）的端到端声学模型，该模型在不借助任何外部数据的情况下在LibriSpeech数据集上取得了SOTA的结果。

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址

[J. Li, V. Lavrukhin, B. Ginsburg, R. Leary, O. Kuchaiev, J. M.Cohen, H. Nguyen, and R. T. Gadde, “Jasper: An End-to-End Convolutional Neural Acoustic Model,” in arXiv, 2019.](https://arxiv.org/pdf/1904.03288.pdf)

### 1.2 代码地址

```
url=https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper 
branch=master 
commit_id=15af494a8e7e0c33fcbdc6ef9cc12e3929e313aa
```

通过Git获取对应commit_id的代码方法如下：

```
git clone {repository_url}        # 克隆仓库的代码
cd {repository_name}              # 切换到模型的代码仓目录
git checkout {branch}             # 切换到对应分支
git reset --hard {commit_id}      # 代码设置到对应的commit_id
cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
```

## 2. 环境准备

-   **[文件说明](#21-文件说明)**  

-   **[环境依赖准备](#22-环境依赖准备)**  

### 2.1 文件说明

  1. `acl_net.py`：PyACL推理工具代码
  2. `om_infer_acl.py`：Jasper推理代码，基于om推理
  3. `Jasper_pth2onnx.py`：根据pth文件得到onnx模型

### 2.2 环境依赖准备

本环境在Ubuntu 18.04，python3.7.11 版本下测试通过，具体环境依赖见requirements.txt文件，用户可根据自己的运行环境自行安装所需依赖

- 文件下载
  
  - 源码下载
  
    下载[Jasper源码](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper)并解压文件，切换到 `DeepLearningExamples/PyTorch/SpeechRecognition/Jasper` 目录下。
  
  - 权重下载
  
    ```shell
    mkdir ./checkpoints
    wget -P ./checkpoints https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/audio/Jasper/jasper_fp16.pt
    ```
  
  - 数据集下载
  
    下载 [LibriSpeech-test-other.tar.gz](https://www.openslr.org/resources/12/test-other.tar.gz)数据集并根据 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper#quick-start-guide) 的 `Download and preprocess the dataset` 部分进行数据预处理，将原始 `.flac` 文件转换成 `.wav` 文件，数据集预处理成功后Jasper目录下应存在如下结构目录
    
    ```
    dataset/LibriSpeech/
    ├── dev-clean 
    ├── dev-clean-wav
    ├── librispeech-dev-clean-wav.json
    ```
  
- 文件拷贝®

  1. 将源码包中除```Jasper.patch```的其他文件移动到`DeepLearningExamples/PyTorch/SpeechRecognition/Jasper` 目录下
  
  2. 将源码包中`Jasper.patch`文件移动到代码仓`DeepLearningExamples`目录下执行
  
     ```bash
     git apply Jasper.patch
     ```
- 安装依赖
  ```
  pip3 install ONNX==1.7.0
  pip3 install librosa==0.8.0
  pip3 install Pytorch==1.5.0
  pip3 install numpy==1.18.5
  pip3 install ascii-graph==1.5.1
  pip3 install ipdb
  pip3 install pandas==1.1.4
  pip3 install pyyaml
  pip3 install soundfile
  apt-get install sox
  pip3 install sox==1.4.1
  pip3 install tqdm==4.53.0
  pip3 install wrapt==1.10.11
  pip3 install unidecode==1.2.0
  pip3 install inflect==5.3.0
  ```
## 3. 模型转换

- **[pth转onnx模型](#31-pth转onnx模型)** 

- **[onnx转om模型](#32-onnx转om模型)**

### 3.1 pth转onnx模型

```
#导入pth模型。
mkdir checkpoints
mv  nvidia_jasper_210205  checkpoints/jasper_fp16.pt。
```

```
注释所有apex依赖和源工程代码修改。
将源码中diff.patch文件移动到代码仓“DeepLearningExamples”目录下，根据diff.path内容，手动调整/PyTorch/SpeechRecognition/Jasper/common/features.py /PyTorch/SpeechRecognition/Jasper/common/helpers.py /PyTorch/SpeechRecognition/Jasper/jasper/model.py三个文件
```


```bash
# Jasper_pth2onnx.py需要三个参数，第一个为pth模型路径，第二个为转换后的模型，第三个为模型的batch size
# 生成batch size为1的onnx模型
python3.7 Jasper_pth2onnx.py checkpoints/jasper_fp16.pt jasper.onnx 1
```

因为atc工具目前对动态shape场景支持度不高，官方提供的onnx模型给模型调测带来较大困难，所以需要使用pth2onnx脚本重新生成带feat_lens的模型。

### 3.2 onnx转om模型

1. 设置环境变量

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh	
   ```

2. 使用atc将onnx模型转换为om模型文件
  
    ${chip_name}可通过`npu-smi info`指令查看
   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)


   ```bash
   # 将jasper_1batch.onnx模型转换为jasper_1batch.om，对于不同batch size的onnx模型，需要修改input_shape参数重feats的第一维
   atc --model=jasper.onnx \
       --framework=5 \
       --input_format=ND \
       --input_shape="feats:1,64,4000;feat_lens:1" \
       --output=jasper_1batch \
       --soc_version=${chip_name} \
       --log=error
   ```

## 4. 端到端推理及验证

- **[离线推理](#41-离线推理)** 
- **[精度验证](#42-精度验证)**

- **[性能验证](#43-性能验证)**

### 4.1 离线推理

```bash
datasets_path="./dataset/LibriSpeech/"

# 使用jasper_1batch.om模型在LibriSpeech数据集的dev-clean上进行推理，推理结果保存在result_bs1.txt中
# 执行离线推理后会输出wer值，与参考精度值3.20比较，保证精度差异在1%以内即可。
# 对于不同batch size的om模型，需要修改batch_size参数
python3.7 om_infer_acl.py \
        --batch_size 1 \
        --model ./jasper_1batch.om \
        --val_manifests ${datasets_path}/librispeech-dev-clean-wav.json \
        --model_config configs/jasper10x5dr_speedp-online_speca.yaml \
        --dataset_dir ${datasets_path} \
        --max_duration 40 \
        --pad_to_max_duration \
        --save_predictions ./result_bs1.txt
```

### 4.2 精度验证

执行离线推理后会输出wer值，与参考精度值9.66比较，保证精度差异在1%以内即可。

| 模型      | pth精度 | 310精度   | 310P精度
| -------- | ------- | ------- |-----|
| Jasper   | 9.66   | 9.730| 9.726   
### 4.3 性能验证

使用ais_infer.py推理测试模型性能 [ais_infer具体参考](http://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)


``````shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh

获取ais_infer
pip3 install aclruntime-0.0.1-cp37-cp37m-linux_x86_64.whl
git clone https://gitee.com/ascend/tools.git


推理
python3 {ais_infer_path}/ais_infer.py --model {jasper_path}/jasper_batch_1.om --output ./ --outfmt BIN --batchsize 1

--model：模型地址
--input：预处理完的数据集文件夹
--output：推理结果保存地址
--outfmt：推理结果保存格式
--batchsize: batchsize的值
``````

纯推理后性能结果保存在```result/PureInfer_perf_of_jasper_1batch_in_device_0.txt```，使用tail命令查看性能数据

```bash
tail result/PureInfer_perf_of_jasper_1batch_in_device_0.txt
```
性能结果
|batch_size|310       |310P        |
|----------|----------|------------|
|bs1       |17.9528057|23.9646092  |
|bs4       |19.6234026|20.59439543 |
|bs8       |19.44036742|21.2358747 |
|bs16      |19.74849299|28.21203738|
|bs32      |19.80529607|28.21540613|
|bs64      |19.79865766|26.20766411|
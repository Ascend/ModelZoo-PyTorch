# Jasper Onnx模型端到端推理指导

-   [1 模型概述](#1-模型概述)
	-   [1.1 论文地址](#11-论文地址)
	-   [1.2 代码地址](#12-代码地址)
-   [2 环境准备](#2-环境准备)
	-   [2.1 深度学习框架](#21-深度学习框架)
	-   [2.2 python第三方库](#22-python第三方库)
-   [3 模型转换](#3-模型转换)
	-   [3.1 pth转onnx模型](#31-pth转onnx模型)
	-   [3.2 onnx转om模型](#32-onnx转om模型)
-   [4 端到端推理及验证](#4-端到端推理及验证)
  -   [4.1 离线推理](#41-离线推理)
  -   [4.2 精度验证](#42-精度验证)
  -   [4.3 性能验证](#43-性能验证)



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

## 3. 模型转换

- **[pth转onnx模型](#31-pth转onnx模型)** 

- **[onnx转om模型](#32-onnx转om模型)**

### 3.1 pth转onnx模型

```bash
# Jasper_pth2onnx.py需要三个参数，第一个为pth模型路径，第二个为转换后的模型，第三个为模型的batch size
# 生成batch size为1的onnx模型
python3.7 Jasper_pth2onnx.py checkpoints/jasper_fp16.pt jasper_1batch.onnx 1
```

因为atc工具目前对动态shape场景支持度不高，官方提供的onnx模型给模型调测带来较大困难，所以需要使用pth2onnx脚本重新生成带feat_lens的模型。

### 3.2 onnx转om模型

1. 设置环境变量

   ```bash
   source env.sh		
   ```

2. 使用atc将onnx模型转换为om模型文件

   ```bash
   # 将jasper_1batch.onnx模型转换为jasper_1batch.om，对于不同batch size的onnx模型，需要修改input_shape参数重feats的第一维
   atc --model=jasper_1batch.onnx \
       --framework=5 \
       --input_format=ND \
       --input_shape="feats:1,64,4000;feat_lens:1" \
       --output=jasper_1batch \
       --soc_version=Ascend310 \
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

执行离线推理后会输出wer值，与参考精度值3.20比较，保证精度差异在1%以内即可。

### 4.3 性能验证

使用benchmark纯推理测试模型性能

``````shell
source env.sh

arch=`uname -m`
chmod u+x benchmark.${arch}
./benchmark.${arch} -batch_size=1 -om_path=./jasper_1batch.om -round=50 -device_id=0 
``````

纯推理后性能结果保存在```result/PureInfer_perf_of_jasper_1batch_in_device_0.txt```，使用tail命令查看性能数据

```bash
tail result/PureInfer_perf_of_jasper_1batch_in_device_0.txt
```

显示结果为

```
ave_throughputRate = 5.20623samples/s, ave_latency = 192.894ms
```

batch1 310单卡吞吐率：5.20623x4=20.8fps 

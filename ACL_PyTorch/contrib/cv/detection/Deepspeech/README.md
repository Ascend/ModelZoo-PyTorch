# DeepSpeech模型PyTorch离线推理指导
-   [1 环境说明](#1-环境说明)
	-   [1.1 环境搭建与使用说明](#11-环境搭建与使用说明)
-   [2 推理流程](#2-推理流程)
	-   [2.1 获取开源PyTorch模型代码与权重文件](#21-获取开源PyTorch模型代码与权重文件)
	-   [2.2 导出onnx模型](#22-导出onnx模型)
	-   [2.3 转换为om模型](#23-转换为om模型) 
	-   [2.4 数据集处理](#24-数据集处理)  
	-   [2.5 离线推理](#25-离线推理)
-   [3 精度统计](#3-精度统计)
	-   [3.1 离线推理精度](#31-离线推理精度)
	-   [3.2 精度对比](#32-精度对比)
-   [4 性能对比](#4-性能对比)
	-   [4.1 npu性能数据](#41-npu性能数据) 
	-   [4.2 gpu性能数据](#42-gpu性能数据)
	-   [4.3 性能数据对比](#43-性能数据对比)
        



## 1 环境说明

-   **[环境搭建与使用说明](#11-环境搭建与使用说明)**  


### 1.1 环境搭建与使用说明


深度学习框架与第三方库

```
python3.7.5

torch == 1.8.0
torchaudio == 0.8.0
torchvision == 0.9.0
torchelastic == 0.2.2

onnx
onnxruntime
onnxoptimizer

fairscale
flask
google-cloud-storage
hydra-core
jupyter
librosa
matplotlib
numpy
optuna
pytest
python-levenshtein
pytorch-lightning>=1.1
scipy
sklearn
sox
tqdm
wget
git+https://github.com/romesco/hydra-lightning/#subdirectory=hydra-configs-pytorch-lightning
```

其中apex安装使用pip install会报错，应使用下述方式安装：
```
git clone https://github.com/NVIDIA/apex.git
cd apex
python3 setup.py install --cpp_ext --cuda_ext
```

**说明：** 
> 
> X86架构：pytorch和torchvision可以通过官方下载whl包安装，其他可以通过pip3.7 install 包名 安装
>
> Arm架构：pytorch，torchvision和opencv可以通过github下载源码编译安装，其他可以通过pip3.7 install 包名 安装
>
> 以上为多数网络需要安装的软件与推荐的版本，根据实际情况安装。如果python脚本运行过程中import 模块失败，安装相应模块即可，如果报错是缺少动态库，网上搜索报错信息找到相应安装包，执行apt-get install 包名安装即可



## 2 推理流程

-   **[获取开源PyTorch模型代码与权重文件](#21-获取开源PyTorch模型代码与权重文件)**  

-   **[导出onnx模型](#22-导出onnx模型)**  

-   **[转换为om模型](#23-转换为om模型)**  

-   **[数据集处理](#24-数据集处理)**  

-   **[离线推理](#25-离线推理)**


### 2.1 获取开源PyTorch模型代码与权重文件

#### 2.1.1 基于开源PyTorch框架的Deepspeech开源模型代码
```
git clone https://github.com/SeanNaren/deepspeech.pytorch.git -b V3.0   
```
#### 2.1.2 修改deepspeech.pytorch/deepspeech_pytorch/model.py

#### 2.1.3 [下载ckpt权重文件](https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/an4_pretrained_v3.ckpt)
```
wget https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/an4_pretrained_v3.ckpt
```


### 2.2 导出onnx模型

#### 2.2.1 配置环境变量
 - 将env.sh文件放到根目录下
 - source环境变量
```
source env.sh
```
#### 2.2.2 将ckpt2onnx.py放到根目录下
#### 2.2.3 执行pth2onnx脚本，生成onnx模型文件
```
python3 ckpt2onnx.py --ckpt_path ./an4_pretrained_v3.ckpt --out_file deepspeech.onnx
```
**说明：** 
> 
> --ckpt_path：ckpt权重文件
> --out_file：生成的onnx文件名
>


### 2.3 转换为om模型

#### 2.3.1 设置环境变量
```
source env.sh
```
#### 2.3.2 使用Ascend atc工具将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01
```
atc --framework=5 --model=./deepspeech.onnx --input_format=NCHW --input_shape="spect:1,1,161,621;transcript:1" --output=deepspeech_bs1 --log=debug --soc_version=Ascend310
```


### 2.4 数据集处理

#### 2.4.1 获取数据集
获取AN4数据集  
```
cd deepspeech.pytorch/data
python3 an4.py
cd ../..
```

#### 2.4.2 数据预处理
 - 将预处理脚本deepspeech_preprocess.py放到根目录下
 - 执行预处理脚本，生成数据集预处理后的bin文件
```
python3 deepspeech_preprocess.py --data_file ./deepspeech.pytorch/data/an4_test_manifest.json --save_path ./deepspeech.pytorch/data/an4_dataset/test --label_file ./deepspeech.pytorch/labels.json
```
**说明：** 
> 
> --data_file：存放数据路径的json文件
> --save_path：预处理产生的bin文件存储路径（会在save_path目录下建立两个文件夹spect和sizes分别存放两组输入文件）
> --label_file: labels文件路径
>


### 2.5 离线推理

#### 2.5.1 msame工具概述
输入.om模型和模型所需要的输入bin文件，输出模型的输出数据文件，支持多次推理（指对同一输入数据进行推理）。
模型必须是通过atc工具转换的om模型，输入bin文件需要符合模型的输入要求（支持模型多输入）。
**说明：** 
> 
> benchmark工具暂不支持多输入，因此改用msame
>
#### 2.5.2 离线推理
 - [获取msame工具](https://gitee.com/ascend/tools/tree/master/msame)
 - 执行离线推理
```
./msame --model "./deepspeech_bs1.om" --input "./deepspeech.pytorch/data/an4_dataset/test/spect,./deepspeech.pytorch/data/an4_dataset/test/sizes" --output "./deepspeech.pytorch/result" --outfmt TXT

```
**说明：** 
> 
> 将/tools/msame/msame文件复制到根目录下，执行上述命令，或直接在msame文件夹下执行命令，将input、output等路径改为绝对路径
> 输出保存在--output路径下，会自动生成新文件夹
>



## 3 精度统计

-   **[离线推理精度](#31-离线推理精度)**  

-   **[精度对比](#32-精度对比)**  


### 3.1 离线推理精度
 - 将后处理脚本deepspeech_postprocess.py放到根目录下
 - 调用后处理脚本产生推理结果
```
python3 deepspeech_postprocess.py --out_path ./deepspeech.pytorch/result --info_path ./deepspeech.pytorch/data/an4_dataset/test --label_file ./deepspeech.pytorch/labels.json
```
**说明：** 
> 
> --out_path：离线推理输出的路径，是msame推理后的输出路径
> --info_path：与执行数据预处理脚本deepspeech_preprocess.py时设置的--save_path一致
> --label_file: labels文件路径
> 

### 3.2 精度对比

| 模型      | 官网ckpt精度  | 310离线推理精度  |
| :------: | :------: | :------: |
| Deepspeech bs1  | [Average WER 9.573 Average CER 5.515](https://github.com/SeanNaren/deepspeech.pytorch/releases) | Average WER 9.573 Average CER 5.515 |

**说明：** 
> 
> 将得到的om离线模型推理精度与该模型github代码仓上公布的精度对比，精度与之一致，故精度达标
> 



## 4 性能对比

-   **[npu性能数据](#41-npu性能数据)**  

-   **[gpu性能数据](#42-gpu性能数据)** 

-   **[性能优化](#43-性能优化)** 

 
### 4.1 npu性能数据
由于benchmark工具不支持多输入，改为使用msame进行om的离线推理。msame工具在推理时会输出每条数据运行的时间，计算10条数据运行的时间均值，作为性能的衡量标准。由于msame不支持多batch，因此以bs1的数据为准。
```
Run time of each data: 9.09s
performance: 0.11seq/s
```

### 4.2 gpu性能数据
在装有T4卡的服务器上测试gpu性能，在GPU上进行在线推理，取5次运行的平均时长作为性能的衡量标准。
```
Run time of each data: 0.28s
performance: 3.44seq/s
```

### 4.3 性能优化
使用性能分析工具profiling，查看了模型中每类算子总体耗时与百分比和模型每个算子的aicore耗时，发现DynamicRNN耗时最多，使用autotune进行性能优化，优化后性能如下：
```
Run time of each data: 2.03s
performance: 0.49seq/s
```
在此基础上，对TransData算子进行优化，优化后性能如下：
```
Run time of each data: 1.41s
performance: 0.71seq/s
```
# IntraDA-deeplabv2 ONNX模型端到端推理指导

- [1 模型概述](#1-模型概述)
    -   [1.1 论文地址](#11-论文地址)
    -   [1.2 代码地址](#12-代码地址)
- [2 环境说明](#2-环境说明)
    -   [2.1 深度学习框架](#21-深度学习框架)
    -   [2.2 python第三方库](#22-python第三方库)
- [3 模型转换](#3-模型转换)
    -   [3.1 pth转onnx模型](#31-pth转onnx模型)
    -   [3.2 onnx转om模型](#32-onnx转om模型)
- [4 数据集预处理](#4-数据集预处理)
    -   [4.1 数据集获取](#41-数据集获取)
    -   [4.2 数据集预处理](#42-数据集预处理)
    -   [4.3 生成数据集info文件](#43-生成数据集info文件)
- [5 离线推理](#5-离线推理)
    -   [5.1 benchmark工具概述](#51-benchmark工具概述)
    -   [5.2 离线推理](#52-离线推理)
- [6 精度对比](#6-精度对比)
    -   [6.1 离线推理精度统计](#61-离线推理Acc精度统计)
- [7 性能对比](#7-性能对比)
    - [7.1 npu性能数据](#71-npu性能数据)



## 1 模型概述
- **[论文地址](#11-论文地址)**
- **[代码地址](#12-论文地址)**
### 1.1 论文地址
[InstraDA论文](https://arxiv.org/abs/2004.07703)
### 1.2 代码地址
[InstraDA代码](https://github.com/feipan664/IntraDA)

branch=master 

commit_id=070b0b702fe94a34288eba4ca990410b5aaadc4a
## 2 环境准备 

- **[深度学习框架](#21-深度学习框架)**
- **[python第三方库](#22-python第三方库)**

### 2.1 深度学习框架
```
CANN 5.1.RC1
pytorch == 1.8.1
torchvision == 0.9.1
onnx == 1.11.0
```
### 2.2 python第三方库

```
numpy == 1.22.3
Pillow == 8.3.2
```

**说明：**
>可通过pip install -r requirements.txt  安装环境
>测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
> pytorch以及torchvision可在pytorch官网查看具体版本的适配关系以及安装命令

## 3 模型转换

- **[pth转onnx模型](#31-pth转onnx模型)**  
- **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型
1.准备pth权重文件  
[deeplabv2 InTraDA训练的pth权重文件](https://z1hqmg.dm.files.1drv.com/y4mWPCqb6XAb6jMyXF82UaWj8kkUvBaUydSOdglID3YB_r1dolC0cc3-cWlI5RH7cN5PuKf96Bqg3e366STMbXLqpGmze8-gXy7Lq71OEAEx7ZSHjp9wNIIPNCNxE2F1s6u8xRxClXO2K5Tbs8Tde4hiGSm119pGKmn3W8X4U7IQZa8KS5lS5BztIOKkqEH1hZol7ELwbqm7i5TRzPPIMsycg)

2.导出onnx文件
(1)当前目录结构
![img_2.png](img_2.png)

(1)安装IntraDA
```shell
git clone https://github.com/feipan664/IntraDA.git -b master
cd IntraDA
git reset --hard 070b0b702fe94a34288eba4ca990410b5aaadc4a
pip3.7 install -e ./ADVENT
cd ..
```
(2)运行intrada_pth2onnx.py脚本
```shell
python3.7 intrada_pth2onnx.py ./cityscapes_easy2hard_intrada_with_norm.pth ./intraDA_deeplabv2.onnx
```

### 3.2 onnx转om模型
1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01
```
atc \
--model=intraDA_deeplabv2.onnx \
--framework=5 \
--output=intraDA_deeplabv2_bs1 \
--input_format=NCHW \
--input_shape="image:1,3,512,1024" \
--log=debug \
--soc_version=Ascend${chip_name}
```
${chip_name}可通过npu-smi info查看：
![img_1.png](img_1.png)

## 4 数据集预处理

- **[数据集获取](#41-数据集获取)**  

- **[数据集预处理](#42-数据集预处理)**  

- **[生成数据集信息文件](#43-生成数据集信息文件)** 

### 4.1 数据集获取
[CityScape](https://www.cityscapes-dataset.com/downloads/) 下载其中gtFine_trainvaltest.zip和leftImg8bit_trainvaltest.zip两个压缩包，并解压。

上传数据集到服务器任意目录并解压（如：/opt/npu）。


### 4.2 数据集预处理
数据预处理将原始数据集转换为模型输入二进制格式。通过缩放、均值方差手段归一化，输出为二进制文件。
执行intrada_preprocess.py脚本，保存图像数据到bin文件。
```
python3.7 intrada_preprocess.py /opt/npu/cityscapes/ ./pre_dataset_bin
```
参数说明：
第一个参数为验证集图片路径，第二个参数为预处理后的文件保存路径。运行后生成pre_dataset_bin文件夹包含了预处理后的图片二进制文件。
### 4.3 生成数据集info文件
使用benchmark推理需要输入图片数据集的info文件，用于获取数据集。使用gen_dataset_info.py脚本，输入已经获得的图片文件，输出生成图片数据集的info文件。运行get_info.py脚本。
```
python3.7 gen_dataset_info.py bin ./pre_dataset_bin ./deeplabv2_pre_bin_512_1024.info 512 1024
```
第一个参数为生成的数据集文件格式，第二个参数为预处理后的数据文件的相对路径，第三个参数为生成的数据集文件保存的路径，第4，5个参数为图片的长宽。运行成功后，在当前目录中生成deeplabv2_pre_bin_512_1024.info。

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.1 推理benchmark工具用户指南 01
### 5.2 离线推理
1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理
增加执行权限
```
chmod u+x benchmark.x86_64
```
执行推理
```
./benchmark.x86_64  -model_type=vision -device_id=1 -batch_size=1 -om_path=./intraDA_deeplabv2_bs1.om -input_text_path=./deeplabv2_pre_bin_512_1024.info -input_width=1024 -input_height=512 -output_binary=True -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_devicex，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理精度统计](#61-离线推理精度)**

### 6.1 离线推理精度统计

后处理统计精度

调用postprocess_deepmar_pytorch.py脚本与数据集标签label比对，可以获得可以获得mIoU数据。
```
python3.7 -u intrada_postprocess.py /opt/npu/cityscapes ./result/dumpOutput_device0 ./out
```
第一个参数为生成推理结果所在路径，第二个参数为标签数据，第三个参数为生成结果文件。

如未显示eval start 可在当前目录新建out目录 命令如下
```
mkdir out
```
比较精度：
```
310：  mIoU 47.008
310P： mIoU 47.009
```
经过对bs1与bs8的om测试，本模型310的精度与310P的精度相差不超过1%，精度数据均如上


## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device。为快速获取性能数据，也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准。这里给出两种方式，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs8的性能数据为准，对于使用benchmark工具测试的batch4，16，32，64的性能数据在README.md中如下作记录即可。  
下面展示不同设备下的最优batchsize的性能情况

310： 
bs4 28.792(单卡吞吐量=Interface throughputRate * 4)

310P：
bs1 46.2538(单卡吞吐率)

GPU：
bs32 28.654

310P/310 = 1.606

310P/GPU = 1.614
性能达标

**性能优化：**  

>没有遇到性能不达标的问题，故不需要进行性能优化



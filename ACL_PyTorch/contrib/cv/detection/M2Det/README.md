# M2Det ONNX模型端到端推理指导

-   [1 模型概述](#1-模型概述)
	-   [1.1 论文地址](#11-论文地址)
	-   [1.2 代码地址](#12-代码地址)
-   [2 环境说明](#2-环境说明)
	-   [2.1 深度学习框架](#21-深度学习框架)
	-   [2.2 python第三方库](#22-python第三方库)
-   [3 模型转换](#3-模型转换)
	-   [3.1 pth转onnx模型](#31-pth转onnx模型)
	-   [3.2 onnx转om模型](#32-onnx转om模型)
-   [4 数据集预处理](#4-数据集预处理)
	-   [4.1 数据集获取](#41-数据集获取)
	-   [4.2 数据集预处理](#42-数据集预处理)
	-   [4.3 生成数据集信息文件](#43-生成数据集信息文件)
-   [5 离线推理](#5-离线推理)
	-   [5.1 获取ais_infer工具](#51-获取ais_infer工具)
	-   [5.2 离线推理](#52-离线推理)
-   [6 精度对比](#6-精度对比)
	-   [6.1 离线推理Acc精度统计](#61-离线推理Acc精度统计)
	-   [6.2 精度对比](#62-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[[M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network](https://arxiv.org/abs/1811.04533)] 

### 1.2 代码地址
[url=https://github.com/qijiezhao/M2Det](https://github.com/VDIGPKU/M2Det)  
branch:master  
commit_id:de4a6241bf22f7e7f46cb5cb1eb95615fd0a5e12

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.1.RC1
pytorch == 1.8.1
torchvision == 0.9.1
onnx == 1.12.0
```

### 2.2 python第三方库

```
onnx==1.12.0
Torch==1.8.1
TorchVision==0.9.1
numpy==1.21.6
absl-py==0.13.0
Cython==0.29.24
opencv-python==4.5.3.56
setuptools==41.2.0
matplotlib==2.2.5
addict==2.4.0
alabaster==0.7.12
Antlr4-python3-runtime==4.8
appdirs==1.4.4
asnlcryto==1.4.0
astroid==2.7.3
astropy==4.3.1

```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1. 准备pth权重文件  
从源码包中获取m2det512_vgg.pth、vgg16_reducedfc.pth文件，放在目录M2Det/weights下。

下载路径：
https://www.hiascend.com/zh/software/modelzoo/models/detail/1/1d97da1320994a16b0b8921e58000657

2. 下载PAMTRI源码并安装

```shell
git clone https://github.com/qijiezhao/M2Det 
cd M2Det
git reset --hard de4a6241bf22f7e7f46cb5cb1eb95615fd0a5e12
patch -p1 < ../M2Det.patch
sh make.sh
mkdir weights
mkdir logs
mkdir eval
cd ..
mkdir result

```

 **说明：**  
> 安装所需的依赖说明请参考M2Det/requirements.txt
>


3. 准备COCO2014 5000张图片的验证集，数据集获取参见本文第四章第一节 

4.  运行如下命令，生成M2Det.onnx模型文件
   使用m2det512_vgg.pth导出onnx文件，注意需要有预训练权重weights/vgg16_reducedfc.pth文件。
   运行“M2Det_pth2onnx.py”脚本：
```
python3.7 M2Det_pth2onnx.py -c=M2Det/configs/m2det512_vgg.py -pth=M2Det/weights/m2det512_vgg.pth -onnx=m2det512.onnx
```

   -c：配置文件。

   -pth：权重文件。

   -onnx：输出文件名称。

   获得“m2det512.onnx”文件。

### 3.2 onnx转om模型

1. 设置环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 使用atc将onnx模型
${chip_name}可通过npu-smi info指令查看，例：310P3
![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

执行ATC命令：
```shell
atc --framework=5 --model=m2det512.onnx --input_format=NCHW --input_shape="image:1,3,512,512" --output=m2det512_bs1 --log=debug --soc_version=Ascend${chip_name} --out-nodes="Softmax_1234:0;Reshape_1231:0"
```

参数说明：
--model：为ONNX模型文件。

--framework：5代表ONNX模型。

--input_format：输入数据的格式。

--input_shape：输入数据的shape。

--output：输出模型文件名称

--log：日志级别。

--soc_version：处理器型号。

--out-nodes：参数为指定输出节点，当选择的torch版本不同是可能会改变算子序号，如果torch不同请查看对应onnx文件算子进行相应的修改。

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
本模型支持COCO2014 5000张图片的验证集。请用户需自行获取val2014数据集，上传数据集到服务器根目录并解压（如：/root/data/coco/images）。本模型将使用到coco val2014验证集及instances_minival2014.json。

### 4.2 数据集预处理
数据预处理将原始数据集转换为模型输入的数据。

执行M2Det_preprocess.py脚本。

```shell
python3.7 M2Det_preprocess.py --config=configs/m2det512_vgg.py --save_folder=pre_dataset --COCO_imgs=coco_imgs_path --COCO_anns=coco_anns_path
```
--config：模型配置文件。

--save_folder：预处理后的数据文件保存路径。

--COCO_imgs：数据集images存放路径。

--COCO_anns：数据集annotations存放路径。

### 4.3 生成数据集信息文件
使用ais_infer推理需要输入图片数据集的info文件，用于获取数据集。使用gen_dataset_info.py脚本，输入已经获得的图片文件，输出生成图片数据集的info文件。运行gen_dataset_info.py脚本。

生成BIN文件输入info文件

```shell
python3.7 gen_dataset_info.py bin pre_dataset coco_prep_bin.info 512 512
```
“bin”：生成的数据集文件格式。

“pre_dataset”：预处理后的数据文件的**相对路径**。

“coco_prep_bin.info”：生成的数据集文件保存的路径。

“512 512”：图片宽高。

运行成功后，在当前目录中生成coco_prep_bin.info。

生成coco_images.info文件

```shell
python3.7 gen_dataset_info.py jpg ${coco_imgs_path}/val2014 coco_images.info
```
“jpg”：生成的数据集文件格式。

“${coco_imgs_path}/val2014”：验证集路径。

“coco_images.info”：输出文件名称。

## 5 离线推理

-   **[获取ais_infer工具](#51-获取ais_infer工具)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 获取ais_infer工具

[获取ais_infer工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)
将工具编译后的压缩包放置在当前目录；

解压工具包，安装工具压缩包中的whl文件；

pip3 install aclruntime-0.01-cp37-cp37m-linux_xxx.whl

### 5.2 离线推理
1.设置环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理
执行推理
```shell
python3.7  ais_infer.py --model /home/zzl/M2Det/m2det_bs1.om  --input /home/zzl/M2Det/pre_dataset/ --output /home/zzl/M2Det/result
```
参数说明：

--model：输入的模型。

--input：数据集预处理后的路径。

--output：推理结果输出路径。

## 6 精度对比

-   **[离线推理IoU精度](#61-离线推理IoU精度)**  
-   **[开源IoU精度](#62-开源IoU精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理Acc精度统计

后处理统计Acc精度

调用M2Det_postprocess.py脚本获得Accuracy数据，结果保存在result/detection-results_1/COCO/detections_minival2014_results.json中。
```python
python3.7 M2Det_postprocess.py --bin_data_path=result/2022_07_22-09_31_54/ --test_annotation=coco_images.info --det_results_path==result/detection-results_0_bs1 --net_out_num=2 --prob_thres=0.1 --COCO_imgs=/opt/npu/dataval2014/images --COCO_anns=/opt/npu/dataval2014/annotations --is_ais_infer
```
参数说明：

--bin_data_path：推理结果所在路径（根据具体的推理结果进行修改）。

--test_annotation：验证集数据信息。

--det_results_path=：生成结果文件。

--net_out_num：网络输出类型个数（此处为score,box，2个）。

--prob_thres：参数阈值。

--COCO_imgs：coco数据集images路径。

--COCO_anns：coco数据集annotations路径。

--is_ais_infer：使用ais_infer推理工具。

执行完后得到310P上的精度：
```
｜batchsize｜  Acc   ｜
| ------ | -------- |
｜    1    ｜ IoU=[0.50,0.95]:37.8% ｜
｜    4    ｜ IoU=[0.50,0.95]:37.8% ｜
｜    8    ｜ IoU=[0.50,0.95]:37.8% ｜
｜    16   ｜ IoU=[0.50,0.95]:37.8% ｜
｜    32   ｜ IoU=[0.50,0.95]:37.8% ｜
```

### 6.2 精度对比
将得到的om离线模型推理Acc精度与该模型github代码仓上公布的精度对比，精度相同，故精度达标。

 ** 精度调试**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
离线推理的Interface throughputRate即为吞吐量，对于310，需要乘以4，310P只有一颗芯片，FPS为该值本身。

|        | 310      | 310P    | T4      | 310P/310    | 310P/T4     |
| ------ | -------- | ------- | ------- | ----------- | ----------- |
| bs1    | 34.212  | 37.817 | 50.658 | 1.105372384 | 0.746515851 |
| bs4    | 34.086 | 54.707 | 60.997 | 1.604969782 | 0.896880174 |
| bs8    | 34.142  | 46.461  | 52.89 | 1.36081659 | 0.878445831 |
| bs16   | 34.32  | 47.113 | 62.186 | 1.37275641 | 0.757614254  |
| bs32   | 34.303  | 46.84 | 64.602 | 1.365478238 | 0.725054952 |
| 最优bs | 34.32 | 54.707 | 64.602 | 1.594026807 | 0.846831367 |

310P的最优batchsize为：bs4。
最优batch：310P大于310的1.2；310P小于T4的1.6

**性能优化：**  

>具体详见PyTorch离线推理-M2Det模型测试报告
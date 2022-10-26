# FSAF Onnx模型端到端推理指导

- [FSAF Onnx模型端到端推理指导](#fsaf-onnx模型端到端推理指导)
	- [1 模型概述](#1-模型概述)
		- [1.1 论文地址](#11-论文地址)
		- [1.2 代码地址](#12-代码地址)
	- [2 环境说明](#2-环境说明)
		- [2.1 深度学习框架](#21-深度学习框架)
		- [2.2 python第三方库](#22-python第三方库)
	- [3 模型转换](#3-模型转换)
		- [3.1 pth转onnx模型](#31-pth转onnx模型)
		- [3.2 onnx转om模型](#32-onnx转om模型)
	- [4 数据集预处理](#4-数据集预处理)
		- [4.1 数据集获取](#41-数据集获取)
		- [4.2 数据集预处理](#42-数据集预处理)
	- [5 离线推理](#5-离线推理)
		- [5.1 benchmark工具概述](#51-benchmark工具概述)
		- [5.2 离线推理](#52-离线推理)
	- [6 精度对比](#6-精度对比)
		- [6.1 离线推理精度统计](#61-离线推理精度统计)
		- [6.2 开源精度](#62-开源精度)
		- [6.3 精度对比](#63-精度对比)
	- [7 性能对比](#7-性能对比)
		- [7.1 npu性能数据](#71-npu性能数据)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[FSAF论文](https://arxiv.org/pdf/1903.00621.pdf) FSAF 是 CVPR2019发表的一种无锚定方法。实际上它等价于基于锚的方法，在每个 FPN 级别的每个特征映射位置只有一个锚。我们就是这样实施的。只有没有锚的分支被释放，因为它与当前框架的兼容性更好，计算预算更少。

### 1.2 代码地址
[FSAF代码](https://github.com/open-mmlab/mmdetection)  
branch:master  
commit_id:604bfe9618533949c74002a4e54f972e57ad0a7a
  
备注：commit_id是指基于该次提交时的模型代码做推理，通常选择稳定版本的最后一次提交，或代码仓最新的一次提交  

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN == 5.1.RC1
onnx == 1.9.0
pytorch == 1.8.0
torchvision == 0.9.0
```

### 2.2 python第三方库

```
numpy == 1.21.2
pillow == 8.2.0
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.下载pth权重文件  
FSAF预训练pth权重文件,[下载页面链接](https://github.com/open-mmlab/mmdetection/tree/master/configs/fsaf)
此页面下载box AP=37.4的权重pth文件

2.获取，修改与安装开源模型代码。
```
git clone https://github.com/open-mmlab/mmcv -b master 
cd mmcv
git reset --hard 04daea425bcb0a104d8b4acbbc16bd31304cf168
MMCV_WITH_OPS=1 pip3.7 install -e .
cd ..
git clone https://github.com/open-mmlab/mmdetection -b master
cd mmdetection
git reset --hard 604bfe9618533949c74002a4e54f972e57ad0a7a
patch -p1 < ../fsaf.diff
pip3.7 install -r requirements/build.txt
python3.7 setup.py develop
cd ..
```

3.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 ./mmdetection/tools/deployment/pytorch2onnx.py ./mmdetection/configs/fsaf/fsaf_r50_fpn_1x_coco.py ./fsaf_r50_fpn_1x_coco-94ccc51f.pth --output-file fsaf.onnx --input-img ./mmdetection/demo/demo.jpg --shape 800 1216
```
--output-file:输出的onnx文件位置。
--input-img:输入图片位置。
--shape:输入图片的高宽

### 3.2 onnx转om模型

1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

 **说明：**  
>环境变量影响atc命令是否成功，在测试时如报错需验证环境变量的正确性

2.使用atc将onnx模型转换为om模型文件
```
atc --framework=5 --model=./fsaf.onnx --output=fsaf_bs1 --input_format=NCHW --input_shape="input:1,3,800,1216" --log=debug --soc_version=Ascend${chip_name} --out_nodes="dets;labels"
```
${chip_name}可通过`npu-smi info`指令查看，例：310P3
![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)
    
    --input_shape：输入数据的shape。
    --output：输出的OM模型。  
    --log：日志级别。  
    --soc_version：处理器型号，Ascend310或Ascend310P。  
    --soc_version：处理器型号。
    --out_nodes：模型输出节点。

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
本模型支持coco的验证集。请用户需自行获取coco数据集，上传数据集到服务器任意目录并解压（如：/root/datasets/）。本模型将使用到val2017验证集及annotations中的instances_val2017.json

### 4.2 数据集预处理
1.进行数据预处理， 执行Fsaf_preprocess.py脚本,生成bin文件。
```
python3.7 Fsaf_preprocess.py --image_src_path=/root/datasets/coco/val2017 --bin_file_path=val2017_bin --model_input_height=800 --model_input_width=1216
```
--image_src_path:原始数据集路径
--bin_file_path:转化后的bin文件路径
--model_input_height:输入图片的高
--model_input_width:输入图片的宽

2.执行预处理脚本，生成数据集info文件
+ bin文件输入
```
python3.7 get_info.py bin val2017_bin fsaf.info 1216 800
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息。运行成功后，在当前目录中生成fsaf.info。
+ 生成JPEG图片输入info文件。
```
python3.7 get_info.py jpg /root/datasets/coco/val2017 fsaf_jpeg.info
```
第一个参数为生成的数据集文件格式，第二个参数为预处理后的数据文件的相对路径，第三个参数为生成的数据集文件保存的路径。运行成功后，在当前目录中生成fsaf_jpeg.info。

## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310/310P上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.1 推理benchmark工具用户指南 01
### 5.2 离线推理
1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理

```
chmod u+x benchmark.x86_64
./benchmark.x86_64 -model_type=vision -om_path=fsaf_bs1.om -device_id=0 -batch_size=1 -input_text_path=fsaf.info -input_width=1216 -input_height=800 -useDvpp=false -output_binary=true
```
推理后的输出默认在当前目录result下。

## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计

后处理统计map精度


```shell
python3.7 Fsaf_postprocess.py --bin_data_path=./result/dumpOutput_device0/ --test_annotation=fsaf_jpeg.info --det_results_path=./ret_npuinfer/ --net_out_num=3 --net_input_height=800 --net_input_width=1216 --ifShowDetObj --annotations_path=/root/datasets/coco/annotations/instances_val2017.json
```
--bin_data_path为benchmark推理结果，

--val2017_path为原始图片路径，

--test_annotation为图片信息文件，

--net_out_num为网络输出个数，

--net_input_height为网络高，

--net_input_width为网络高 
执行完后会打印出精度：

```
Evaluating bbox...
Loading and preparing results...
DONE (t=1.98s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=52.96s).
Accumulating evaluation results...
DONE (t=18.60s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.371
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.565
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.395
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.200
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.499
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.544
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.544
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.544
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.336
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.588
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.705

```

### 6.2 开源精度
[官网精度](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)

参考[mmdetection框架](https://mmdetection.readthedocs.io/en/stable/get_started.html)，安装依赖PyTorch(GPU版本)与设置环境变量，在GPU上执行推理，测得GPU精度如下：
```shell
git clone https://github.com/open-mmlab/mmcv
cd mmcv
MMCV_WITH_OPS=1 pip3.7 install -e .
cd ..
git clone https://github.com/open-mmlab/mmdetection
cd mmdetection
pip3.7 install -r requirements/build.txt
python3.7 setup.py develop
下载pth权重到当前目录下， https://github.com/open-mmlab/mmdetection/tree/master/configs/fsaf页面下载box AP=37.4的权重pth文件
在当前目录按结构构造数据集：data/coco目录下有annotations与val2017，annotations目录存放coco数据集的instances_val2017.json，val2017目录存放coco数据集的5000张验证图片。
python3.7 ./mmdetection/tools/test.py ./mmdetection/configs/fsaf/fsaf_r50_fpn_1x_coco.py ./fsaf_r50_fpn_1x_coco-94ccc51f.pth --eval bbox  --cfg-options data.test.samples_per_gpu=1
```
--cfg-options 给出batch_size的大小

获得精度为：

```
Evaluating bbox...
Loading and preparing results...
DONE (t=1.32s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=53.11s).
Accumulating evaluation results...
DONE (t=16.91s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.374
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.568
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.398
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.204
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.411
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.488
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.348
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.590
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.695
```

### 6.3 精度对比

310上om推理box map精度为0.371，官方开源pth推理box map精度为0.374，精度下降在1个点之内，因此可视为精度达标，310P上fp16精度0.370, 可视为精度达标

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device。为快速获取性能数据，也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准。这里给出两种方式，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  
1.benchmark工具在整个数据集上推理获得性能数据  
|      | 310     | 310P    | 310P_aoe | T4   | 310P_aoe/310 | 310P/T4     |
|------|---------|---------|----------|------|--------------|-------------|
| bs1  | 47.2984 | 49.4762 | 57.1742  | 9.2  | 1.208797761  | 6.214586957 |
| bs4  | 54.166  | 49.4762 | 72.5245  | 7    | 1.338930325  | 10.36064286 |
| bs8  | 54.7236 | 51.9316 | 67.9739  | 6.1  | 1.242131366  | 11.1432623  |
| bs16 | 54.1664 | 53.7677 | 60.6753  | 7.2  | 1.120164899  | 8.427125    |
| bs32 | 52.112  | 55.449  | 70.1504  | 显存不足 | 1.346146761  | \           |
| bs64 | 显存不足    | 54.2045 | 64.1296  | 显存不足 | \            | \           |
| 最优batch | 54.7236 | 58.1587 | 72.5245 | 9.2 | 1.325287445 | 7.883097826 |

最优batch：310P_aoe大于310的1.2倍；310P大于T4的1.6倍，性能达标。

 **性能优化：**  
> Fsaf通过act工具导出的om模型在多个bs下310P性能没有达到310的1.2倍，通过aoe进行算子调优：


`aoe --framework=5 --model=./fsaf.onnx --job_type=2 --output=aoe_fsaf_bs1 --input_format=NCHW --input_shape="input:1,3,800,1216"`


调整"input:1,3,800,1216"的bs维度的数字实现多bs的经过优化后的om模型导出，通过算子调优，310P下最优batch下性能达到310的1.2倍
>

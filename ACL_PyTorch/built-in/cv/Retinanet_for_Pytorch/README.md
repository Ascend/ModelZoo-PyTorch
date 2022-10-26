# 基于detectron2训练的retinanet Onnx模型端到端推理指导
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
	-   [5.1 benchmark工具概述](#51-benchmark工具概述)
	-   [5.2 离线推理](#52-离线推理)
-   [6 精度对比](#6-精度对比)
	-   [6.1 离线推理精度统计](#61-离线推理精度统计)
	-   [6.2 开源精度](#62-开源精度)
	-   [6.3 精度对比](#63-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)

## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[Retinanet论文](https://arxiv.org/abs/1708.02002)
论文提出了一个简单、灵活、通用的损失函数Focal loss，用于解决单阶段目标检测网络检测精度不如双阶段网络的问题。这个损失函数是针对了难易样本训练和正负样本失衡所提出的，使得单阶段网络在运行快速的情况下，获得与双阶段检测网络相当的检测精度。此外作者还提出了一个Retinanet用于检验网络的有效性，其中使用Resnet和FPN用于提取多尺度的特征。


### 1.2 代码地址

[cpu,gpu版detectron2框架retinanet代码](https://github.com/facebookresearch/detectron2)
commit_id:60fd4885d7cfd52d4267d1da9ebb6b2b9a3fc937

[离线推理版detectron2框架retinanet代码](https://github.com/facebookresearch/detectron2)
commit_id:60fd4885d7cfd52d4267d1da9ebb6b2b9a3fc937

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
pytorch == 1.8.0
torchvision == 0.9.0
onnx == 1.8.1
```

**注意：** 

>   转onnx的环境上pytorch需要安装1.8.0版本
>

### 2.2 python第三方库

```
numpy == 1.18.5
opencv-python == 4.2.0.34
sclblonnx == 0.1.9
om_gener
```

**说明：** 
>   X86架构：opencv,pytorch,torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip install 包名 安装
>
>   Arm架构：opencv,pytorch,torchvision和onnx可以通过源码编译安装，其它可以通过pip install 包名 安装
>
>   另外需要从https://gitee.com/liurf_hw/om_gener 安装om_gener

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

detectron2暂支持pytorch1.8导出pytorch框架的onnx，npu权重可以使用开源的detectron2加载，因此基于pytorch1.8与开源detectron2导出含npu权重的onnx。atc暂不支持动态shape小算子，可以使用大颗粒算子替换这些小算子规避，这些小算子可以在转onnx时的verbose打印中找到其对应的python代码，从而根据功能用大颗粒算子替换，onnx能推导出变量正确的shape与算子属性正确即可，变量实际的数值无关紧要，因此这些大算子函数的功能实现无关紧要，因包含自定义算子需要去掉对onnx模型的校验。


### 3.1 pth转onnx模型

1.获取pth权重文件  
[retinanet基于detectron2预训练的权重文件，下载路径：

github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

下载Retinanet项 Name R50 3x 38.7精度的model

md5sum：5bd44e2eaaabb0e1877c1c91f37ce513
2.下载detectron2源码并安装

```shell
git clone https://github.com/facebookresearch/detectron2
git reset 60fd4885d7cfd52d4267d1da9ebb6b2b9a3fc937 --hard
python -m pip install -e detectron2
```

 **说明：**  
> 安装所需的依赖说明请参考detectron2/INSTALL.md
>
> 重装pytorch后需要rm -rf detectron2/build/ **/*.so再重装detectron2

3.detectron2代码迁移，参见retinanet_detectron2.diff：

 **修改依据：**  
> 1.由于om不支持Non_Zero算子，而是用掩模筛选数据在输出onnx时会引入Non_Zero操作，因此不能使用掩模。所以就用直接输出index的torch.topk代替。修改参见retinanet_detectron2.diff  
> 2.由于onnx不支持torchvision的nms函数，因此使用自定义的om支持的nms算子
> 3.数据预处理步骤和后处理模块应当从网络中移除  
> 4.slice跑在aicpu有错误，所以改为dx = denorm_deltas[:, :, 0:1:].view(-1, 80) / wx，使其运行在aicore上  
> 5.atc转换时根据日志中报错的算子在转onnx时的verbose打印中找到其对应的python代码，然后找到规避方法解决，此外opt应该设为11，具体修改参见retinanet_detectron2.diff  
> 6.其它地方的修改原因参见精度调试与性能优化  


通过打补丁的方式修改detectron2：
```shell
cd detectron2
patch -p1 < ../retinanet_detectron2.diff
cd ..
```
4.修改pytorch代码去除导出onnx时进行检查  
将/root/anconda3/lib/python3.7/site-packages/torch/onnx/utils.py文件的_check_onnx_proto(proto)改为pass

5.准备coco2017验证集，数据集获取参见本文第四章第一节  
在当前目录按结构构造数据集：datasets/coco目录下有annotations与val2017，annotations目录存放coco数据集的instances_val2017.json，val2017目录存放coco数据集的5000张验证图片。  
或者修改detectron2/detectron2/data/datasets/builtin.py为_root = os.getenv("DETECTRON2_DATASETS", "/root/datasets/")指定coco数据集所在的目录/root/datasets/。

6.运行如下命令，在output目录生成model.onnx
```shell
python3.7 detectron2/tools/deploy/export_model.py --config-file detectron2/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml --output ./output --export-method tracing --format onnx MODEL.WEIGHTS model_final.pkl MODEL.DEVICE cpu

mv output/model.onnx retinanet.onnx
```

### 3.2 onnx转om模型

1. 设置环境变量

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2. 量化工具安装

   参考链接：support.huawei.com/enterprise/zh/doc/EDOC1100219269/805ec438

3. 修改并量化onnx

   onnx中部分cast算子会走到aicpu上，我们修改数据类型使其走到aicore上，并且由于onnx中有自定义算子，我们先拆分onnx，量化部分onnx，再merge两个onnx。

   1. 首先创建文件夹int8data，用于存放量化矫正数据，矫正数据为4.2数据集预处理中16张生成的bin文件，最好将其合并成一个bin文件（310上内存可能会出现不够的现象），在Linux环境下合成命令：cat 000000397133.bin 000000037777.bin 000000252219.bin 000000087038.bin 000000174482.bin 000000403385.bin 000000006818.bin 000000480985.bin 000000458054.bin 000000331352.bin 000000296649.bin 000000386912.bin 000000502136.bin 000000491497.bin 000000184791.bin 000000348881.bin 000000289393.bin > quant.bin

   2. mv quant.bin ./int8data

   3. 运行 python3.7.5 adaptretinanet.py

      若遇到类似onnx校验不通过的情况，则将对应报错的check代码注释掉即可

   4. 生成的retinanet_revise.onnx和retinanet_int8_revise.onnx即为用于转om离线模型的onnx文件

3. 使用atc将onnx模型（包括量化模型和非量化模型）转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)，需要指定输出节点以去除无用输出，使用netron开源可视化工具查看具体的输出节点名，如使用的设备是710，则将--soc_version设置为Ascend710：

```shell
atc --model=retinanet_revise.onnx --framework=5 --output=retinanet_detectron2_npu --input_format=NCHW --input_shape="input0:1,3,1344,1344" --out_nodes="Cast_1224:0;Reshape_1218:0;Gather_1226:0" --log=info --soc_version=Ascend310
```

量化模型转om(注意输出节点名字已改变，使用netron打开后手动修改)

```
atc --model=retinanet_int8_revise.onnx --framework=5 --output=retinanet_detectron2_npu --input_format=NCHW --input_shape="input0:1,3,1344,1344" --out_nodes="Cast_1229_sg2:0;Reshape_1223_sg2:0;Gather_1231_sg2:0" --log=info --soc_version=Ascend310
```



## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用coco2017的5千张验证集进行测试，图片与标签分别存放在/opt/npu/dataset/coco/val2017/与/opt/npu/dataset/coco/annotations/instances_val2017.json。

### 4.2 数据集预处理
1.预处理脚本retinanet_pth_preprocess_detectron2.py


2.执行预处理脚本，生成数据集预处理后的bin文件
```shell
python3.7 retinanet_pth_preprocess_detectron2.py --image_src_path=/root/datasets/coco/val2017 --bin_file_path=val2017_bin --model_input_height=1344 --model_input_width=1344
```

### 4.3 生成预处理数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```shell
python3.7 get_info.py bin val2017_bin retinanet.info 1344 1344
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息


## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)

### 5.2 离线推理
1.设置环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理
```shell
./benchmark.x86_64 -model_type=vision -om_path=retinanet_detectron2_npu.om -device_id=0 -batch_size=1 -input_text_path=retinanet.info -input_width=1344 -input_height=1344 -useDvpp=false -output_binary=true
```
输出结果默认保存在当前目录result/dumpOutput_device0，模型有四个输出，每个输入对应的输出对应四个_x.bin文件
```
输出       shape                 数据类型    数据含义
output1    100 * 4               FP32       boxes
output2    100 * 1               Int32       labels
output2    100 * 1               FP32       scores
```

## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计

后处理统计map精度


调用retinanet_pth_postprocess_detectron2.py评测map精度：
```shell
python3.7 get_info.py jpg /opt/npu/dataset/coco/val2017 origin_image.info

python3.7 retinanet_pth_postprocess_detectron2.py --bin_data_path=./result/dumpOutput_device0/ --val2017_path=${datasets_path}/coco --test_annotation=origin_image.info --det_results_path=./ret_npuinfer/ --net_out_num=3 --net_input_height=1344 --net_input_width=1344 
```
--bin_data_path为benchmark推理结果，

--val2017_path为原始图片路径，

--test_annotation为图片信息文件，

--net_out_num为网络输出个数，

--net_input_height为网络高，

--net_input_width为网络高 
执行完后会打印出精度：

```
Loading and preparing results...
DONE (t=4.64s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=75.71s).
Accumulating evaluation results...
DONE (t=16.46s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.384
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.576
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.412
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.227
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.326
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.522
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.367
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.582
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.700

```
 **精度调试：**  
> 1.当NMS前选取各类的TOPK过小时，会使精度下降一个点，调为各类保留200较为合适
> 2.因gather算子处理-1会导致每张图的第一个score为0，故maskrcnn_detectron2.diff中已将dets[:, -1]改为dets[:, 4]  
> 3.单张图调试  
>
> ```
> demo.py分数改为0.05，defaults.py MIN_SIZE_TEST与MAX_SIZE_TEST改为1344：
> python3.7 demo.py --config-file ./detectron2/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml --input 000000252219_1344x1344.jpg --opts MODEL.WEIGHTS ./model_final.pkl MODEL.DEVICE cpu
> 说明：
> 比较pth的retinanet与om的retinanet输出前提是detectron2/config/defaults.py的_C.INPUT.MIN_SIZE_TEST与_C.INPUT.MAX_SIZE_TEST要改为1344，并且注意因为000000252219_1344x1344.jpg 是等比例缩放四边加pad的处理结果，因此pth推理时等价于先进行了pad然后再进行标准化的，因此图片tensor边缘是负均值。开始误认为预处理与mmdetection相同因此SIZE_TEST的值与000000252219_1344x1344.jpg缩放是按上述方式处理的，经此与后面的调试步骤发现预处理与mmdetection不同。om算子输出与开源pth推理时变量的打印值对比，找到输出不对的算子，发现前处理均值方差不同于mmdetection框架，且是BGR序。
> 发现做topk筛选时，发现每个类保留200个性能和精度最佳。
> ```
> 4.精度调试  
>
> ```
> 对开源代码预处理与参数修改，使得cpu,gpu版的pth公开推理精度，参见pth的diff文件与执行精度测评的命令。
> 说明：
> 1.GPU固定1344,1344的前处理方式（缩放加pad）
> FIX_SHAPE->./detectron2/data/dataset_mapper.py->ResizeShortestEdge，最短边800最大1333。
> ```
### 6.2 开源精度
[官网精度](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)

参考[detectron2框架的retinanet](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)，安装依赖PyTorch(GPU版本)与设置环境变量，在GPU上执行推理，测得GPU精度如下：
```shell
git clone https://github.com/facebookresearch/detectron2
python3.7 -m pip install -e detectron2
cd ./detectron2/tools
python3.7 train_net.py --eval-only --config-file ./detectron2/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml MODEL.WEIGHTS ./model_final.pkl
```
配置文件与权重文件分别为retinanet_R_50_FPN_3x.yaml与model_final.pkl，删除retinanet_R_50_FPN_3x.yaml的SOLVER和DATALOADER配置，root/datasets/coco下面放置coco2017验证集图片与标签（参考本文第三章第一节步骤五）
获得精度为

```
AP,AP50,AP75,APs,APm,APl
38.679,57.998,41.489,23.348,42.303, 50.316
```


### 6.3 精度对比

310上om推理box map精度为0.384，官方开源pth推理box map精度为0.387，精度下降在1个点之内，因此可视为精度达标，710上fp16精度0.383, int8 0.382，可视为精度达标

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
batch1的性能：

5.2步骤中，离线推理的Interface throughputRate即为吞吐量，对于310，需要乘以4，710只有一颗芯片，FPS为该值本身

retinanet detectron2不支持多batch

 **性能优化：**  
> 查看profiling导出的op_statistic_0_1.csv算子总体耗时统计发现gather算子耗时最多，然后查看profiling导出的task_time_0_1.csv找到具体哪些gather算子耗时最多，通过导出onnx的verbose打印找到具体算子对应的代码，因gather算子计算最后一个轴会很耗时，因此通过转置后计算0轴规避，比如retinanet_detectron2.diff文件中的如下修改：
> ```
>    boxes_prof = boxes.permute(1, 0)
>    widths = boxes_prof[2, :] - boxes_prof[0, :]
> ```
>

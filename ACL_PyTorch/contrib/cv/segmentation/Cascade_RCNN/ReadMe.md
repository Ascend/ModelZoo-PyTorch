# 基于detectron2的Cascade-Mask-Rcnn Onnx模型端到端推理指导
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
[cascadercnn论文](https://arxiv.org/abs/1712.00726)  
在目标检测中，需要一个交并比(IOU)阈值来定义物体正负标签。使用低IOU阈值(例如0.5)训练的目标检测器通常会产生噪声检测。然而，随着IOU阈值的增加，检测性能趋于下降。影响这一结果的主要因素有两个：1)训练过程中由于正样本呈指数级消失而导致的过度拟合；2)检测器为最优的IOU与输入假设的IOU之间的推断时间不匹配。针对这些问题，提出了一种多级目标检测体系结构-级联R-CNN.它由一系列随着IOU阈值的提高而训练的探测器组成，以便对接近的假阳性有更多的选择性。探测器是分阶段训练的，利用观察到的探测器输出是训练下一个高质量探测器的良好分布。逐步改进的假设的重采样保证了所有探测器都有一组等效尺寸的正的例子，从而减少了过拟合问题。同样的级联程序应用于推理，使假设与每个阶段的检测器质量之间能够更紧密地匹配。Cascade R-CNN的一个简单实现显示，在具有挑战性的COCO数据集上，它超过了所有的单模型对象检测器。实验还表明，Cascade R-CNN在检测器体系结构中具有广泛的适用性，独立于基线检测器强度获得了一致的增益。

### 1.2 代码地址
[cpu,gpu版detectron2框架cascadercnn代码](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)   
branch:master
commit_id:13afb035142734a309b20634dadbba0504d7eefe
## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
pytorch == 1.8.0
torchvision == 0.9.0
onnx == 1.8.0
```

**注意：** 
>   转onnx的环境上pytorch需要安装1.8.0版本
>

### 2.2 python第三方库

```
numpy == 1.18.5
opencv-python == 4.2.0.34
```

**说明：** 
>   X86架构：opencv,pytorch,torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：opencv,pytorch,torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

detectron2暂支持pytorch1.8导出pytorch框架的onnx，npu权重可以使用开源的detectron2加载，因此基于pytorch1.8与开源detectron2导出含npu权重的onnx。atc暂不支持动态shape小算子，可以使用大颗粒算子替换这些小算子规避，这些小算子可以在转onnx时的verbose打印中找到其对应的python代码，从而根据功能用大颗粒算子替换，onnx能推导出变量正确的shape与算子属性正确即可，变量实际的数值无关紧要，因此这些大算子函数的功能实现无关紧要，因包含自定义算子需要去掉对onnx模型的校验。

### 3.1 pkl转onnx模型

1.获取pkl权重文件  
[cascade-mask-rcnn基于detectron2预训练的权重文件](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md),下面简称cascadercnn  
文件md5sum:480dd8c7cfbe5aea4f159793080466ed  
2.下载detectron2源码并安装
```shell
git clone https://github.com/facebookresearch/detectron2
python3.7 -m pip install -e detectron2
```

 **说明：**  
> 安装所需的依赖说明请参考detectron2/INSTALL.md
>
> 重装pytorch后需要rm -rf detectron2/build/ **/*.so再重装detectron2

3.detectron2代码迁移，参见cascadercnn_detectron2.diff  
 **模型转换要点：**  
> 1.slice，topk算子问题导致pre_nms_topk未生效，atc转换报错，修改参见cascadercnn_detectron2.diff  
> 2.expand会引入where动态算子因此用expand_as替换  
> 3.slice跑在aicpu有错误，所以改为dx = denorm_deltas[:, :, 0:1:].view(-1, 80) / wx，使其运行在aicore上  
> 4.atc转换时根据日志中报错的算子在转onnx时的verbose打印中找到其对应的python代码，然后找到规避方法解决，具体修改参见cascadercnn_detectron2.diff  
> 5.转onnx文件时报算子_ScaleGradient的错，推理阶段不需要这个算子，故注释掉  
> 6.其它地方的修改原因参见精度调试与性能优化  


通过打补丁的方式修改detectron2：
```shell
cd detectron2
patch -p1 < ../cascadercnn_detectron2.diff
cd ..
```
4.修改pytorch代码去除导出onnx时进行检查  
将/usr/local/python3.7.5/lib/python3.7/site-packages/torch/onnx/utils.py文件的_check_onnx_proto(proto)改为pass

5.准备coco2017验证集，数据集获取参见本文第四章第一节  
在当前目录按结构构造数据集：datasets/coco目录下有annotations与val2017，annotations目录存放coco数据集的instances_val2017.json，val2017目录存放coco数据集的5000张验证图片。  
或者修改detectron2/detectron2/data/datasets/builtin.py为_root = os.getenv("DETECTRON2_DATASETS", "/root/datasets/")指定coco数据集所在的目录/opt/npu/dataset/。

6.运行如下命令，在output目录生成model.onnx
```shell
python3.7 detectron2/tools/deploy/export_model.py --config-file detectron2/configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml --output ./output --export-method tracing --format onnx MODEL.WEIGHTS cascadercnn.pkl MODEL.DEVICE cpu

mv output/model.onnx model_py1.8.onnx
```

### 3.2 onnx转om模型

1.设置环境变量
```shell
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)，需要指定输出节点以去除无用输出，使用netron开源可视化工具查看具体的输出节点名：
```shell
atc --model=model_py1.8.onnx --framework=5 --output=cascadercnn_detectron2_npu --input_format=NCHW --input_shape="0:1,3,1344,1344" --out_nodes="Cast_1853:0;Gather_1856:0;Reshape_1847:0;Slice_1886:0" --log=debug --soc_version=Ascend310
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[COCO官网](https://cocodataset.org/#download)的coco2017的5千张验证集进行测试，图片与标签分别存放在/root/datasets/coco/val2017/与/root/datasets/coco/annotations/instances_val2017.json。

### 4.2 数据集预处理
1.预处理脚本cascadercnn_pth_preprocess_detectron2.py
2.执行预处理脚本，生成数据集预处理后的bin文件
```shell
python3.7 cascadercnn_pth_preprocess_detectron2.py --image_src_path=/root/datasets/coco/val2017 --bin_file_path=val2017_bin --model_input_height=1344 --model_input_width=1344
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py
2.执行生成数据集信息脚本，生成数据集信息文件
```shell
python3.7 get_info.py bin val2017_bin cascadercnn.info 1344 1344
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
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/
```
2.执行离线推理
```shell
./benchmark.x86_64 -model_type=vision -om_path=cascadercnn_detectron2_npu.om -device_id=0 -batch_size=1 -input_text_path=cascadercnn.info -input_width=1344 -input_height=1344 -useDvpp=false -output_binary=true
```
输出结果默认保存在当前目录result/dumpOutput_device0，模型有四个输出，每个输入对应的输出对应四个_x.bin文件
```
输出       shape                 数据类型    数据含义
output1    100 * 4               FP32       boxes
output2    100 * 1               FP32       scores
output3    100 * 1               INT64      labels
output4    100 * 80 * 28 * 28    FP32       masks
```

## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计

后处理统计map精度，调用cascadercnn_pth_postprocess_detectron2.py评测map精度：
```shell
python3.7 get_info.py jpg /root/datasets/coco/val2017 cascadercnn_jpeg.info

python3.7 cascadercnn_pth_postprocess_detectron2.py --bin_data_path=./result/dumpOutput_device0/ --test_annotation=cascadercnn_jpeg.info --det_results_path=./ret_npuinfer/ --net_out_num=4 --net_input_height=1344 --net_input_width=1344 --ifShowDetObj
```
第一个参数为benchmark推理结果，第二个为原始图片信息文件，第三个为后处理输出结果，第四个为网络输出个数，第五六个为网络高宽，第七个为是否将box画在图上显示  
执行完后会打印出精度：
```
INFO:detectron2.data.datasets.coco:Loaded 5000 images in COCO format from /home/wxx/detectron2_npu/datasets/coco/annotations/instances_val2017.json
INFO:detectron2.evaluation.coco_evaluation:Preparing results for COCO format ...
INFO:detectron2.evaluation.coco_evaluation:Evaluating predictions with unofficial COCO API...
Loading and preparing results...
DONE (t=2.38s)
creating index...
index created!
INFO:detectron2.evaluation.fast_eval_api:Evaluate annotation type *bbox*
INFO:detectron2.evaluation.fast_eval_api:COCOeval_opt.evaluate() finished in 20.47 seconds.
INFO:detectron2.evaluation.fast_eval_api:Accumulating evaluation results...
INFO:detectron2.evaluation.fast_eval_api:COCOeval_opt.accumulate() finished in 2.40 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.439
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.619
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.477
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.259
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.472
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.352
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.552
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.579
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.376
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.615
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.735
INFO:detectron2.evaluation.coco_evaluation:Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 43.916 | 61.856 | 47.659 | 25.927 | 47.247 | 57.657 |
INFO:detectron2.evaluation.coco_evaluation:Per-category bbox AP:
| category      | AP     | category     | AP     | category       | AP     |
|:--------------|:-------|:-------------|:-------|:---------------|:-------|
| person        | 58.518 | bicycle      | 34.208 | car            | 47.891 |
| motorcycle    | 45.491 | airplane     | 68.934 | bus            | 67.638 |
| train         | 68.632 | truck        | 37.125 | boat           | 31.227 |
| traffic light | 28.846 | fire hydrant | 68.794 | stop sign      | 71.740 |
| parking meter | 47.046 | bench        | 26.830 | bird           | 38.628 |
| cat           | 71.501 | dog          | 63.997 | horse          | 59.504 |
| sheep         | 53.279 | cow          | 58.018 | elephant       | 65.251 |
| bear          | 70.799 | zebra        | 67.613 | giraffe        | 69.831 |
| backpack      | 17.793 | umbrella     | 41.941 | handbag        | 15.845 |
| tie           | 37.194 | suitcase     | 40.098 | frisbee        | 67.049 |
| skis          | 26.852 | snowboard    | 40.849 | sports ball    | 48.154 |
| kite          | 44.503 | baseball bat | 31.925 | baseball glove | 36.880 |
| skateboard    | 55.498 | surfboard    | 42.578 | tennis racket  | 48.609 |
| bottle        | 41.320 | wine glass   | 38.322 | cup            | 43.942 |
| fork          | 39.684 | knife        | 21.676 | spoon          | 19.082 |
| bowl          | 45.006 | banana       | 25.631 | apple          | 21.556 |
| sandwich      | 36.754 | orange       | 32.271 | broccoli       | 23.367 |
| carrot        | 22.343 | hot dog      | 37.058 | pizza          | 55.796 |
| donut         | 46.395 | cake         | 36.794 | chair          | 29.161 |
| couch         | 45.875 | potted plant | 26.646 | bed            | 45.699 |
| dining table  | 29.721 | toilet       | 63.905 | tv             | 59.488 |
| laptop        | 62.087 | mouse        | 66.021 | remote         | 33.904 |
| keyboard      | 54.539 | cell phone   | 37.193 | microwave      | 59.493 |
| oven          | 35.518 | toaster      | 44.030 | sink           | 38.810 |
| refrigerator  | 58.260 | book         | 16.807 | clock          | 51.457 |
| vase          | 41.650 | scissors     | 29.540 | teddy bear     | 48.781 |
| hair drier    | 1.733  | toothbrush   | 28.883 |                |        |

```

 **精度调试：**  
> 1.根据代码语义RoiExtractor参数finest_scale不是224而是56  
> 2.因gather算子处理-1会导致每张图的第一个score为0，故maskrcnn_detectron2.diff中已将dets[:, -1]改为dets[:, 4]  
> 3.单张图调试  
> ```
> demo.py分数改为0.05，defaults.py MIN_SIZE_TEST与MAX_SIZE_TEST改为1344：
> python3.7 demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml --input 000000252219_1344x1344.jpg --opts MODEL.WEIGHTS ../../cascadercnn.pkl MODEL.DEVICE cpu
> 说明：
> 精度最初只达到38%，经排查是aligned未生效，将参数aligned设置为1后，精度达标
> ```
> 4.精度调试  
> ```
> 对开源代码预处理与参数修改，使得cpu,gpu版的pkl推理达到npu版代码的pkl推理精度，参见pth的diff文件与执行精度测评的命令。
> 说明：
> 1.查看npu固定1344,1344的前处理方式（缩放加pad）
> from torchvision import utils as vutils
> vutils.save_image(images.tensor, 'test.jpg')
> FIX_SHAPE->./detectron2/data/dataset_mapper.py->ResizeShortestEdge，最短边800最大1333。
> ```


### 6.2 开源精度
[官网精度](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)bbox为：AP:44.3%

### 6.3 精度对比
om推理box map精度为0.439，GPU推理box map精度为0.430，精度下降在1个点之内，因此可视为精度达标

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
batch1的性能：
 测试npu性能要确保device空闲，使用npu-smi info命令可查看device是否在运行其它推理任务
```shell
./benchmark.x86_64 -round=20 -om_path=cascadercnn_detectron2_npu.om -device_id=0 -batch_size=1
```
执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_cascadercnn_detectron2_npu_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 1.32574samples/s, ave_latency: 755.75ms
----------------------------------------------------------------
```
cascadercnn不支持多batch，故只测试batch1的性能  

#### 性能优化
查看profiling导出的op_statistic_0_1.csv算子总体耗时统计发现gather算子耗时最多，然后查看profiling导出的task_time_0_1.csv找到具体哪些gather算子耗时最多，通过导出onnx的verbose打印找到具体算子对应的代码，因gather算子计算最后一个轴会很耗时，因此通过转置后计算0轴规避，比如cascadercnn_detectron2.diff文件中的如下修改：
```
boxes_prof = boxes.permute(1, 0)
widths = boxes_prof[2, :] - boxes_prof[0, :]
```

依据npu版代码修改cpu,gpu版detectron2，参见cascadercnn_pth_npu.diff，测评pth精度与性能：
```shell
git clone https://github.com/facebookresearch/detectron2
python3.7 -m pip install -e detectron2
cd detectron2
patch -p1 < ../cascadercnn_pth_npu.diff
cd tools
mkdir datasets
cp -rf ../../datasets/coco datasets/（数据集构造参考本文第三章第一节步骤五）
python3.7 train_net.py --config-file ../configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml --eval-only MODEL.WEIGHTS ../../cascadercnn.pkl MODEL.DEVICE cuda:0
```
```
Inference done 4999/5000. 0.2339 s / img.
```

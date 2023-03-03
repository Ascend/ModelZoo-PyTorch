# MaskRcnn Onnx模型端到端推理指导
- [MaskRcnn Onnx模型端到端推理指导](#MaskRcnn-onnx模型端到端推理指导)
	- [1 模型概述](#1-模型概述)
		- [1.1 论文地址](#11-论文地址)
		- [1.2 代码地址](#12-代码地址)
	- [2 环境说明](#2-环境说明)
		- [2.1 深度学习框架](#21-深度学习框架)
		- [2.2 python第三方库](#22-python第三方库)
	- [3 数据集预处理](#3-数据集预处理)
		- [3.1 数据集获取](#31-数据集获取)
		- [3.2 数据集预处理](#32-数据集预处理)
		- [3.3 生成数据集信息文件](#33-生成数据集信息文件)
	- [4 模型转换](#3-模型转换)
	    - [4.1 安装detectrin2库](#41-安装detectrin2库) 
		- [4.2 生成onnx模型](#42-生成onnx模型)
		- [4.3 onnx转om模型](#43-onnx转om模型)
	- [5 离线推理](#5-离线推理)
		- [5.1 benchmark工具概述](#51-benchmark工具概述)
		- [5.2 离线推理](#52-离线推理)
	- [6 精度对比](#6-精度对比)
		- [6.1 精度对比](#61-精度对比)
	- [7 性能对比](#7-性能对比)
		- [7.1 310性能数据](#71-310性能数据)
		- [7.2 310p性能数据](#72-310p性能数据)
		- [7.3 T4性能数据](#73-t4性能数据)
		- [7.4 性能对比](#74-性能对比)


## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[论文](https://arxiv.org/pdf/1703.06870.pdf)  

### 1.2 代码地址
[代码](https://github.com/facebookresearch/detectron2) 
``` 
branch=master
commit_id=068a93a
```
## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.1.RC1
python = 3.7.5
pytorch = 1.8.0
torchvision = 0.9.0
onnx = 1.8.0
```

### 2.2 python第三方库

```
numpy == 1.20.1
Pillow == 8.2.0
opencv-python == 4.5.1.48
pycocotools == 12.0
detectron2 == 0.4
```
**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装
>   pycocotools库通过 pip3.7 install "git+https://gitee.com/ztdztd/cocoapi.git#subdirectory=pycocotools"  安装


## 3 数据集预处理

-   **[数据集获取](#31-数据集获取)**  

-   **[数据集预处理](#32-数据集预处理)**  

-   **[生成数据集信息文件](#33-生成数据集信息文件)**  

### 3.1 数据集获取
1.datasets/coco目录下有annotations与val2017，annotations目录存放coco数据集的instances_val2017.json，val2017目录存放coco数据集的5000张验证图片。
2.在服务器home目录下创建自己的文件夹，并将数据集存放于根目录。

### 3.2 数据集预处理
1.将原始数据（.jpg）转化为二进制文件（.bin）。通过缩放、均值方差手段归一化，输出为二进制文件。
  执行maskrcnn_pth_preprocess_detectron2.py脚本。
2.运行成成功后，生成二进制文件val2017_bin。 
```
# 执行以下命令：
   python3.7 maskrcnn_pth_preprocess_detectron2.py --image_src_path=./datasets/coco/val2017 --bin_file_path=val2017_bin --model_input_height=1344 --model_input_width=1344
```
### 3.3 生成数据集信息文件
1.使用get_info.py脚本，输入已经得到的二进制文件，输出生成二进制数据集的info文件。
  运行成功后，在当前目录中生成maskrcnn.info。
```
python3.7 get_info.py bin ./val2017_bin maskrcnn.info 1344 1344

```
2.JPG图片info文件生成,运行成功后，在当前目录中生成maskrcnn_jpeg.info。
```
python3.7 get_info.py jpg ./datasets/coco/val2017 maskrcnn_jpeg.info
```

## 3 模型转换

-   **[安装detectrin2库](#41-安装detectrin2库)**  
-   **[生成onnx模型](#42-生成onnx模型)**
-   **[onnx转om模型](#43-onnx转om模型)**  

### 4.1 安装detectrin2库

1.下载代码仓，到ModleZoo获取的源码包根目录下
```
git clone https://github.com/facebookresearch/detectron2到当前文件夹
cd detectron2/
git reset --hard 068a93a 
```
2.安装detectron2。
```
rm -rf detectron2/build/ **/*.so
pip install -e .
```

3.修改源代码
```
patch -p1 < ../maskrcnn_detectron2.diff
cd ..
```

4.找到自己conda环境中的pytorch安装地址
```
# 打开/root/anaconda3/envs/自己创建的环境名称/lib/python3.7/site-packages/torch/onnx/utils.py文件
  搜索_check_onnx_proto(proto)并注释代码，添加pass代码，后保存并退出。
# _check_onnx_proto(proto)
pass       
```

### 4.2 生成onnx模型

1.运行命令，在outpu文件夹下生成model.onnx文件，获取权重文件[maskrcnn.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Mask-RCNN/PTH/maskrcnn.pth)
```
python3.7 detectron2/tools/deploy/export_model.py --config-file detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml --output ./output --export-method tracing --format onnx MODEL.WEIGHTS maskrcnn.pth MODEL.DEVICE cpu
```
2. 将其改名.
```
mv output/model.onnx model_py1.8.onnx
```
**说明：**  
>该模型目前仅支持batchsize=1。

 **模型转换要点：**  
### 4.3 onnx转om模型

1.运行atc_crnn.sh脚本将2.2生成的model.onnx文件转换om模型,引用环境变量  
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.查看卡的型号，步骤3中的${chip_name}通过该命令查询
```
npu-smi info
```
3.onnx文件转为离线推理模型文件.om文件(请勿直接执行，请根据2中查找的卡型号修改命令)
```
atc --model=model_py1.8.onnx --framework=5 --output=maskrcnn_detectron2_npu --input_format=NCHW --input_shape="0:1,3,1344,1344" --out_nodes="Cast_1673:0;Gather_1676:0;Reshape_1667:0;Slice_1706:0" --log=error --soc_version=Ascend${chip_name}
```

## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.1 推理benchmark工具用户指南 01
### 5.2 离线推理
1.增加benchmark.{arch}可执行权限:
```
chmod u+x benchmark.x86_64
```
2.执行离线推理
```
./benchmark.x86_64 -model_type=vision -om_path=maskrcnn_detectron2_npu.om -device_id=0 -batch_size=1 -input_text_path=maskrcnn.info -input_width=1344 -input_height=1344 -useDvpp=false -output_binary=true
```
推理后的输出默认在当前目录result下。
3.调用后处理脚本maskrcnn_pth_postprocess_detectron2获取txt文件
```
python3.7 maskrcnn_pth_postprocess_detectron2.py --bin_data_path=./result/dumpOutput_device0/ --test_annotation=maskrcnn_jpeg.info --det_results_path=./ret_npuinfer/ --net_out_num=4 --net_input_height=1344 --net_input_width=1344 --ifShowDetObj
```

## 6 精度对比

-   **[精度对比](#61-精度对比)**  

### 6.1 离线推理mAP精度
```
model	    Ap	    AP50	  AP75	    Aps	    APm	    APl	
MaskRcnn	32.739	53.770	  35.056	17.911  36.798  43.272
```
将得到的om离线模型推理mAP精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[310性能数据](#71-310性能数据)**
-   **[310p性能数据](#72-310p性能数据)**  
-   **[T4性能数据](#73-T4性能数据)**  
-   **[性能对比](#74-性能对比)**  

### 7.1 310性能数据
1.benchmark工具在整个数据集上推理获得性能数据  

batch1初始性能：
```
[e2e] throughputRate: 1.77441, latency: 2.81784e+06
[data read] throughputRate: 1.83966, moduleLatency: 543.58
[preprocess] throughputRate: 1.80165, moduleLatency: 555.046
[infer] throughputRate: 1.77503, Interface throughputRate: 2.00815, moduleLatency: 561.56
[post] throughputRate: 1.775, moduleLatency: 563.381
```
batch1 310单卡吞吐率：2.00815 * 4 = 8.0326fps

### 7.2 310p性能数据
在装有310p卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务

batch1性能：
```
[e2e] throughputRate: 5.74867, latency: 869767
[data read] throughputRate: 6.02356, moduleLatency: 166.015
[preprocess] throughputRate: 5.8614, moduleLatency: 170.608
[infer] throughputRate: 5.7554, Interface throughputRate: 11.4222, moduleLatency: 163.148
[post] throughputRate: 5.75507, moduleLatency: 173.76
```
batch1 310p单卡吞吐率：11.4222

### 7.3 T4性能数据
在装有T4卡的服务器上进行在线推理
batch1性能：
```
1 / 0.1633 = 6.12 fps
```

### 7.4 性能对比
batch1： (310)2.00815 * 4 = 8.0326fps < (310p) 11.422fps
batch1： (T4)1 / 0.1633 = 6.12 fps < (310p) 11.422fps
310p上的性能为310性能上的1.2倍以上，310p上的性能为T4性能上的1.6倍以上,性能达标。
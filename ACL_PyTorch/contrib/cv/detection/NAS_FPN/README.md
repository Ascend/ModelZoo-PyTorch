# NAS-FPN模型PyTorch离线推理指导

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
	-   [5.1 安装ais_bench推理工具](#51-安装ais_bench推理工具)
	-   [5.2 离线推理](#52-离线推理)
-   [6 精度对比](#6-精度对比)
	-   [6.1 离线推理mAP精度统计](#61-离线推理mAP精度统计)
	-   [6.2 开源mAP精度](#62-开源mAP精度)
	-   [6.3 精度对比](#63-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[NAS-FPN论文](https://arxiv.org/pdf/1904.07392.pdf)  

### 1.2 代码地址
[NAS-FPN代码](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)  
branch:master  
commit_id:a21eb25535f31634cef332b09fc27d28956fb24b

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.1.RC1
torch==1.7.0
torchvision==0.8.0
onnx==1.8.0
onnxruntime==1.9.0
```

### 2.2 python第三方库

```
numpy==1.20.0
mmdet==2.8.0
mmcv-full==1.2.4
opencv-python==4.4.0.46
mmpycocotools==12.0.3
protobuf==3.20.0
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.准备pth权重文件  
使用训练好的[pth权重文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/NAS-FPN/PTH/retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth)

2.使用开源仓，获取开源命令

```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection  
git reset a21eb25535f31634cef332b09fc27d28956fb24b --hard
pip install -v -e .
```
3.mmdetection代码迁移，执行命令。
```
patch -p1 < ../NAS_FPN.patch   
cd ..
```

4.修改mmcv安装包以适配模型。

通过命令找到mmcv-full安装位置。
```
pip3 show mmcv-full
```
返回mmcv安装位置（如：xxx/lib/python3.7/site-packages）。利用提供的change文件夹中的patch文件，完成补丁操作，命令参考如下示例,请用户根据安装包位置自行修改。
```
cd change
patch -p0 xxx/lib/python3.7/site-packages/mmcv/ops/deform_conv.py deform_conv.patch
patch -p0 xxx/lib/python3.7/site-packages/mmcv/ops/merge_cells.py merge_cells.patch
```

5.调用“mmdetection/tools”目录中的“pytorch2onnx”脚本导出ONNX模型。

当前框架原因仅支持batchsize=1的场景。
```
python3 mmdetection/tools/pytorch2onnx.py mmdetection/configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py xxx.pth --output-file=nas_fpn.onnx --shape=640 --verify --show
```
参数说明：

"mmdetection/configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py"：模型执行文件，不需要修改。

"xxx.pth": 输入pth文件路径。

--output-file：输出onnx文件路径

--shape=640：shape，应指定为640

--verify：是否验证onnx

--show：是否显示onnx结构

 **说明：**  
>注意目前ATC支持的onnx算子版本为11


### 3.2 onnx转om模型

1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 
${chip_name}可通过npu-smi info指令查看，例：310P3

![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```
atc \
--model=./nas_fpn.onnx \
--framework=5 \
--output=nas_fpn_bs1 \
--input_format=NCHW \
--input_shape="input:1,3,640,640" \
--log=error \
--soc_version=${chip_name} \
--out_nodes="Concat_1487:0;Reshape_1489:0"
```
运行成功后生成“nas_fpn_bs1.om”模型文件。
## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
本模型已在coco 2017数据集上验证过精度。推理数据集采用coco_val_2017，请用户自行获取coco_val_2017数据集。

将instances_val2017.json文件和val2017文件夹按照如下目录结构上传并解压数据集到ModelZoo的源码包路径下。
```
├── coco    
       └── val2017                 // 验证集文件夹
├── instances_val2017.json    //验证集标注信息      
```


### 4.2 数据集预处理
将原始数据（.jpg）转化为二进制文件（.bin）。转化方法参考mmdetection预处理方法，以获得最佳精度。以coco_2017数据集为例，通过缩放、均值方差手段归一化，输出为二进制文件。

执行“mmdetection_coco_preprocess.py”脚本。
```
python3 mmdetection_coco_preprocess.py --image_folder_path ./coco/val2017 --bin_folder_path val2017_bin
```
参数说明：

--image_folder_path：原始数据验证集（.jpeg）所在路径。

--bin_folder_path：输出的二进制文件（.bin）所在路径。

每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成“val2017_bin”二进制文件夹。
### 4.3 生成数据集info文件
生成bin文件的输入info文件。

使用ais_bench推理需要输入图片数据集的info文件，用于获取数据集。使用“get_info.py”脚本，输入已经获得的图片文件，输出生成图片数据集的info文件。运行“get_info.py”脚本。

```
python3 get_info.py jpg ./coco/val2017 coco2017_jpg.info
```

参数说明:

“jpg”：生成的数据集文件格式。

“./coco/val2017”：预处理后的数据文件的相对路径。

“coco2017_jpg.info”：生成的数据集文件保存的路径。

运行成功后，在当前目录中生成“coco2017_jpg.info”。

## 5 离线推理

-   **[安装ais_bench推理工具](#51-安装ais_bench推理工具)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 安装ais_bench推理工具

请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

### 5.2 离线推理
昇腾芯片上执行，执行时使npu-smi info查看设备状态，确保device空闲

1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理

执行推理
```
python3 -m ais_bench --model ../NAS_FPN/nas_fpn_1.om --input ../NAS_FPN/val2017_bin --output ../result/
```
参数说明:

--model：模型地址

--input：预处理完的数据集文件夹

--output：推理结果保存地址


## 6 精度对比

-   **[离线推理mAP精度](#61-离线推理mAP精度)**  
-   **[开源mAP精度](#62-开源mAP精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理mAP精度统计

后处理统计mAP精度

调用mmdetection_coco_postprocess.py脚本，将二进制数据转化为txt文件，同时生成画出检测框后的图片，直观展示推理结果。
```
python3 mmdetection_coco_postprocess.py --bin_data_path=../result/2022_xx_xx-xx_xx_xx/ --prob_thres=0.05 --ifShowDetObj --det_results_path=detection-results --test_annotation=coco2017_jpg.info --img_path /opt/npu/coco/val2017 --is_ais_infer
```
参数说明:

--bin_data_path：推理输出目录。

--prob_thres：目标框的置信度阈值，低于阈值的框将被舍弃。

--ifShowDetObj：决定是否生成检测图片。

--det_results：后处理输出目录。

--test_annotation：原始图片信息文件，步骤4.3生成。

--img_path：推理数据集。

--is_ais_infer: 是否使用的是ais_infer进行推理。


评测结果的mAP值需要使用官方的pycocotools工具，首先将后处理输出的txt文件转化为coco数据集评测精度的标准json格式。

执行转换脚本。
```
python3 txt_to_json.py --npu_txt_path=detection-results 
```
参数说明:

--json_output_file=coco_detection_result

--npu_txt_path: txt文件目录，上述后处理步骤中的 det_results 对应的目录

--json_output_file: 结果生成文件

调用“coco_eval.py”脚本，输出推理结果的详细评测报告。
```
python3 coco_eval.py --ground_truth=/opt/npu/coco/annotations/instances_val2017.json --detection_result=coco_detection_result.json
```
参数说明:

--ground_truth: coco数据集标准文件

--detection_result: 模型推理结果文件

### 6.2 开源mAP精度
[开源代码仓精度](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2%2B)

```
Model          mAP     
NAS-FPN        0.405      
```
### 6.3 精度对比
将得到的om离线模型推理mAP精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
1.ais_bench推理工具在整个数据集上推理获得性能数据  
batch1的性能，ais_bench推理工具在整个数据集上推理后生成result/sumary.json：  
```
"NPU_compute_time": {
   "min": 14.676570892333984, 
   "max": 18.673419952392578, 
   "mean": 16.08246865272522, 
   "median": 15.99884033203125, 
   "percentile(99%)": 17.815630435943604}, 
"H2D_latency": {"min": 0.8842945098876953, 
   "max": 9.976387023925781, 
   "mean": 0.9730440616607666, 
   "median": 0.9641647338867188, 
   "percentile(99%)": 1.1441969871521003}, 
"D2H_latency": 
   {"min": 0.02193450927734375, 
   "max": 0.1983642578125, 
   "mean": 0.05659308433532715, 
   "median": 0.054836273193359375, 
   "percentile(99%)": 0.11205911636352545}, 
   "throughput": 62.17950873049252}
```
Interface throughputRate:1000 * batchsize/npu_compute_time.mean=62.30既是batch1 310P单卡吞吐率

**性能优化：**  

>没有遇到性能不达标的问题，故不需要进行性能优化

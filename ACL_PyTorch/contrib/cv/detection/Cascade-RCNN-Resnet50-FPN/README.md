# Cascade R-CNN Onnx模型端到端推理指导

-   [1 模型概述](#1-模型概述)
	-   [1.1 论文地址](#11-论文地址)
	-   [1.2 代码地址](#12-代码地址)
-   [2 环境说明](#2-环境说明)
	-   [2.1 深度学习框架和第三方库](#21-深度学习框架和第三方库)
-   [3 模型转换](#3-模型转换)
	-   [3.1 pth转onnx模型](#31-pth转onnx模型)
	-   [3.2 onnx转om模型](#32-onnx转om模型)
-   [4 数据集预处理](#4-数据集预处理)
	-   [4.1 获取原始数据集](#41-获取原始数据集)
	-   [4.2 数据集预处理](#42-数据集预处理)
	-   [4.3 生成数据集info文件](#43-生成数据集info文件)
-   [5 精度对比](#5-精度对比)
	-   [5.1 离线推理精度](#51-离线推理精度)
	-   [5.2 开源精度](#52-开源精度)
	-   [5.3 精度对比](#53-精度对比)
-   [6 性能对比](#6-性能对比)
	-   [6.1 性能对比](#61-性能对比)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[论文地址](https://arxiv.org/abs/1712.00726)  

### 1.2 代码地址
[代码地址](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn)  
branch:master    
commit id：a21eb25535f31634cef332b09fc27d28956fb24b
## 2 环境说明

-   **[深度学习框架和第三方库](#21-深度学习框架和第三方库)**  

### 2.1 深度学习框架和第三方库
```
python3.7.5
CANN 5.1.RC1

pytorch = 1.7.0
torchvision = 0.8.0
onnx = 1.7.0
onnxoptimizer = 0.2.7
onnxruntime = 1.5.2
opencv-python = 4.4.0.46
pillow = 9.0.1
numpy == 1.21.5
cpython = 0.29.30
mmcv = 1.2.5
mmdet = 2.8.0
mmpycocotools = 12.0.3
```

**说明：** 
   pytorch 安装 cpu版本
```
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 -c pytorch
```
   mmcv 也安装cpu版本
```
pip install mmcv-full==1.2.5 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.7.0/index.html
```
   mmdetecton 需要下载代码仓后，在源码包根目录下用源码安装
```
git clone --branch v2.8.0 https://github.com/open-mmlab/mmdetection
cd mmdetection
git reset --hard a21eb25535f31634cef332b09fc27d28956fb24b
pip install -v -e .
```
   其他包可以通过pip安装
```
pip install opencv-python==4.4.0.46 onnxruntime==1.5.2 mmpycocotools==12.0.3 onnxoptimizer
```
  

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.下载pth权重文件  
[Cascade_RCNN预训练pth权重文件](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth) 
下载经过训练的Cascade-RCNN-R50-FPN-1X-COCO模型权重文件，并移动到Modelzoo源码包中。

2.修改修改mmdetection源码适配Ascend NPU

使用mmdetection（v2.8.0）导出onnx前, 需要对源码做一定的改动，以适配Ascend NPU。具体的代码改动请参考Modelzoo源码包中的Cascade_RCNN修改实现.md文档。添加NPU自定义算子后需要屏蔽掉torch.onnx中的model_check相关代码，否则导出onnx过程中无法识别自定义算子会导致报错。

代码修改方式使用patch，在源码包目录下运行命令：
```
patch -p1 < Cascade_RCNN.patch
```


b.（可选）修改cascade_rcnn_r50_fpn.py文件中nms_post参数
打开文件。
```
vi mmdetection/configs/_base_/models/cascade_rcnn_r50_fpn.py
```
修改参数。
```python
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=500,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
```
**说明：** 
由于NPU RoiExtractor算子的特殊性，适当减少其输入框的数量可以在小幅度影响精度的基础上大幅度提高性能，推荐将test_cfg中rpn层的nms_post参数从1000改为500，用户可以自行决定是否应用此项改动。

d.替换相应源代码后，调用mmdete/tools目录中的pytorch2onnx脚本导出ONNX模型。这里注意指定shape为1216。
```
python3.7 mmdetection/tools/pytorch2onnx.py mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py ./cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth --output-file=cascade_rcnn_r50_fpn.onnx --shape=1216 --verify --show
```
运行成功后在当前目录生成cascade_crnn_r50_fpn.onnx文件。此模型当前仅支持batch_size=1。


### 3.2 onnx转om模型

1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)

修改好的onnx可以直接利用ATC工具转换为om，进行离线推理。在转换om文件前，通过ATC工具的--out_nodes参数，指定输出节点为boxes和labels前的最后两个节点，可以剔除修改模型时产生的多余分支。不同模型的输出节点会有区别，可以用netron等模型可视化工具打开后查看output节点前的最后算子名称，并指定为ATC输出节点--out_nodes。

运行atc命令，完成onnx到om模型转换。

```
atc --model=./cascade_rcnn_r50_fpn.onnx --framework=5 --output=cascade_rcnn_r50_fpn_wjy --input_format=NCHW --input_shape="input:1,3,1216,1216" --log=info --soc_version=Ascend${chip_name} --out_nodes="Concat_828:0;Reshape_830:0"
```
注：${chip_name}由“npu-smi info”命令查看处理器获得。

运行成功后，生成cascade_rcnn_r50_fpn.om文件。

**参数说明：**
- --model：为ONNX模型文件。
- --framework：5代表ONNX模型。
- --output：输出的OM模型。
- --input_format：输入数据的格式。
- --input_shape：输入数据的shape。
- --out_nodes：输出节点名称。
- --log：日志级别。
- --soc_version：推理设备名称。

## 4 数据集预处理

-   **[获取原始数据集](#41-获取原始数据集)**  
-   **[数据集预处理](#42-数据集预处理)**  
-   **[生成数据集info文件](#43-生成数据集info文件)**  

### 4.1 获取原始数据集
本模型已在coco 2017数据集上验证过精度。推理数据集采用coco_val_2017，请用户自行获取coco_val_2017数据集或者参考mmdetection相关指导获取其他数据集.
将instances_val2017.json文件和val2017文件夹上传并解压数据集到ModelZoo的源码包路径下。
```
├── instances_val2017.json    //验证集标注信息       
└── data/val2017             // 验证集文件夹
```

### 4.2 数据集预处理
1.将原始数据集转换为模型输入的二进制数据。

将原始数据（.jpg）转化为二进制文件（.bin）。转化方法参考mmdetection预处理方法，以获得最佳精度。以coco_2017数据集为例，通过缩放、均值方差手段归一化，输出为二进制文件。

执行mmdetection_coco_preprocess.py脚本。
```
python3.7 mmdetection_coco_preprocess.py --image_folder_path ./data/val2017 --bin_folder_path ./data/val2017_bin
```
**说明：**
- --image_folder_path：原始数据验证集（.jpg）所在路径。
- --bin_folder_path：输出的二进制文件（.bin）所在路径。
每个图像对应生成一个二进制文件。

### 4.3 生成数据集info文件
1.二进制输入info文件生成
使用脚本计算精度时需要输入二进制数据集的info文件

1.JPG图片info文件生成
后处理时需要输入数据集.jpg图片的info文件。使用get_info.py脚本，输入已经获得的图片文件，输出生成图片数据集的info文件。

运行get_info.py脚本。
```
python3.7 get_info.py jpg ../dataset/coco/val2017 coco2017_jpg.info
```
第一个参数为生成的数据集文件格式，第二个参数为coco图片数据文件的相对路径，第三个参数为生成的数据集信息文件保存的路径。运行成功后，在当前目录中生成coco2017_jpg.info。


## 5 精度对比

-   **[离线推理精度](#51-离线推理精度)**  
-   **[开源精度](#52-开源精度)**  
-   **[精度对比](#53-精度对比)**  

### 5.1 离线推理精度
1.使用ais_infer工具进行离线推理.

```
python3.7 ais_infer.py --model cascade_rcnn_r50_fpn_aoe.om --input data/val2017_bin/  --output ais_infer_result --outfmt BIN --batchsize 1
```
**参数说明:**
- --mode：om模型路径。
- --input：二进制数据集文件夹路径。
- --output：输出文件夹路径。
- --outfmt：后处理输出格式。
- --batchsize：推理的batchsize大小。

2.推理结果展示。

本模型提供后处理脚本，将二进制数据转化为txt文件，同时生成画出检测框后的图片，直观展示推理结果。
执行脚本。
```
python mmdetection_coco_postprocess.py --bin_data_path=ais_infer_result/日期文件夹 --prob_thres=0.05 --ifShowDetObj --det_results_path=ais_infer_detection_results --test_annotation=coco2017_jpg.info
```
**参数说明:**
- --bin_data_path：推理输出目录。
- --prob_thres：框的置信度阈值，低于阈值的框将被舍弃。
- --ifShowDetObj：决定是否生成检测图片。
- --det_results：后处理输出目录。
- --test_annotation：原始图片信息文件，源码包中提供。

3.精度验证
评测结果的mAP值需要使用官方的pycocotools工具，首先将后处理输出的txt文件转化为coco数据集评测精度的标准json格式。

执行转换脚本。
```
python txt_to_json.py --npu_txt_path ais_infer_detection_results --json_output_file coco_detection_aisInfer_result
```
运行成功后，生成coco_detection_result.json文件。

调用coco_eval.py脚本，输出推理结果的详细评测报告。
```
python coco_eval.py --detection_result coco_detection_aisInfer_result.json
```

4.精度结果：
310 batch1的精度：
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.405
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.589
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.438
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.236
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.444
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.516
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.551
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.551
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.551
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.354
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.590
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.687
{'bbox_mAP': 0.405, 'bbox_mAP_50': 0.589, 'bbox_mAP_75': 0.438, 'bbox_mAP_s': 0.236, 'bbox_mAP_m': 0.444, 'bbox_mAP_l': 0.516, 'bbox_mAP_copypaste': '0.405 0.589 0.438 0.236 0.444 0.516'}
```
310p batch1的精度：
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.405
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.589
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.438
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.235
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.444
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.515
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.551
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.551
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.551
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.355
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.590
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.688
{'bbox_mAP': 0.405, 'bbox_mAP_50': 0.589, 'bbox_mAP_75': 0.438, 'bbox_mAP_s': 0.235, 'bbox_mAP_m': 0.444, 'bbox_mAP_l': 0.515, 'bbox_mAP_copypaste': '0.405 0.589 0.438 0.235 0.444 0.515'}
```

T4 batch1的精度：
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.403
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.586
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.440
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.225
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.438
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.529
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.543
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.543
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.543
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.333
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.582
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.689
OrderedDict([('bbox_mAP', 0.403), ('bbox_mAP_50', 0.586), ('bbox_mAP_75', 0.44), ('bbox_mAP_s', 0.225), ('bbox_mAP_m', 0.438), ('bbox_mAP_l', 0.529), ('bbox_mAP_copypaste', '0.403 0.586 0.440 0.225 0.438 0.529')])
```


### 5.2 开源精度
[github开源代码仓精度](https://github.com/open-mmlab/mmdetection/blob/master/configs/cascade_rcnn/README.md)

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: |:------:|:--------:|
|    R-50-FPN     | pytorch |   1x    |   4.4    |      16.1      |  40.3  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316_214748.log.json) |


### 5.3 精度对比

分别将310和310p上得到的om离线模型推理精度与该模型github代码仓上公布的精度对比，精度均达标。
|    精度    | 310     | 310p    | T4     | 开源   | 
| ---------- | ------- | ------- | ------ | ------ |
|   box AP   |  40.5   |  40.5   |  40.3  |  40.3  |

在310上满足精度要求（40.5>=40.3);
在310p上满足精度要求（40.5>=40.3）;
在T4上满足精度要求（40.3>=40.3）;

## 6 性能对比

-   **[性能对比](#61-性能对比)**  

### 6.1 性能对比

| Throughput | 310     |310P(aoe)| T4     | 310P/310 | 310P/T4   |
| :---------- | :------- | :------- | :------ | :-------- | :--------- |
| bs1        | 3.53833 | 6.23197 | 2.60000 | 1.76127 | 2.39691 |

经过AOE优化后性能达标：
310p的最优batch性能 >=1.2倍310最优batch性能x 4，（1.76>1.2,性能满足）
310p的最优batch性能 >=1.6倍T4最优batch性能，（2.39>1.6,性能满足）


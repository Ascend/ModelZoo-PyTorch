## Faster-R-CNN Onnx模型PyTorch端到端推理指导

### 1 模型概述

#### 1.1 论文地址

[Faster-R-CNN论文](https://arxiv.org/abs/1506.01497)



#### 1.2 代码地址

[Faster-R-CNN代码](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn)



### 2 环境说明

#### 2.1 深度学习框架

```
CANN 5.0.2
torch == 1.5.0
torchvision == 0.8.0
onnx == 1.7.0
```



#### 2.2 python第三方库

```
numpy == 1.19.4
Pillow == 8.2.0
opencv-python == 4.4.0.46
mmdet == 2.8.0
mmcv == 1.2.5
mmpycocotools == 12.0.3
```

**说明：**

> X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
> Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

### 3 准备数据集

#### 3.1 获取原始数据集
本模型已在coco 2017数据集上验证过精度。推理数据集采用coco_val_2017，请用户自行获取coco_val_2017数据集或者参考mmdetection相关指导获取其他数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到coco_val_2017数据集及其ground truth文件instances_val2017。
#### 3.2 数据预处理
数据预处理将原始数据集转换为模型输入的二进制数据。

将原始数据（.jpeg）转化为二进制文件（.bin）。转化方法参考mmdetection预处理方法，以获得最佳精度。以coco_2017数据集为例，通过缩放、均值方差手段归一化，输出为二进制文件。

执行mmdetection_coco_preprocess.py脚本。

   ```
   python3.7 mmdetection_coco_preprocess.py --image_folder_path val2017 --bin_folder_path val2017_bin
   ```
#### 3.3 生成数据集info文件
- 二进制输入info文件生成
使用benchmark推理需要输入二进制数据集的info文件，用于获取数据集。使用get_info.py脚本，输入已经得到的二进制文件，输出生成二进制数据集的info文件。运行get_info.py脚本
   ```
   python3.7 get_info.py bin ./val2017_bin coco2017.info 1216 1216
   ```
  第一个参数为生成的数据集文件格式，第二个参数为预处理后的数据文件相对路径，第三个参数为生成的数据集文件保存的路径，第四个和第五个参数分别为模型输入的宽度和高度。

   运行成功后，在当前目录中生成coco2017.info。


- JPG图片info文件生成

   后处理时需要输入数据集.jpg图片的info文件。使用get_info.py脚本，输入已经获得的图片文件，输出生成图片数据集的info文件。运行get_info.py脚本。
   ```
   python3.7 get_info.py jpg ../dataset/coco/val2017 coco2017_jpg.info
   ```
  第一个参数为生成的数据集文件格式，第二个参数为coco图片数据文件的相对路径，第三个参数为生成的数据集信息文件保存的路径。运行成功后，在当前目录中生成coco2017_jpg.info。


### 4 模型转换

#### 4.1 pth转onnx模型

1. 下载pth权重文件

   [Faster-RCNN-R50-FPN-1X-COCO预训练pth权重文件](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth)

   

2. 导出onnx模型文件需要安装mmdetection项目及其依赖。下载代码仓，到ModleZoo获取的源码包根目录下，并安装安装mmdetection。

   ```
   git clone https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   git reset --hard a21eb25535f31634cef332b09fc27d28956fb24b
   pip3.7 install -v -e .
   cd ..
   ```
   
3. 修改mmdetection源码适配Ascend NPU。使用mmdetection（v2.8.0）导出onnx前, 
需要对源码做一定的改动，以适配Ascend NPU。具体的代码改动请参考Modelzoo源码包中的
Faster_RCNN修改实现.md文档，修改后的同名文件已在源码包中提供，
用户可以直接在相应目录中备份原文件并替换。
   ```
    cp ./pytorch_code_change/bbox_nms.py ./mmdetection/mmdet/core//post_processing/bbox_nms.py
    cp ./pytorch_code_change/rpn_head.py ./mmdetection/mmdet/models/dense_heads/rpn_head.py
    cp ./pytorch_code_change/single_level_roi_extractor.py ./mmdetection/mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py
    cp ./pytorch_code_change/delta_xywh_bbox_coder.py ./mmdetection/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py
   ```

4. 屏蔽掉torch.onnx中的model_check相关代码。
注册添加NPU自定义算子后需要手动屏蔽掉torch.onnx中的model_check相关代码，否则导出onnx过程中无法识别自定义算子会导致报错。

   通过命令找到pytorch安装位置。
   ```
   pip3.7 show torch
   ```
   返回pytorch安装位置（如：xxx/lib/python3.7/site-packages）。打开文件改路径下的/torch/onnx/utils.py文件。

    ```
    vi xxx/lib/python3.7/site-packages/torch/onnx/utils.py
    ```
   搜索_check_onnx_proto(proto)并注释代码，添加pass代码，后保存并退出。

    ```
    if enable_onnx_checker and \
        operator_export_type is OperatorExportTypes.ONNX and \
            not val_use_external_data_format:
        # Only run checker if enabled and we are using ONNX export type and
        # large model format export in not enabled.
        # _check_onnx_proto(proto)
        pass
    ```
5. （可选）修改cascade_rcnn_r50_fpn.py文件中nms_post参数

    打开文件。
    ```
    vi mmdetection/configs/_base_/models/cascade_rcnn_r50_fpn.py
    ```
   修改参数。
    ```
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
   说明：
   > 由于NPU RoiExtractor算子的特殊性，适当减少其输入框的数量可以在小幅度影响精度的基础上大幅度提高性能，推荐将test_cfg中rpn层的nms_post参数从1000改为500，用户可以自行决定是否应用此项改动。


6. 替换相应源代码后，调用mmdetection/tools目录中的pytorch2onnx脚本导出ONNX模型。这里注意指定shape为1216。

   ```
   python3.7 mmdetection/tools/pytorch2onnx.py mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py ./faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --output-file faster_rcnn_r50_fpn.onnx --shape=1216 --verify --show
   ```
   运行成功后在当前目录生成faster_rcnn_r50_fpn.onnx文件。此模型当前仅支持batch_size=1。


#### 4.2 onnx转om模型

1. 设置环境变量

   ```
   source env.sh
   ```

2. 使用ATC工具将ONNX模型转OM模型。
修改好的onnx可以直接利用ATC工具转换为om，进行离线推理。通过ATC工具的--out_nodes参数，指定输出节点为boxes和labels前的最后两个节点，可以剔除修改模型时产生的多余分支。不同模型的输出节点会有区别，可以用netron等模型可视化工具打开后查看output节点前的最后算子名称为Concat_569:0;Reshape_571:0，并指定为ATC输出节点--out_nodes。

   ```
   atc --framework=5 --model=faster_rcnn_r50_fpn.onnx --output=faster_rcnn_r50_fpn --input_format=NCHW --input_shape="image:1,3,1216,1216" --log=debug --soc_version=Ascend${chip_name}
   ```
   ${chip_name}可通过npu-smi info指令查看，例：710

   运行成功后，生成faster_rcnn_r50_fpn.om文件。



### 5 离线推理
ais_infer推理工具链接
https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer

####  5.1 使用ais_infer工具进行推理

```
python tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model faster_rcnn_r50_fpn.om --input=val2017_bin --output=ais_infer_result
```

推理结果输出路径，默认会建立日期+时间的子文件夹保存输出结果
推理默认输出目录为ais_infer_result，结果保存为二进制格式。

####  5.2 数据后处理
本模型提供后处理脚本，将二进制数据转化为txt文件，同时生成画出检测框后的图片，直观展示推理结果。
执行脚本。

```
python3.7 mmdetection_coco_postprocess.py --bin_data_path=result/dumpOutput_device0 --prob_thres=0.05 --ifShowDetObj --det_results_path=detection-results --test_annotation=coco2017_jpg.info
```
bin_data_path是推理输出目录。prob_thres是框的置信度阈值，低于阈值的框将被舍弃。 ifShowDetObj决定是否生成检测图片。det_results为后处理输出目录。test_annotation是原始图片信息文件，源码包中提供。

#### 5.3 精度验证

评测结果的mAP值需要使用官方的pycocotools工具，首先将后处理输出的txt文件转化为coco数据集评测精度的标准json格式。
执行转换脚本。

```
python3.7 txt_to_json.py
```

运行成功后，生成json文件。

调用coco_eval.py脚本，输出推理结果的详细评测报告。

```
python3.7 coco_eval.py
```



### 6 精度与性能
#### 6.1 精度
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.372
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.585
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.402
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.227
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.408
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.464
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.519
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.519
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.519
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.340
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.560
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.642
{'bbox_mAP': 0.372, 'bbox_mAP_50': 0.585, 'bbox_mAP_75': 0.402, 'bbox_mAP_s': 0.227, 'bbox_mAP_m': 0.408, 'bbox_mAP_l': 0.464, 'bbox_mAP_copypaste': '0.372 0.585 0.402 0.227 0.408 0.464'}

```

github仓库中给出的官方精度为box AP：37.5，npu离线推理的精度为box AP：37.2，超过参考精度的99%，故精度达标
#### 6.2 性能
| Model | Batch size | 310(FPS/Card) | 310P(FPS/Card) | T4(FPS/Card) | 310P/310 | 310P/T4 |
| --- | --- | --- | --- | --- | --- | --- |  
| Faster R-CNN | 1 | 8.828 | 15.2619 | 5.83 | 1.728 | 2.617 |
性能达标



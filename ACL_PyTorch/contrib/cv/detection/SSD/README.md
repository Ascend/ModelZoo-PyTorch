# 基于开源mmdetection预训练的SSD Onnx模型端到端推理指导
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
	-   [7.2 基准性能数据](#72-基准性能数据)
	-   [7.3 性能对比](#73-性能对比)

## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[SSD论文](https://arxiv.org/abs/1512.02325)  
SSD将detection转化为regression的思路，可以一次完成目标定位与分类。该算法基于Faster RCNN中的Anchor，提出了相似的Prior box；该算法修改了传统的VGG16网络：将VGG16的FC6和FC7层转化为卷积层，去掉所有的Dropout层和FC8层。同时加入基于特征金字塔的检测方式，在不同感受野的feature map上预测目标。

### 1.2 代码地址
[mmdetection框架SSD代码](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd)   

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
onnx==1.7.0
torch==1.8.1
torchvision==0.9.1
```

### 2.2 python第三方库

```
numpy==1.18.5
opencv-python==4.2.0.34
mmdet==2.8.0
mmcv-full==1.2.4
mmpycocotools==12.0.3
```

**说明：** 
>   X86架构：opencv,pytorch,torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：opencv,pytorch,torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

atc暂不支持动态shape小算子，可以使用大颗粒算子替换这些小算子规避，这些小算子可以在转onnx时的verbose打印中找到其对应的python代码，从而根据功能用大颗粒算子替换，onnx能推导出变量正确的shape与算子属性正确即可，变量实际的数值无关紧要，因此这些大算子函数的功能实现无关紧要，因包含自定义算子需要去掉对onnx模型的校验。

### 3.1 pth转onnx模型

1.下载pth权重文件
[SSD300预训练pth权重文件](http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20200307-a92d2092.pth)
文件md5sum: 496e671b20bda2b4f53051f298947bba

```
wget http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20200307-a92d2092.pth
```

2.mmdetection源码安装

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git reset --hard a21eb25535f31634cef332b09fc27d28956fb24b
pip install -r requirements/build.txt
pip install -v -e .
```

 **说明：**  
> 安装所需的依赖说明请参考mmdetection/docs/get_started.md
>

3.转原始onnx

```shell
python3.7 mmdetection/tools/pytorch2onnx.py mmdetection/configs/ssd/ssd300_coco.py ./ssd300_coco_20200307-a92d2092.pth --output-file=ssd_300_coco.onnx --shape=300 --verify --show --mean 123.675 116.28 103.53 --std 1 1 1
```
4.修改mmdetection代码，参见ssd_mmdetection.diff

~~~python
diff --git a/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py b/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py
index e9eb3579..e8b53dce 100644
--- a/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py
+++ b/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py
@@ -168,8 +168,13 @@ def delta2bbox(rois,
                 [0.0000, 0.3161, 4.1945, 0.6839],
                 [5.0000, 5.0000, 5.0000, 5.0000]])
     """
-    means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 4)
-    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 4)
+    # fix shape for means and stds for onnx
+    if torch.onnx.is_in_onnx_export():
+        means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1).numpy() // 4)
+        stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1).numpy() // 4)
+    else:
+        means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 4)
+        stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 4)
     denorm_deltas = deltas * stds + means
     dx = denorm_deltas[:, 0::4]
     dy = denorm_deltas[:, 1::4]
@@ -178,12 +183,22 @@ def delta2bbox(rois,
     max_ratio = np.abs(np.log(wh_ratio_clip))
     dw = dw.clamp(min=-max_ratio, max=max_ratio)
     dh = dh.clamp(min=-max_ratio, max=max_ratio)
-    # Compute center of each roi
-    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
-    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
-    # Compute width/height of each roi
-    pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dw)
-    ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dh)
+    # improve gather performance on NPU
+    if torch.onnx.is_in_onnx_export():
+        rois_perf = rois.permute(1, 0)
+        # Compute center of each roi
+        px = ((rois_perf[0, :] + rois_perf[2, :]) * 0.5).unsqueeze(1).expand_as(dx)
+        py = ((rois_perf[1, :] + rois_perf[3, :]) * 0.5).unsqueeze(1).expand_as(dy)
+        # Compute width/height of each roi
+        pw = (rois_perf[2, :] - rois_perf[0, :]).unsqueeze(1).expand_as(dw)
+        ph = (rois_perf[3, :] - rois_perf[1, :]).unsqueeze(1).expand_as(dh)
+    else:
+        # Compute center of each roi
+        px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
+        py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
+        # Compute width/height of each roi
+        pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dw)
+        ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dh)
     # Use exp(network energy) to enlarge/shrink each roi
     gw = pw * dw.exp()
     gh = ph * dh.exp()
diff --git a/mmdet/core/post_processing/bbox_nms.py b/mmdet/core/post_processing/bbox_nms.py
index 463fe2e4..1f8ad5a8 100644
--- a/mmdet/core/post_processing/bbox_nms.py
+++ b/mmdet/core/post_processing/bbox_nms.py
@@ -4,6 +4,57 @@ from mmcv.ops.nms import batched_nms
 from mmdet.core.bbox.iou_calculators import bbox_overlaps
 
 
+class BatchNMSOp(torch.autograd.Function):
+    @staticmethod
+    def forward(ctx, bboxes, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size):
+        """
+        boxes (torch.Tensor): boxes in shape (batch, N, C, 4).
+        scores (torch.Tensor): scores in shape (batch, N, C).
+        return:
+            nmsed_boxes: (1, N, 4)
+            nmsed_scores: (1, N)
+            nmsed_classes: (1, N)
+            nmsed_num: (1,)
+        """
+
+        # Phony implementation for onnx export
+        nmsed_boxes = bboxes[:, :max_total_size, 0, :]
+        nmsed_scores = scores[:, :max_total_size, 0]
+        nmsed_classes = torch.arange(max_total_size, dtype=torch.long)
+        nmsed_num = torch.Tensor([max_total_size])
+
+        return nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num
+
+    @staticmethod
+    def symbolic(g, bboxes, scores, score_thr, iou_thr, max_size_p_class, max_t_size):
+        nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = g.op('BatchMultiClassNMS',
+            bboxes, scores, score_threshold_f=score_thr, iou_threshold_f=iou_thr,
+            max_size_per_class_i=max_size_p_class, max_total_size_i=max_t_size, outputs=4)
+        return nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num
+
+def batch_nms_op(bboxes, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size):
+    """
+    boxes (torch.Tensor): boxes in shape (N, 4).
+    scores (torch.Tensor): scores in shape (N, ).
+    """
+
+    if bboxes.dtype == torch.float32:
+        bboxes = bboxes.reshape(1, bboxes.shape[0].numpy(), -1, 4).half()
+        scores = scores.reshape(1, scores.shape[0].numpy(), -1).half()
+    else:
+        bboxes = bboxes.reshape(1, bboxes.shape[0].numpy(), -1, 4)
+        scores = scores.reshape(1, scores.shape[0].numpy(), -1)
+
+    nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = BatchNMSOp.apply(bboxes, scores,
+        score_threshold, iou_threshold, max_size_per_class, max_total_size)
+    nmsed_boxes = nmsed_boxes.float()
+    nmsed_scores = nmsed_scores.float()
+    nmsed_classes = nmsed_classes.long()
+    dets = torch.cat((nmsed_boxes.reshape((max_total_size, 4)), nmsed_scores.reshape((max_total_size, 1))), -1)
+    labels = nmsed_classes.reshape((max_total_size, ))
+    return dets, labels
+
+
 def multiclass_nms(multi_bboxes,
                    multi_scores,
                    score_thr,
@@ -36,13 +87,25 @@ def multiclass_nms(multi_bboxes,
     if multi_bboxes.shape[1] > 4:
         bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
     else:
-        bboxes = multi_bboxes[:, None].expand(
-            multi_scores.size(0), num_classes, 4)
+        # export expand operator to onnx more nicely
+        if torch.onnx.is_in_onnx_export:
+            bbox_shape_tensor = torch.ones(multi_scores.size(0), num_classes, 4)
+            bboxes = multi_bboxes[:, None].expand_as(bbox_shape_tensor)
+        else:
+            bboxes = multi_bboxes[:, None].expand(
+                multi_scores.size(0), num_classes, 4)
+
 
     scores = multi_scores[:, :-1]
     if score_factors is not None:
         scores = scores * score_factors[:, None]
 
+    # npu
+    if torch.onnx.is_in_onnx_export():
+        dets, labels = batch_nms_op(bboxes, scores, score_thr, nms_cfg.get("iou_threshold"), max_num, max_num)
+        return dets, labels
+
+    # cpu and gpu
     labels = torch.arange(num_classes, dtype=torch.long)
     labels = labels.view(1, -1).expand_as(scores)
 
@@ -53,6 +116,8 @@ def multiclass_nms(multi_bboxes,
     # remove low scoring boxes
     valid_mask = scores > score_thr
     inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
+    # vals, inds = torch.topk(scores, 1000)
+
     bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
     if inds.numel() == 0:
         if torch.onnx.is_in_onnx_export():
@@ -76,6 +141,7 @@ def multiclass_nms(multi_bboxes,
         return dets, labels[keep]
 
 
+
 def fast_nms(multi_bboxes,
              multi_scores,
              multi_coeffs,

~~~

**修改依据：**  

> 1. 在bbox_nms.py文件中用NPU算子BatchMultiNMS代替原mmdetection中的NMS层算子，替换后精度无损失。同时等价换一个expand算子，使导出的onnx中不含动态shape。
>
> 4. delta_xywh_bbox_coder.py 中修改坐标的轴顺序，使切片操作在NPU上效率更高，整网性能提升约7%；修改means和std计算方法使其表现为固定shape。


通过打补丁的方式修改mmdetection：
```shell
patch -p1 < ../ssd_mmdetection.diff
```
5.修改pytorch代码去除导出onnx时进行检查  
将/usr/local/python3.7.5/lib/python3.7/site-packages/torch/onnx/utils.py文件的_check_onnx_proto(proto)改为pass

6.运行如下命令，生成含有npu自定义算子的onnx：

```shell
python3.7 mmdetection/tools/pytorch2onnx.py mmdetection/configs/ssd/ssd300_coco.py ./ssd300_coco_20200307-a92d2092.pth --output-file=ssd_300_coco.onnx --shape=300 --verify --show --mean 123.675 116.28 103.53 --std 1 1 1
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
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)，需要指定输出节点以去除无用输出，节点序号可能会因网络结构不同而不同，使用netron开源可视化工具查看具体的输出节点名：

```shell
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=ssd_300_coco.onnx --framework=5 --output=ssd_300_coco --input_format=NCHW --input_shape="input:1,3,300,300" --log=info --soc_version=Ascend310 --out_nodes="Concat_637:0;Reshape_639:0" --buffer_optimize=off_optimize --precision_mode allow_mix_precision
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[COCO官网](https://cocodataset.org/#download)的coco2017的5千张验证集进行测试，图片与标签分别存放在/root/datasets/coco/val2017/与/root/datasets/coco/annotations/instances_val2017.json。

### 4.2 数据集预处理
1.预处理脚本mmdetection_coco_preprocess.py

```python
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import cv2
import argparse
import mmcv
import torch

dataset_config = {
        'resize': (300, 300),
        'mean': [123.675, 116.28, 103.53],
        'std': [1, 1, 1],
}

tensor_height = 300
tensor_width = 300
    
def coco_preprocess(input_image, output_bin_path):
    #define the output file name 
    img_name = input_image.split('/')[-1]
    #print(img_name)
    bin_name = img_name.split('.')[0] + ".bin"
    bin_fl = os.path.join(output_bin_path, bin_name)

    one_img = mmcv.imread(os.path.join(input_image), backend='cv2')
    # one_img = mmcv.imrescale(one_img, (tensor_height, tensor_width))
    one_img = mmcv.imresize(one_img, (tensor_height, tensor_width))
    # calculate padding
    h = one_img.shape[0]
    w = one_img.shape[1]
    #print(h,w,tensor_height,tensor_width)
    pad_left = (tensor_width - w) // 2
    pad_top = (tensor_height - h) // 2
    pad_right = tensor_width - pad_left - w
    pad_bottom = tensor_height - pad_top - h

    mean = np.array(dataset_config['mean'], dtype=np.float32)
    std = np.array(dataset_config['std'], dtype=np.float32)
    one_img = mmcv.imnormalize(one_img, mean, std)
    # one_img = mmcv.impad(one_img, padding=(pad_left, pad_top, pad_right, pad_bottom), pad_val=0)
    one_img = one_img.transpose(2, 0, 1)
    one_img.tofile(bin_fl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess of FasterRCNN pytorch model')
    parser.add_argument("--image_folder_path", default="./coco2014/", help='image of dataset')
    parser.add_argument("--bin_folder_path", default="./coco2014_bin/", help='Preprocessed image buffer')
    flags = parser.parse_args()    

    if not os.path.exists(flags.bin_folder_path):
        os.makedirs(flags.bin_folder_path)
    images = os.listdir(flags.image_folder_path)
    for image_name in images:
        if not (image_name.endswith(".jpeg") or image_name.endswith(".JPEG") or image_name.endswith(".jpg")):
            continue
        #print("start to process image {}....".format(image_name))
        path_image = os.path.join(flags.image_folder_path, image_name)
        coco_preprocess(path_image, flags.bin_folder_path)

```
2.执行预处理脚本，生成数据集预处理后的bin文件
```shell
python3.7 mmdetection_coco_preprocess.py --image_folder_path /root/datasets/coco/val2017 --bin_folder_path val2017_ssd_bin
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py

```python
import os
import sys
import cv2
from glob import glob


def get_bin_info(file_path, info_name, width, height):
    bin_images = glob(os.path.join(file_path, '*.bin'))
    with open(info_name, 'w') as file:
        for index, img in enumerate(bin_images):
            content = ' '.join([str(index), img, width, height])
            file.write(content)
            file.write('\n')


def get_jpg_info(file_path, info_name):
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    image_names = []
    for extension in extensions:
        image_names.append(glob(os.path.join(file_path, '*.' + extension)))  
    with open(info_name, 'w') as file:
        for image_name in image_names:
            if len(image_name) == 0:
                continue
            else:
                for index, img in enumerate(image_name):
                    img_cv = cv2.imread(img)
                    shape = img_cv.shape
                    width, height = shape[1], shape[0]
                    content = ' '.join([str(index), img, str(width), str(height)])
                    file.write(content)
                    file.write('\n')


if __name__ == '__main__':
    file_type = sys.argv[1]
    file_path = sys.argv[2]
    info_name = sys.argv[3]
    if file_type == 'bin':
        width = sys.argv[4]
        height = sys.argv[5]
        assert len(sys.argv) == 6, 'The number of input parameters must be equal to 5'
        get_bin_info(file_path, info_name, width, height)
    elif file_type == 'jpg':
        assert len(sys.argv) == 4, 'The number of input parameters must be equal to 3'
        get_jpg_info(file_path, info_name)
```
2.执行生成数据集信息脚本，生成数据集信息文件
```shell
python3.7 get_info.py bin ./val2017_ssd_bin coco2017_ssd.info 300 300
python3.7 get_info.py jpg /root/datasets/coco/val2017 coco2017_ssd_jpg.info
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
./benchmark.x86_64 -model_type=vision -batch_size=1 -device_id=0 -input_text_path=./coco2017_ssd.info -input_width=300 -input_height=300 -useDvpp=False -output_binary=true -om_path=ssd_300_coco.om
```
 **注意：**  
> label是int64，benchmark输出非二进制时会将float转为0
>

输出结果默认保存在当前目录result/dumpOutput_device0，模型有两个输出，每个输入对应的输出对应两个_x.bin文件
```
输出       shape                 数据类型    数据含义
output1    200 * 5               FP32       boxes and scores
output2    200 * 1               INT64      labels
```

## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计

后处理统计map精度
```python
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import argparse
import cv2

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def coco_postprocess(bbox: np.ndarray, image_size, 
                        net_input_width, net_input_height):
    """
    This function is postprocessing for FasterRCNN output.

    Before calling this function, reshape the raw output of FasterRCNN to
    following form
        numpy.ndarray:
            [x, y, width, height, confidence, probability of 80 classes]
        shape: (100,)
    The postprocessing restore the bounding rectangles of FasterRCNN output
    to origin scale and filter with non-maximum suppression.

    :param bbox: a numpy array of the FasterRCNN output
    :param image_path: a string of image path
    :return: three list for best bound, class and score
    """
    w = image_size[0]
    h = image_size[1]
    #print(w,h,net_input_width,net_input_height)
    scale_w = net_input_width/w
    scale_h = net_input_height/h

    # cal predict box on the image src
    pbox = bbox.copy()
    pbox[:, 0] = (bbox[:, 0]) / scale_w
    pbox[:, 1] = (bbox[:, 1])  / scale_h
    pbox[:, 2] = (bbox[:, 2]) / scale_w
    pbox[:, 3] = (bbox[:, 3])  / scale_h
    return pbox


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_data_path", default="./result/dumpOutput_device0")
    parser.add_argument("--test_annotation", default="./coco2017_jpg.info")
    parser.add_argument("--det_results_path", default="./detection-results/")
    parser.add_argument("--net_out_num", default=2)
    parser.add_argument("--net_input_width", default=300)
    parser.add_argument("--net_input_height", default=300)
    parser.add_argument("--prob_thres", default=0.02)
    parser.add_argument("--ifShowDetObj", action="store_true", help="if input the para means True, neither False.")
    flags = parser.parse_args()
    # print(flags.ifShowDetObj, type(flags.ifShowDetObj))
    # generate dict according to annotation file for query resolution
    # load width and height of input images
    img_size_dict = dict()
    with open(flags.test_annotation)as f:
        for line in f.readlines():
            temp = line.split(" ")
            img_file_path = temp[1]
            img_name = temp[1].split("/")[-1].split(".")[0]
            img_width = int(temp[2])
            img_height = int(temp[3])
            img_size_dict[img_name] = (img_width, img_height, img_file_path)

    # read bin file for generate predict result
    bin_path = flags.bin_data_path
    det_results_path = flags.det_results_path
    os.makedirs(det_results_path, exist_ok=True)
    total_img = set([name[:name.rfind('_')]
                     for name in os.listdir(bin_path) if "bin" in name])
    for bin_file in sorted(total_img):
        path_base = os.path.join(bin_path, bin_file)
        # load all detected output tensor
        res_buff = []
        for num in range(1, flags.net_out_num + 1):
            if os.path.exists(path_base + "_" + str(num) + ".bin"):
                if num == 1:
                    buf = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="float32")
                    buf = np.reshape(buf, [200, 5])
                elif num == 2:
                    buf = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="int64")
                    buf = np.reshape(buf, [200, 1])
                res_buff.append(buf)
            else:
                print("[ERROR] file not exist", path_base + "_" + str(num) + ".bin")
        res_tensor = np.concatenate(res_buff, axis=1)
        current_img_size = img_size_dict[bin_file]
        #print("[TEST]---------------------------concat{} imgsize{}".format(len(res_tensor), current_img_size))
        #print(res_tensor)
        predbox = coco_postprocess(res_tensor, current_img_size, flags.net_input_width, flags.net_input_height)

        if flags.ifShowDetObj == True:
            imgCur = cv2.imread(current_img_size[2])

        det_results_str = ''
        det_results = []
        for idx, class_ind in enumerate(predbox[:,5]):
            if float(predbox[idx][4]) < float(flags.prob_thres):
                continue
            # skip negative class index
            if class_ind < 0 or class_ind > 80:
                continue

            class_name = CLASSES[int(class_ind)]
            det_results.append([class_name, str(predbox[idx][4]), predbox[idx][0], predbox[idx][1], predbox[idx][2], predbox[idx][3]])
            det_results_str += "{} {} {} {} {} {}\n".format(class_name, str(predbox[idx][4]), predbox[idx][0],
                                                            predbox[idx][1], predbox[idx][2], predbox[idx][3])
            if flags.ifShowDetObj == True:
                imgCur=cv2.rectangle(imgCur, (int(predbox[idx][0]), int(predbox[idx][1])), 
                                    (int(predbox[idx][2]), int(predbox[idx][3])), (0,255,0), 1)
                imgCur = cv2.putText(imgCur, class_name+'|'+str(predbox[idx][4]), 
                                    (int(predbox[idx][0]), int(predbox[idx][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                  # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
            
        if flags.ifShowDetObj == True:
            print(os.path.join(det_results_path, bin_file +'.jpg'))
            cv2.imwrite(os.path.join(det_results_path, bin_file +'.jpg'), imgCur, [int(cv2.IMWRITE_JPEG_QUALITY),70])

        det_results_file = os.path.join(det_results_path, bin_file + ".txt")
        with open(det_results_file, "w") as detf:
            detf.write(det_results_str)      
```
txt文件转json

```python
import glob
import os
import sys
import argparse
import mmcv

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

cat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''

def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def error(msg):
    print(msg)
    sys.exit(0)


def get_predict_list(file_path, gt_classes):
    dr_files_list = glob.glob(file_path + '/*.txt')
    dr_files_list.sort()

    bounding_boxes = []
    for txt_file in dr_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        lines = file_lines_to_list(txt_file)
        for line in lines:
            try:
                sl = line.split()
                if len(sl) > 6:
                    class_name = sl[0] + ' ' + sl[1]
                    scores, left, top, right, bottom = sl[2:]
                else:
                    class_name, scores, left, top, right, bottom = sl
                if float(scores) < 0.02:
                    continue
            except ValueError:
                error_msg = "Error: File " + txt_file + " wrong format.\n"
                error_msg += " Expected: <classname> <conf> <l> <t> <r> <b>\n"
                error_msg += " Received: " + line
                error(error_msg)

            # bbox = left + " " + top + " " + right + " " + bottom
            left = float(left)
            right = float(right)
            top = float(top)
            bottom = float(bottom)
            bbox = [left, top, right-left, bottom-top]
            #bounding_boxes.append({"image_id": int(file_id), "bbox": bbox,"score": float(scores), "category_id": 1+CLASSES.index(class_name)})
            bounding_boxes.append({"image_id": int(file_id), "bbox": bbox,"score": float(scores), "category_id": cat_ids[CLASSES.index(class_name)]})
        # sort detection-results by decreasing scores
        # bounding_boxes.sort(key=lambda x: float(x['score']), reverse=True)
    return bounding_boxes



if __name__ == '__main__':
    parser = argparse.ArgumentParser('mAp calculate')
    parser.add_argument('--npu_txt_path', default="detection-results",
                        help='the path of the predict result')
    parser.add_argument("--json_output_file", default="coco_detection_result")
    args = parser.parse_args()

    res_bbox = get_predict_list(args.npu_txt_path, CLASSES)
    mmcv.dump(res_bbox, args.json_output_file + '.json')
```

调用coco_eval.py评测map精度：

```shell
python3.7 mmdetection_coco_postprocess.py --bin_data_path=result/dumpOutput_device0 --prob_thres=0.02 --ifShowDetObj --det_results_path=detection-results --test_annotation=coco2017_ssd_jpg.info
python3.7 txt_to_json.py
python3.7 coco_eval.py --ground_truth /root/datasets/coco/annotations/instances_val2017.json
```

执行完后会打印出精度：

```shell
loading annotations into memory...
Done (t=0.88s)
creating index...
index created!
Loading and preparing results...
DONE (t=9.06s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=120.63s).
Accumulating evaluation results...
DONE (t=30.40s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.255
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.438
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.263
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.070
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.278
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.422
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.375
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.376
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.376
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.124
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.417
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.586
{'bbox_mAP': 0.255, 'bbox_mAP_50': 0.438, 'bbox_mAP_75': 0.263, 'bbox_mAP_s': 0.07, 'bbox_mAP_m': 0.278, 'bbox_mAP_l': 0.422, 'bbox_mAP_copypaste': '0.255 0.438 0.263 0.070 0.278 0.422'}
```


### 6.2 开源精度
[官网精度](http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20200307_174216.log.json)

```
{"mode": "val", "epoch": 24, "iter": 9162, "lr": 2e-05, "bbox_mAP": 0.256, "bbox_mAP_50": 0.438, "bbox_mAP_75": 0.263, "bbox_mAP_s": 0.068, "bbox_mAP_m": 0.278, "bbox_mAP_l": 0.422, "bbox_mAP_copypaste": "0.256 0.438 0.263 0.068 0.278 0.422"}
```
### 6.3 精度对比
om推理box map精度为0.255，开源box map50精度为0.256，精度下降0.1%，精度达标  

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  
-   **[基准性能数据](#72-基准性能数据)**  
-   **[性能对比](#73-性能对比)**  

### 7.1 npu性能数据
batch1的性能：
 测试npu性能要确保device空闲，使用npu-smi info命令可查看device是否在运行其它推理任务

在前面测试精度时，已经得到性能数据，运行
```shell
python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt
```
parse.py文件为，
```shell
import sys
import json
import re

if __name__ == '__main__':
    if sys.argv[1].endswith('.json'):
        result_json = sys.argv[1]
        with open(result_json, 'r') as f:
            content = f.read()
        tops = [i.get('value') for i in json.loads(content).get('value') if 'Top' in i.get('key')]
        print('om {} top1:{} top5:{}'.format(result_json.split('_')[1].split('.')[0], tops[0], tops[4]))
    elif sys.argv[1].endswith('.txt'):
        result_txt = sys.argv[1]
        with open(result_txt, 'r') as f:
            content = f.read()
        txt_data_list = [i.strip() for i in re.findall(r':(.*?),', content.replace('\n', ',') + ',')]
        fps = float(txt_data_list[7].replace('samples/s', '')) * 4
        print('310 bs{} fps:{}'.format(result_txt.split('_')[3], fps))
```
执行结果为
```shell
310 bs1 fps:56.9588
```
SSD mmdetection不支持多batch，故只在batch1上测试


### 7.2 基准性能数据
batch1性能：
onnx包含自定义算子，因此不能使用开源TensorRT测试性能数据，故在基准机器上使用pth在线推理测试性能数据

测评基准精度与性能：
```shell
wget http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20200307-a92d2092.pth
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git reset --hard a21eb25535f31634cef332b09fc27d28956fb24b
pip install -r requirements/build.txt
pip install -v -e .
mkdir data
ln -s /root/datasets/coco data/coco
python3 tools/test.py configs/ssd/ssd300_coco.py ../ssd300_coco_20200307-a92d2092.pth --eval bbox
```
```shell
loading annotations into memory...
Done (t=0.67s)
creating index...
index created!
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 5000/5000, 37.6 task/s, elapsed: 133s, ETA:     0s
Evaluating bbox...
Loading and preparing results...
DONE (t=8.12s)
creating index...
index created!
```

### 7.3 性能对比
310单卡4个device，benchmark测试的是一个device。基准一个设备相当于4个device，测试的是整个设备。benchmark时延是吞吐率的倒数，基准时延是吞吐率的倒数乘以batch。对于batch1，56.9588 > 37.6，即npu性能超过基准性能  
对于batch1，npu性能高于基准性能1.2倍，该模型放在benchmark/cv/detection目录下



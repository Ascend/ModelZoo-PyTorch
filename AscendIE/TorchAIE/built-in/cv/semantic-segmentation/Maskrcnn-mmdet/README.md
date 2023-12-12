# Maskrcnn-mmdet 模型推理指导

- [概述](#概述)
- [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
    - [获取源码](#获取源码)
    - [模型序列化](#模型序列化)
    - [准备数据集](#准备数据集)
    - [模型转换与推理](#模型转换与推理)
- [性能&精度](#性能精度)

----

## 概述
本项目为Maskrcnn模型在昇腾pytorch插件运行的样例，本样例展现了如何对Maskrcnn模型进行trace和导出TorchScript模型，以及在310P下利用昇腾pytorch插件运行Maskrcnn对COCO2017数据集进行测试，本模型代码基于mmdetection仓中的Maskrcnn修改

- 参考实现：  
    ```
    url = https://github.com/open-mmlab/mmdetection.git
    code_path = https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn  
    model_name=MaskRCNN
    ```
 

## 输入输出数据
- 模型输入  
  | 输入数据  | 数据类型  | 大小                        | 数据排布格式  |
  | -------- | -------- | --------------------------- | ------------ |
  | input    | FLOAT32 | batchsize x 3 x 1216 x 1216 |     NCHW     |

- 模型输出  
  | 输出数据  | 数据类型  | 大小               | 数据排布格式  |
  | -------- | -------- | ------------------ | ------------ |
  | output1  | FLOAT32  | 100 x 5            | ND           |
  | output2  | INT32    | 100                | ND           |
  | output3  | FLOAT32  | 100 x 80 X 28 X 28 | ND           |


## 推理环境准备

由于模型trace所需的环境与推理所需的环境冲突，因此需要构造两个环境。

其中(Torch 1.11.0环境)如下
| 配套                  | 版本             |
| --------------------- | ---------------- |
| Python                | 3.9              |
| torch                 | 1.11.0+cpu        |
| torchvision           | 0.12.0+cpu       |
| 芯片类型               | Ascend310P3      |

而(Torch 2.0.1环境)如下
| 配套                  | 版本             |
| --------------------- | ---------------- |
| Python                | 3.9              |
| torch                 | 2.0.1+cpu        |
| CANN                  | 7.0.RC1.alpha003 |
| Ascend-cann-torch-aie | 6.3rc2           |
| Ascend-cann-aie       | 6.3rc2           |
| 芯片类型               | Ascend310P3      |

两个环境需要分别安装依赖，如下
```bash
pip3 install -r requirements_111.txt --extra-index-url https://download.pytorch.org/whl/cpu   # (Torch 1.11.0环境)所需依赖
pip3 install -r requirements_201.txt --extra-index-url https://download.pytorch.org/whl/cpu   # (Torch 2.0.1环境)所需依赖
```

不同步骤需要采用特定的环境，具体如下：
- (Torch 1.11.0环境): [模型序列化](#模型序列化)、[准备数据集](#准备数据集)、[精度验证](#精度验证)
- (Torch 2.0.1环境): [模型推理与性能验证](#模型推理与性能验证)

## 快速上手

### 获取源码

获取开源仓源码
```bash
git clone -b v2.8.0 https://github.com/open-mmlab/mmdetection.git
cd mmdetection/
git apply --ignore-space-change --ignore-whitespace ../mmdet_maskrcnn.patch
cd ..
```

### 模型序列化(Torch 1.11.0环境)

1. 安装mmcv，前往[官方下载界面](https://download.openmmlab.com/mmcv/dist/cpu/torch1.11.0/index.html)选择mmcv_full-1.4.7-cp39-cp39-manylinux1_x86_64.whl并下载到本地，使用如下命令进行安装
    ```bash
    pip3 install mmcv_full-1.4.7-cp39-cp39-manylinux1_x86_64.whl
    ```

2. 编译自定义算子，使用如下命令
    ```bash
    cd mmdet_ops
    export PATH=/path/to/torch:${PATH}  # 此处/path/to/torch填写上述环境安装后的torch1.11.0的路径
    bash build.sh
    cd ..
    ```

3. 获取权重文件
    下载权重文件，使用如下命令
    ```bash
    cd mmdetection/
    wget https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth
    cd ..
    ```

4. 模型trace，使用如下命令
    ```
    python3 mmdet_trace.py --save_path ./mmdet_torch.ts
    ```
    参数说明：
    - --save_path：序列化TorchScript模型保存路径

### 准备数据集(Torch 1.11.0环境)

1. 获取原始数据集  
   本模型支持coco2017验证集。用户需自行获取数据集，建立data目录，将coco_val2017数据集放在该目录下。目录结构如下：
   ```
   coco/
   ├── annotations    //验证集标注信息       
   └── val2017        // 验证集文件夹
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。
    执行mmdet_preprocess脚本，完成预处理。
    ```bash
    python3 mmdet_preprocess.py \
        --image_src_path=/path/to/coco/val2017 \
        --bin_file_path=./val2017_bin \
        --info_file_path=./val2017.info \
        --model_input_height=1216 \
        --model_input_width=1216
    ```
    参数说明：
    - --image_src_path：数据原路径
    - --bin_file_path：数据保存路径
    - --info_file_path: 数据基本信息保存路径
    - --model_input_height：输入图像高
    - --model_input_width：输入图像宽

### 模型推理与性能验证(Torch 2.0.1环境)
1. 性能验证
    运行下述脚本进行性能验证
    ```bash
    python mmdet_infer.py \
        --torch_model  ./mmdet_torch.ts \
        --aie_model ./mmdet_aie.ts \
        --batch_size 1 \
        --device 0 \
        --infer_with_zeros
    ```
    参数说明：
    - --torch_model: 序列化TorchScript模型保存路径
    - --aie_model: 模型编译转换后的保存路径
    - --batch_size: 批次大小
    - --device: 设备号
    - --infer_with_zeros: 若设置则使用全0数据进行性能验证


2. 模型推理并保存推理结果
    运行下述脚本进行模型推理并保存推理结果
    ```bash
    python mmdet_infer.py \
        --torch_model  ./mmdet_torch.ts \
        --aie_model ./mmdet_aie.ts \
        --infer_data ./val2017_bin \
        --infer_result ./result \
        --batch_size 1 \
        --device 0 \
        --precision
    ```
    参数说明：
    - --torch_model: 序列化TorchScript模型保存路径
    - --aie_model: 模型编译转换后的保存路径
    - --infer_data: 模型推理数据路径
    - --infer_result: 推理结果保存路径
    - --batch_size: 批次大小
    - --device: 设备号
    - --precision: 若设置则保存推理结果

### 精度验证(Torch 1.11.0环境)
使用后处理脚本计算模型的各精度指标
```bash
python mmdet_postprocess.py \
    --bin_data_path ./result \
    --test_annotation ./val2017.info \
    --det_results_path ./precision_result
```
参数说明：
- --bin_data_path: 模型推理的结果路径
- --test_annotation: 推理数据的标签
- --det_results_path: 精度测试结果的保存路径


## 性能&精度

在310P设备上，本模型将与[基准模型](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/Maskrcnn-mmdet)进行比较。整体性能以及精度如下，性能精度均达标

| 芯片型号 | Batch Size | 数据集  | 基准精度 | 基准性能 | 精度 | 性能 |
| -------- | ---------- | ------ | ------- | -------- | ---- | ---- |
| 310P3    |      1     |  coco  | Bbox Map50: 0.59; Segm Map50: 0.554 | 11.3 fps | Bbox Map50: 0.59; Segm Map50: 0.554 | 12.8 fps |

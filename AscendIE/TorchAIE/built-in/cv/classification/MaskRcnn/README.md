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
Mask RCNN是一个实例分割（Instance segmentation）算法，它是一个多任务的网络，可以用来做“目标检测”、“目标实例分割”、“目标关键点检测”。本项目为Maskrcnn模型在昇腾pytorch插件运行的样例，本样例展现了如何对Maskrcnn模型进行trace和导出TorchScript模型，以及在310P下利用昇腾pytorch插件运行Maskrcnn对COCO2017数据集进行测试

- 参考实现：  
    ```
    url = https://github.com/facebookresearch/detectron2
    model_name=MaskRCNN
    ```
 

## 输入输出数据
- 模型输入  
  | 输入数据  | 数据类型  | 大小                        | 数据排布格式  |
  | -------- | -------- | --------------------------- | ------------ |
  | image    | FLOAT32 | batchsize x 3 x 1344 x 1344 |     NCHW     |

- 模型输出  
  | 输出数据  | 数据类型  | 大小               | 数据排布格式  |
  | -------- | -------- | ------------------ | ------------ |
  | output1  | FLOAT32  | 100 x 4            | ND           |
  | output2  | FLOAT32  | 100                | ND           |
  | output3  | FLOAT32  | 100 x 80 X 28 X 28 | ND           |
  | output4  | FLOAT32  | 100                | ND           |


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

1. 获取开源仓源码
    ```bash
    git clone https://github.com/facebookresearch/detectron2.git
    cd detectron2/
    git reset --hard 068a93a
    pip install -e .
    ```

2. 修改源码
    ```
    git apply --ignore-space-change --ignore-whitespace ../maskrcnn_detectron2.patch
    cd ..
    ```

### 准备数据集(Torch 1.11.0环境)

1. 获取原始数据集  
   本模型支持coco2017验证集。用户需自行获取数据集，建立data目录，将coco_val2017数据集放在该目录下。目录结构如下：
   ```
   datasets/
   └── coco/
       ├── annotations    //验证集标注信息       
       └── val2017        // 验证集文件夹
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。
    执行maskrcnn_detectron2_preprocess.py脚本，完成预处理。
    ```bash
    python3 maskrcnn_detectron2_preprocess.py \
        --image_src_path=./datasets/coco/val2017 \
        --bin_file_path=./val2017_bin \
        --model_input_height=1344 \
        --model_input_width=1344
    ```
    参数说明：
    - --image_src_path：数据原路径
    - --bin_file_path：数据保存路径
    - --info_file_path: 数据基本信息保存路径
    - --model_input_height：保存图像高
    - --model_input_width：保存图像宽

3. 生成数据集信息文件
    使用maskrcnn_detectron2_getInfo.py脚本，输入已经得到的二进制文件，输出生成二进制数据集的info文件。运行成功后，在当前目录中生成maskrcnn.info。之后JPG图片info文件生成,运行成功后，在当前目录中生成maskrcnn_jpeg.info。
    ```
    python3 maskrcnn_detectron2_getInfo.py \
        --file_type bin  \
        --file_path ./val2017_bin \
        --info_name  maskrcnn.info \
        --width 1344 \
        --height 1344
    python3 maskrcnn_detectron2_getInfo.py \
        --file_type jpg  \
        --file_path ./datasets/coco/val2017  \
        --info_name maskrcnn_jpeg.info
    ```
    参数说明：
    - --file_type: 数据格式
    - --file_path: 数据文件
    - --info_name: 保存的信息文件的名称


### 模型序列化(Torch 1.11.0环境)

1. 编译自定义算子，使用如下命令
    ```bash
    cd maskrcnn_detectron2_ops
    export PATH=/path/to/torch:${PATH}  # 此处/path/to/torch填写上述环境安装后的torch1.11.0的路径
    bash build.sh
    cd ..
    ```

2. 获取权重文件
    下载权重文件，使用如下命令
    ```bash
    wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/script/Faster_Mask_RCNNforPyTorch/zh/1.1/Faster_Mask_RCNN_for_PyTorch.zip
    unzip Faster_Mask_RCNN_for_PyTorch.zip
    ```
    预训练模型的路径为./Faster_Mask_RCNN_for_PyTorch/MaskRCNN.pth

3. 模型trace，使用如下命令
    ```
    python3 detectron2/tools/deploy/export_model.py \
        --config-file detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml \
        --output ./output \
        --export-method tracing \
        --format torchscript \
        MODEL.WEIGHTS ./Faster_Mask_RCNN_for_PyTorch/MaskRCNN.pth \
        MODEL.DEVICE cpu
    ```
    参数说明：
    - --config-file: 配置文件
    - --output: 输出路径

### 模型推理与性能验证(Torch 2.0.1环境)
1. 性能验证
    运行下述脚本进行性能验证
    ```bash
    python maskrcnn_detectron2_infer.py \
        --torch_model  ./output/model_torch.ts \
        --aie_model ./output/mmdet_aie.ts \
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
    python maskrcnn_detectron2_infer.py \
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
python maskrcnn_detectron2_postprocess.py \
    --bin_data_path ./result \
    --test_annotation ./val2017.info \
    --det_results_path ./precision_result
```
参数说明：
- --bin_data_path: 模型推理的结果路径
- --test_annotation: 推理数据的标签
- --det_results_path: 精度测试结果的保存路径


## 性能&精度

在310P设备上，本模型将与[基准模型](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/MaskRcnn)进行比较。整体性能以及精度如下，性能精度均达标

| 芯片型号 | Batch Size | 数据集  | 基准精度 | 基准性能 | 精度 | 性能 |
| -------- | ---------- | ------ | ------- | -------- | ---- | ---- |
| 310P3    |      1     |  coco  | AP@50:53.714 | 13.31 fps | AP@50:53.707 | 11.76 fps |

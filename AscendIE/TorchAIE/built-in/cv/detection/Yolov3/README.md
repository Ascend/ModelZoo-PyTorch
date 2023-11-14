# YOLOV3模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

   - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

- [可能遇到的问题](#ZH-CN_TOPIC_0000001172201574)

  ******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

YOLO是一个经典的目标检测网络，将目标检测作为回归问题求解。本文旨在提供基于推理引擎的Yolov3参考样例，使用了coco2017数据集，并测试了昇腾310P3芯片上的推理精度供参考，模型性能仍在持续优化中。


- 参考实现：

  ```shell
  url=https://github.com/ultralytics/yolov3
  branch=master
  commit_id=109527edf2c45dc25983455fbf2d9f76623543c9
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | images   | RGB_FP32 | batchsize x 3 x 640 x 640 | NCHW         |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下依赖

  **表 1**  版本配套表


  | 配套                                                            | 版本    | 环境准备指导                                                                                          |
  | --------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
  | 固件与驱动                                                      | 23.0.rc1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 7.0.RC1.alpha003 | -                                                                                                     |
  | Python                                                          | 3.9.11  | -                                                                                                     |
  | PyTorch                                                         | 2.0.1   | -                                                                                                     |
  | torchvison                                                      | 0.15.2  | -                                                                                                     |
  | torch_aie                                                       | 6.3rc2  | 
  | 说明：芯片类型：Ascend310P3 | \       | \   


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

      ```
       git clone https://github.com/ultralytics/yolov3.git
       cd yolov3
       git checkout v9.6.0  # 切换到所用版本
    
       获取推理部署代码https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/ACL_Pytorch/Yolov3_for_PyTorch
    
       将代码放到yolov3源码相应目录下：
       Yolov3_for_Pytorch
       └── common             放到yolov3下
         ├── util               模型/数据接口
         └── patch              v9.1/v9.6.0 模型修改
       ├── model.yaml         放到yolov3下 
       ├── pth2onnx.sh        放到yolov3下
       ├── onnx2om.sh         放到yolov3下
       ├── om_val.py          放到yolov3下
       └── requirements.txt   放到yolov3下
    ```

2. 安装依赖。

      ```
       pip install -r requirements.txt
      ```

## 准备数据集<a name="section183221994411"></a>

1. 本模型需要coco2017数据集，数据集结构如下。labels下载[地址](https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip)，验证集下载[地址](https://images.cocodataset.org/zips/val2017.zip)
   ```
    coco
     |-- LICENSE
     |-- README.txt
     |-- annotations
     |   `-- instances_val2017.json
     |-- images
     |   |-- train2017
     |   |-- val2017
     |-- labels
     |   |-- train2017
     |   `-- val2017
     |-- test-dev2017.txt
     |-- train2017.txt
     |-- val2017.cache
     `-- val2017.txt
   ```

2. 在yolov3源码根目录下新建coco文件夹，数据集放到coco里，文件结构如下：
   ```
    coco
    ├── val2017
      ├── 00000000139.jpg
      ├── 00000000285.jpg
      ……
      └── 00000581781.jpg
    ├── instances_val2017.json
    └── val2017.txt
   ```

3. val2017.txt中保存.jpg的相对路径，请自行生成该txt文件，文件内容实例如下：
   ```
    ./val2017/00000000139.jpg
    ./val2017/00000000285.jpg
    ……
    ./val2017/00000581781.jpg
   ```


## 模型推理<a name="section741711594517"></a>
1. 获取权重文件。
    ```
     wget https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3.pt
    ```


2. 生成trace模型(onnx, om, ts)
    ```
     bash pth2onnx.sh --tag 9.6.0 --model yolov3 --nms_mode nms_script
     source /usr/local/Ascend/ascend-toolkit/set_env.sh
     bash onnx2om.sh --tag 9.6.0 --model yolov3 --nms_mode nms_script --bs 4 --soc Ascend310P3
     export ASCENDIE_HOME="xxxx/xxxx/xxx" 设置该环境变量，为后续PT编译做准备
    ```
    
    atc命令参数说明（参数见onnx2om.sh）：
    ```
     --model：ONNX模型文件
     --output：输出的OM模型
     --framework：5代表ONNX模型
     --input_format：输入数据的格式
     --input_shape：输入数据的shape
     --soc_version：处理器型号
     --log：日志级别
     --compression_optimize_conf：模型量化配置
    ```
    compression_optimize_conf 的使用说明参考该[链接](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FCANNCommunityEdition%2F600alpha003%2Finfacldevg%2Fatctool%2Fatlasatc_16_0084.html)


3. 保存编译优化模型（非必要，可不执行。后续执行的推理脚本包含编译优化过程）

    ```
     python export_torch_aie_ts.py --batch_size=4
    ```
   命令参数说明（参数见onnx2om.sh）：
    ```
     --torch_script_path：编译前的ts模型路径
     --soc_version：处理器型号
     --batch_size：模型batch size
     --save_path：编译后的模型存储路径
    ```


4. 执行推理脚本

    将pt_val.py放在./yolov3下，model_pt.py放在./yolov3/common/util下
     ```
      cd yolov3
      # 执行推理(yolov3.torchscript.pt为未编译优化前的ts模型)
      python pt_val.py --tag 9.6.0 --model=yolov3.torchscript.pt --batch_size=4
     ```
   命令参数说明（参数见onnx2om.sh）：
    ```
     --data_path：验证集数据根目录，默认"coco"
     --ground_truth_json：标注数据路径
     --tag：yolov3标记
     --soc_version：处理器型号
     --model：输入模型路径
     --need_compile：是否需要进行模型编译（若使用export_torch_aie_ts.py输出的模型，则不用选该项）
     --batch_size：模型batch size
     --img_size：推理size（像素）
     --cfg_file：模型参数配置文件路径，默认model.yaml
     --device_id：硬件编号
     --single_cls：是否视为单类数据集
    ```
   使用的model.yaml配置：
    ```
     # parameters
     img_size: [640, 640]  # height, width
     class_num: 80  # number of classes
     conf_thres: 0.001  # object confidence threshold, conf>0.1 for nms_op
     iou_thres: 0.6  # IOU threshold for NMS
        
     # anchors
     anchors:
       - [10,13, 16,30, 33,23]  # P3/8
       - [30,61, 62,45, 59,119]  # P4/16
       - [116,90, 156,198, 373,326]  # P5/32
     stride: [8, 16, 32]
         ```
# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>



| 芯片型号 | Batch Size   | 数据集    |
| --------- | ---------------- | ---------- |
|    Ascend310P3       |       4    |   coco2017   |


**表 2** yolov3模型精度

| 类型 | 配置   | 精度    |
| --------- | ---------------- | ------------|
|Average Precision  (AP)| @[ IoU=0.50:0.95 , area=   all , maxDets=100 ] | 0.445|
|Average Precision  (AP)| @[ IoU=0.50      , area=   all , maxDets=100 ] | 0.655|
|Average Precision  (AP)| @[ IoU=0.75      , area=   all , maxDets=100 ] | 0.492|
|Average Precision  (AP)| @[ IoU=0.50:0.95 , area= small , maxDets=100 ] | 0.294|
|Average Precision  (AP)| @[ IoU=0.50:0.95 , area=medium , maxDets=100 ] | 0.494|
|Average Precision  (AP)| @[ IoU=0.50:0.95 , area= large , maxDets=100 ] | 0.564|
|Average Recall     (AR)| @[ IoU=0.50:0.95 , area=   all , maxDets=  1 ] | 0.348|
|Average Recall     (AR)| @[ IoU=0.50:0.95 , area=   all , maxDets= 10 ] | 0.572|
|Average Recall     (AR)| @[ IoU=0.50:0.95 , area=   all , maxDets=100 ] | 0.616|
|Average Recall     (AR)| @[ IoU=0.50:0.95 , area= small , maxDets=100 ] | 0.457|
|Average Recall     (AR)| @[ IoU=0.50:0.95 , area=medium , maxDets=100 ] | 0.660|
|Average Recall     (AR)| @[ IoU=0.50:0.95 , area= large , maxDets=100 ] | 0.748|


**表 3** 模型推理性能

| Soc version | Batch Size | Dataset | Performance    |
| -------- | ---------- | ---------- |----------------|
| 310P3    | 4          | coco2017 | 153.085 ms/pic |


# 可能遇到的问题<a name="ZH-CN_TOPIC_0000001172201574"></a>
1. AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor' 解决方法参考[该链接](https://zhuanlan.zhihu.com/p/545926241)
2. libGL.so.1: cannot open shared object file: No such file or directory 解决方法参考[该链接](https://zhuanlan.zhihu.com/p/498478991)
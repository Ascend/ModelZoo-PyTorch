# Mask RCNN 模型推理指导

- [概述](#概述)
    - [输入输出数据](#输入输出数据)
- [推理环境](#推理环境)
- [快速上手](#快速上手)
    - [获取源码](#获取源码)
    - [准备数据集](#准备数据集)
    - [模型转换](#模型转换)
    - [推理验证](#推理验证)
- [性能&精度](#性能精度)

----
# 概述

Mask RCNN是一个实例分割（Instance segmentation）算法，它是一个多任务的网络，可以用来做“目标检测”、“目标实例分割”、“目标关键点检测”。

+ 论文  
    [论文](https://arxiv.org/pdf/1703.06870.pdf)  
    Kaiming He Georgia Gkioxari Piotr Dollar Ross Girshick

+ 参考实现：  
    https://github.com/facebookresearch/detectron2

## 输入输出数据
+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | image | FLOAT32 | NCHW | 1,3,1344,1344 | 

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output1      |  FLOAT32   | NCHW          | 100,4        |
    | output2      |  FLOAT32   | NCHW          | 100        |
    | output3      |  FLOAT32   | NCHW          | 100,80,28,28        |
    | output4      |  FLOAT32   | NCHW          | 100        |

----
# 推理环境

- 该模型推理所需配套的软件如下：

    | 配套      | 版本    | 环境准备指导 |
    | --------- | ------- | ---------- |
    | 固件与驱动 | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN      | 6.0.RC1 | -          |
    | Python    | 3.7.5   | -          |
    
    说明：请根据推理卡型号与 CANN 版本选择相匹配的固件与驱动版本。


----
# 快速上手

## 安装

- 安装推理过程所需的依赖
    ```bash
    pip3 install -r requirements.txt
    ```
- 获取源码
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

- 修改源代码
    ```
    patch -p1 < ../maskrcnn_detectron2.diff
    cd ..
    ```

- 找到自己conda环境中的pytorch安装地址
    ```
    # 打开/root/anaconda3/envs/自己创建的环境名称/lib/python3.7/site-packages/torch/onnx/utils.py文件
    搜索_check_onnx_proto(proto)并注释代码，添加pass代码，后保存并退出。
    # _check_onnx_proto(proto)
    pass       
    ```

## 准备数据集

1. 获取原始数据集  
    本模型推理项目使用 coco 数据集验证模型精度，请在自行下载，并在当前目录创建datasets文件夹放置COCO数据集，其中annotations目录存放coco数据集的instances_val2017.json，val2017目录存放coco数据集的5000张验证图片。。   


2. 数据预处理  
    执行前处理脚本将原始数据转换为OM模型输入需要的bin/npy文件。
    ```bash
    python3 maskrcnn_pth_preprocess_detectron2.py --image_src_path=/root/dataset/coco//val2017 --bin_file_path=val2017_bin --model_input_height=1344 --model_input_width=1344
    ```
    其中"image_src_path"表示处理前原数据集的地址，"bin_file_path"表示生成数据集的文件夹名称

    运行后，将会得到如下形式的文件夹：

    ```
    ├── val2017_bin
    │    ├──000000000139.bin
    │    ├──......     	 
    ```

3 生成数据集信息文件
    使用get_info.py脚本，输入已经得到的二进制文件，输出生成二进制数据集的info文件。
    运行成功后，在当前目录中生成maskrcnn.info。
    之后JPG图片info文件生成,运行成功后，在当前目录中生成maskrcnn_jpeg.info。
    ```
    python3 get_info.py --file_type bin  --file_path ./val2017_bin --info_name  maskrcnn.info --width 1344 --height 1344
    python3 get_info.py --file_type jpg  --file_path ./datasets/coco/val2017  --info_name maskrcnn_jpeg.info
    ```

## 模型转换

1. PyTroch 模型转 ONNX 模型  

    从源码包中获取训练后的权重文件[maskrcnn.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/script/Faster_Mask_RCNNforPyTorch/zh/1.1/Faster_Mask_RCNN_for_PyTorch.zip)。

    然后执行执行以下命令生成 ONNX 模型：
    ```
    python3 detectron2/tools/deploy/export_model.py --config-file detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml --output         ./output --export-method tracing --format onnx MODEL.WEIGHTS MaskRCNN.pth MODEL.DEVICE cpu
    ```
    参数说明：
     + --config-file: 参数配置文件路径
     + --output: 生成ONNX模型的保存路径
     + --export-method: 导出模型的模式
     + --format: 导出文件的格式
     + MODEL.WEIGHTS: 权重文件路径
     + MODEL.DEVICE: 硬件设备


2. ONNX 模型转 OM 模型  

    step1: 查看NPU芯片名称 \${chip_name}
    ```bash
    npu-smi info
    ```
    例如该设备芯片名为 310P3，回显如下：
    ```
    +-------------------+-----------------+------------------------------------------------------+
    | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
    | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
    +===================+=================+======================================================+
    | 0       310P3     | OK              | 15.8         42                0    / 0              |
    | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
    +===================+=================+======================================================+
    | 1       310P3     | OK              | 15.4         43                0    / 0              |
    | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
    +===================+=================+======================================================+
    ```

    step2: ONNX 模型转 OM 模型
    ```bash
    # 配置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    chip_name=310P3  # 根据 step1 的结果设值
    
    # 执行 ATC 进行模型转换
    atc --model=model_py1.8.onnx --framework=5 --output=maskrcnn_detectron2_npu --input_format=NCHW --input_shape="0:4,3,1344,1344" --out_nodes="Cast_1673:0;Gather_1676:0;Reshape_1667:0;Slice_1706:0" --log=error --soc_version=Ascend${chip_name}
    ```

   参数说明：
    + --framework: 5代表ONNX模型
    + --model: ONNX模型路径
    + --input_shape: 模型输入数据的shape
    + --input_format: 输入数据的排布格式
    + --output: OM模型路径，无需加后缀
    + --log：日志级别
    + --soc_version: 处理器型号
    + --out_nodes: 输出节点名
 


## 推理验证

1. 对数据集推理  
    安装ais_bench推理工具。请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。完成安装后，执行以下命令预处理后的数据进行推理。
    ```bash
    python3 -m ais_bench \
        --model ./maskrcnn_detectron2_npu.om \
        --input ./MaskRcnn/val2017_bin \
        --output ./results \
        --outfmt BIN \
        --batchsize 1
    ```
    参数说明：
    + --model OM模型路径
    + --input 存放预处理后数据的目录路径
    + --output 用于存放推理结果的父目录路径
    + --outfmt 推理结果文件的保存格式
    + --batchsize 模型每次输入bin文件的数量


2. 性能验证  
    对于性能的测试，需要注意以下三点：
    + 测试前，请通过`npu-smi info`命令查看NPU设备状态，请务必在NPU设备空闲的状态下进行性能测试。
    + 为了避免测试过程因持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps，反映模型在单位时间（1秒）内处理的样本数。
    ```bash
    python3 -m ais_bench --model ./maskrcnn_detectron2_npu.om --batchsize 1
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python3 maskrcnn_pth_postprocess_detectron2.py --bin_data_path=./results/****/ --test_annotation=maskrcnn_jpeg.info --det_results_path=./ret_npuinfer/ --net_out_num=4 --net_input_height=1344 --net_input_width=1344 --ifShowDetObj
    ```
    参数说明：
    + --bin_data_path: 存放推理结果的目录路径
    + --test_annotation: 标签文件路径
    + --det_results_path: 后处理结果路径。
    + --net_out_num: 推理保存节点数。
    + --ifShowDetObj: 是否显示后处理结果。
    运行成功后，程序会将各top1~top5的正确率记录在 result_batch_size1.json 文件中，可执行以下命令查看：
    ```
    |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
    |:------:|:------:|:------:|:------:|:------:|:------:|
    | 32.720 | 53.714 | 35.030 | 17.926 | 36.745 | 43.239 |
    INFO:detectron2.evaluation.coco_evaluation:Per-category bbox AP: 
    | category      | AP     | category     | AP     | category       | AP     |
    |:--------------|:-------|:-------------|:-------|:---------------|:-------|
    | person        | 49.031 | bicycle      | 24.471 | car            | 37.608 |
    | motorcycle    | 33.499 | airplane     | 51.994 | bus            | 54.955 |
    | train         | 52.013 | truck        | 26.909 | boat           | 20.703 |
    | traffic light | 20.208 | fire hydrant | 58.406 | stop sign      | 59.202 |
    | parking meter | 42.271 | bench        | 17.270 | bird           | 29.241 |
    | cat           | 57.822 | dog          | 52.798 | horse          | 51.655 |
    | sheep         | 40.412 | cow          | 41.296 | elephant       | 55.590 |
    | bear          | 63.353 | zebra        | 59.513 | giraffe        | 58.204 |
    | backpack      | 11.383 | umbrella     | 29.291 | handbag        | 8.690  |
    | tie           | 25.047 | suitcase     | 27.523 | frisbee        | 54.205 |
    | skis          | 16.496 | snowboard    | 24.286 | sports ball    | 40.624 |
    | kite          | 34.385 | baseball bat | 17.272 | baseball glove | 26.036 |
    | skateboard    | 39.424 | surfboard    | 28.258 | tennis racket  | 38.406 |
    | bottle        | 30.770 | wine glass   | 26.648 | cup            | 33.797 |
    | fork          | 19.167 | knife        | 10.838 | spoon          | 8.791  |
    | bowl          | 34.025 | banana       | 18.077 | apple          | 15.500 |
    | sandwich      | 27.840 | orange       | 26.399 | broccoli       | 19.014 |
    | carrot        | 15.487 | hot dog      | 25.517 | pizza          | 44.300 |
    | donut         | 34.920 | cake         | 24.026 | chair          | 18.909 |
    | couch         | 32.853 | potted plant | 18.867 | bed            | 33.891 |
    | dining table  | 20.198 | toilet       | 45.900 | tv             | 48.958 |
    | laptop        | 49.705 | mouse        | 47.144 | remote         | 20.852 |
    | keyboard      | 40.347 | cell phone   | 28.539 | microwave      | 43.172 |
    | oven          | 25.629 | toaster      | 16.271 | sink           | 27.563 |
    | refrigerator  | 42.297 | book         | 10.381 | clock          | 45.611 |
    | vase          | 30.711 | scissors     | 25.720 | teddy bear     | 36.962 |
    | hair drier    | 0.000  | toothbrush   | 12.242 |                |        |
    ```


----
# 性能&精度

在310P设备上，OM模型的精度为  **{AP@50:53.714}**，当batchsize设为1时模型性能最优，达 22.06 fps。

| 芯片型号   | BatchSize | 数据集      | 精度            | 性能       |
| --------- | --------- | ----------- | --------------- | --------- |
|Ascend310P3| 1         | COCO2017  | AP@50:53.714 | 22.06 fps |

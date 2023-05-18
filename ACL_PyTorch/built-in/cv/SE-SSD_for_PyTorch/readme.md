# SE-SSD模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

SE-SSD（Self-Ensembling Single-Stage Object Detector）是一种基于自集成单级目标检测器的室外点云三维目标检测方法。该模型包含一对teacher和student SSD模型，作者设计了一个有效的IOU-based匹配策略来过滤teacher预测的soft目标，并使用一致性损失来使student的预测和teacher预测保持一致。该模型关注点是利用软目标和硬目标以及制定的约束来共同优化模型，而不引入额外计算量。

- 参考实现：

    ```
    url = https://github.com/Vegeta2020/SE-SSD
    mode_name = SE-SSD
    ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

    | 输入数据    | 数据类型 | 大小                      | 数据排布格式 |
    | ----------- | -------- | ------------------------- | ------------ |
    | voxels      | FLOAT32  | batchsize x 20000 x 5 x 4 | ND           |
    | coordinates | FLOAT32  | batchsize x 20000 x 4     | ND           |
    | num_points  | FLOAT32  | batchsize x 20000         | ND           |

- 输出数据

    | 输出数据      | 数据类型 | 大小                       | 数据排布格式 |
    | ------------- | -------- | -------------------------- | ------------ |
    | box_preds     | FLOAT32  | batchsize x 200 x 176 x 14 | ND           |
    | cls_preds     | FLOAT32  | batchsize x 200 x 176 x 2  | ND           |
    | dir_cls_preds | FLOAT32  | batchsize x 200 x 176 x 4  | ND           |
    | iou_preds     | FLOAT32  | batchsize x 200 x 176 x 2  | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动:

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 23.0.RC1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.3.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.13.1  | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 |         |                                                              |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码

   ```bash
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git  # 克隆仓库的代码
   git checkout master                                      # 切换到对应分支
   cd ACL_PyTorch/built-in/cv/SE-SSD_for_PyTorch            # 切换到模型的代码仓目录
   ```

2. 安装依赖

   ```bash
   pip3 install -r requirements.txt
   ```

3. 从源码编译、安装SpConv库

    1. 获取[C++ boost库](https://www.boost.org/)的头文件
        ```bash
        wget https://boostorg.jfrog.io/artifactory/main/release/1.73.0/source/boost_1_73_0.tar.bz2
        tar -xjvf boost_1_73_0.tar.bz2
        cp -r boost_1_73_0/boost/ /usr/include/
        ```

    2. 获取SpConv v1.2.1源码

        ```bash
        git clone -b v1.2.1 https://github.com/traveller59/spconv.git
        ```

    3. 获取合适版本的pybind11

        ```bash
        cd spconv/third_party
        git clone https://github.com/pybind/pybind11.git
        cd pybind11
        git reset --hard 085a29436a8c472caaaf7157aa644b571079bcaa
        cd ../..
        ```

    4. 安装SpConv库
        ```bash
        pip install -e .
        cd ..
        ```

4. 获取、修改模型源码

    ```bash
    git clone https://github.com/Vegeta2020/SE-SSD.git
    cd SE-SSD
    git apply ../source_code_modification.patch
    cd ..
    ```

5. 将模型源码路径添加至环境变量`PYTHONPATH`

    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/SE-SSD
    ```

## 准备数据集<a name="section183221994411"></a>

本模型使用KITTI物体检测(object detection)数据集，请从[KITTI数据集官网](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)依次下载下列文件：
- Left color images of object data set (12 GB)
- Velodyne point clouds (29 GB)
- Camera calibration matrices of object data set (16 MB)
- Training labels of object data set (5 MB)

> 若下载速度缓慢，可自行寻找公开资源下载。

下载后的各个压缩包内的目录结构如下：
```
data_object_image_2
├── testing
│   └── image_2
└── training
    └── image_2

data_object_velodyne
├── testing
│   └── velodyne
└── training
    └── velodyne

data_object_calib
├── testing
│   └── calib
└── training
    └── calib

data_object_label_2
└── training
    └── label_2
```

请将各个文件的子目录 `training` 和 `testing` 合并放至某个目录下，如下所示：

 ```bash
${data_root}
  ├── testing
  │   ├── calib
  │   ├── image_2
  │   └── velodyne
  └── training
      ├── calib
      ├── image_2
      ├── label_2
      └── velodyne
 ```

然后执行预处理脚本:

```bash
python3 preprocessing.py --data_root ${data_root} --val --save_dir ./preprocessed_data
```

- 参数说明：
    - --data_root：数据集文件主目录
    - --val：处理验证集数据
    - --save_dir: 预处理结果输出目录

## 模型推理<a name="section741711594517"></a>

1. 模型转换
    
    1. 从官方代码仓获取模型权重：[下载地址](https://drive.google.com/file/d/1M2nP_bGpOy0Eo90xWFoTIUkjhdw30Pjs/view?usp=sharing)

        获得权重文件`se-ssd-model.pth`，保存在当前目录下。

    2. 执行转换ONNX脚本：

        ```bash
        python3 export_onnx.py
        ```

        获得`se-ssd.onnx`文件。

     3. 使用ATC工具将ONNX模型转OM模型。

        1. 配置环境变量

            ```bash
            source /usr/local/Ascend/ascend-toolkit/set_env.sh
            ```

        2. 执行命令查看芯片名称（$\{chip\_name\}）

            ```
            npu-smi info
            #该设备芯片名为Ascend310P3 （自行替换）
            回显如下：
            +-------------------|-----------------|------------------------------------------------------+
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

        3. 执行ATC指令

            ```bash
            # 设置batch size，目前仅支持batch size=1
            bs=1

            atc \
                --framework=5 \
                --model=se-ssd.onnx \
                --input_shape="voxels:$bs,20000,5,4;coordinates:$bs,20000,4;num_points:$bs,20000" \
                --output=se-ssd_bs$bs \
                --soc_version=Ascend${chip_name} \
                --log=error
            ```

            - 参数说明：
                - --model：为ONNX模型文件
                - --framework：5代表ONNX模型
                - --output：输出的OM模型
                - --input_shape：输入数据的大小
                - --log：日志级别
                - --soc_version：处理器型号

            ---
            执行成功后获得`se-ssd_bs1.om`文件。

2. 开始推理验证

   1. 安装ais_bench推理工具

        请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理

        ```bash
        python3 -m ais_bench --model se-ssd_bs$bs.om --input preprocessed_data/val/voxels/,preprocessed_data/val/coordinates/,preprocessed_data/val/num_points/ --output . --output_dir result_om --device 0
        ```

        - 参数说明：
            - --model：om文件路径
            - --input：输入文件
            - --output：输出目录
            - --output_dir: 输出的文件夹名称
            - --device：NPU设备编号

        推理的结果输出在当下目录的`result_om`中。
        
    3. 精度验证

        执行后处理脚本，查看精度，保存最终结果：

        ```bash
        python3 postprocessing.py --data_root ${data_root} --model_output_dir ./result_om
        ```

        - 参数说明：
            - --data_root：数据集文件主目录
            - --model_output_dir：推理输出目录

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

模型验证集推理精度：

```
AP_11: car AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:98.72, 90.10, 89.57
bev  AP:90.61, 88.75, 88.17
3d   AP:90.21, 86.25, 79.21
aos  AP:98.67, 89.87, 89.17

AP_40: car AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:99.57, 95.58, 93.16
bev  AP:96.70, 92.15, 89.74
3d   AP:93.74, 86.18, 81.67
aos  AP:99.52, 95.28, 92.70
```

参考精度：

```
AP_11: car AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:98.72, 90.10, 89.57
bev  AP:90.61, 88.76, 88.18
3d   AP:90.21, 86.25, 79.22
aos  AP:98.67, 89.86, 89.16

AP_40: car AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:99.57, 95.58, 93.16
bev  AP:96.70, 92.15, 89.74
3d   AP:93.75, 86.18, 83.50
aos  AP:99.52, 95.28, 92.69
```

推理性能：
| 芯片型号 | 模型   | Batch Size | 数据集                 | 性能(fps) |
|:--------:|:------:|:----------:|:----------------------:|:---------:|
| 310P3    | SE-SSD | 1          | KITTI object detection | 0.4388    |
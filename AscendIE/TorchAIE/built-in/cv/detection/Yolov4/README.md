
# YOLOV4模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

   - [输入输出数据](#ZH-CN_TOPIC_0000001126281702)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

YOLO是一个经典的物体检查网络，将物体检测作为回归问题求解。YOLO训练和推理均是在一个单独网络中进行。基于一个单独的end-to-end网络，输入图像经过一次inference，便能得到图像中所有物体的位置和其所属类别及相应的置信概率。YOLOv4在YOLOv3的基础上做了很多改进，其中包括近几年来最新的深度学习技巧，例如Swish、Mish激活函数，CutOut和CutMix数据增强方法，DropPath和DropBlock正则化方法，也提出了自己的创新，例如Mosaic（马赛克）和自对抗训练数据增强方法，提出了修改版本的SAM和PAN，跨Batch的批归一化（BN），共五大改进。所以说该文章工作非常扎实，也极具创新。

[Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao.YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv 2018.Thu, 23 Apr 2020 02:10:02 UTC (3,530 KB)](https://arxiv.org/abs/2004.10934)

- 参考实现：

  ```shell
  url=https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
  branch=master
  commit_id=78ed10cc51067f1a6bac9352831ef37a3f842784
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- | -------- |---------------------------| ------------ |
  | images   | RGB_FP32 | batchsize x 3 x 608 x 608 | NCHW         |

- 输出数据

  | 输出数据    | 数据类型 | 大小         | 数据排布格式 |
  | ----------- | -------- |------------| ------------ |
  | Reshape_216 | FLOAT32  | 3x85x76x76 | NCHW         |
  | Reshape_203 | FLOAT32  | 3x85x38x38 | NCHW         |
  | Reshape_187 | FLOAT32  | 3x85x19x19 | NCHW         |

# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                        | 版本          | 环境准备指导                                                                                          |
|-------------------------------------------|-------------| ----------------------------------------------------------------------------------------------------- |
| Ascend-cann-torch-aie                     | -           |  |
| CANN                                      | 7.0.0       | -                                                                                                     |
| Python                                    | 3.9.0       | -                                                                                                     |
| PyTorch                                   | 2.0.1       | -                                                                                                     |
| Ascend-cann-aie                           | -           |                                                                                                       |
| 芯片类型                                      | Ascend310P3 |                                                                                                       |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \           | \                                                                                                     |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 安装CANN包

 ```
 chmod +x Ascend-cann-toolkit_6.3.RC2.alpha002_linux-aarch64.run 
./Ascend-cann-toolkit_6.3.RC2.alpha002_linux-aarch64.run --install
 ```
下载Ascend-cann-torch-aie和Ascend-cann-aie得到run包和压缩包
## 安装Ascend-cann-aie
 ```
  chmod +x Ascend-cann-aie_6.3.T200_linux-aarch64.run
  ./Ascend-cann-aie_6.3.T200_linux-aarch64.run --install
  cd Ascend-cann-aie
  source set_env.sh
  ```
## 安装Ascend-cann-torch-aie
 ```
 tar -zxvf Ascend-cann-torch-aie-6.3.T200-linux_aarch64.tar.gz
 pip3 install torch-aie-6.3.T200-linux_aarch64.whl
 ```

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

    ```
    git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git
    cd pytorch-YOLOv4
    git reset --hard a65d219f9066bae4e12003bd7cdc04531860c672
    patch -p2 < ../yolov4.patch
    cd ..
    ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型需要coco2014数据集，数据集下载[地址](https://cocodataset.org/)

   数据集结构如下
   ```
    coco
     ├── annotations
     │   └── instances_minival2014.json
     └── images
         └── val2014

   ```

2. 生成数据集info文件。
   执行parse_json.py脚本。
   ```
   cd ${dataset}
   mkdir ground-truth
   python3 ${project_path}/parse_json.py --dataset=${dataset}
   ```
   - 参数说明
      - ${dataset}：数据集绝对路径，到annotations同级级目录即可。例：/data/datasets/coco。
      - ${project_path}：yolov4项目路径。

   执行成功后，在当前目录下生成coco2014.name和coco_2014.info文件以及标签文件夹ground-truth。


## 模型推理<a name="section741711594517"></a>
1. 模型转换。

   首先将pytorch-YOLOv4源码仓中的yolov4.pth权重文件转换为ts模型，然后使用AIE推理引擎推理。

   1. 获取权重文件。

      从 [官方仓库链接](https://github.com/Tianxiaomo/pytorch-YOLOv4) 中下载 `yolov4.pth`,放入pytorch-YOLOV4源码仓目录下。

   2. 导出ts模型文件。

        ```shell
        mv ./yolov4_pth2ts.py ./pytorch-YOLOv4
        cd pytorch-YOLOv4
        python yolov4_pth2ts.py yolov4.pth 1 80 608 608
        cd ..
        mv pytorch-YOLOv4/yolov4_1_3_608_608.ts .
        ```

        - 参数说明：

          - yolov4.pth：权重文件。
          - 1 80 608 608：输入信息，分别为batch_size,class_num,H,W。

2. 开始推理验证。

   1. 使用AIE推理引擎执行推理。
      ```shell
      python3 yolov4_aie_infer.py \
              --ts_model_path=./yolov4_1_3_608_608.ts \
              --origin_jpg_path=${dataset}/images \
              --src_path=${dataset}/coco_2014.info \
              --coco_class_names=${dataset}/coco2014.names
      ```

      -   参数说明：

          -   --ts_model_path:上一步中导出的ts模型。
          -   --origin_jpg_path：原始数据集图片路径。
          -   --coco_class_names：数据集类别名称文件names。
          -   --src_path：数据集info文件。

         > **说明：**
         > 执行后会在当前目录下创建两个文件夹，分别是'bin_output'内含AIE推理结果的bin文件，与'detection-results'内含后处理后的结果。

   2. 精度验证。

      使用脚本map_calculate.py脚本计算输出特征图map值：
      ```shell
      python3 map_calculate.py --label_path=${dataset}/ground-truth/ --npu_txt_path=./detection-results/
      ```
      -   参数说明：

          -   --label_path：标签路径
          -   --npu_txt_path：推理特征图路径

   3. 性能验证。
       
        使用脚本calculate_cost_static.py脚本计算推理性能：
        ```shell
        python3 calculate_cost_static.py --ts_path=./yolov4_1_3_608_608.ts --batch_size=1
        ```
      
        -   参数说明：

            -   --ts_path：模型路径
            -   --batch_size：batch_size
            -   --optimization_level(可选):是否使用aoe优化

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>
调用ACL接口推理计算，性能参考下列数据。

| Batch Size   | 数据集 | 精度 | 310P3 |
| ---------------- | ---------- | ---------- | --------------- |
|       1    |   coco2014         |     60.3%       |   152.80              |
|       4       |   coco2014        |            |           170.82      |
|       8       |    coco2014       |            |     171.15            |
|      16       |     coco2014      |    60.3%        |      170.97           |
|   32          |    coco2014      |            |     170.36            |
|   64          |    coco2014      |            |        167.28         |
|  |  | **最优性能** | **171.15** |


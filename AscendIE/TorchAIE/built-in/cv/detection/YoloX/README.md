# YOLOX模型-AIE推理引擎部署指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [模型部署](#ZH-CN_TOPIC_0000001126281700)

  - [安装依赖](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

******

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

YOLOX是基于往年对YOLO系列众多改进而产生的目标检测模型，其采用无锚方式，并应用了解耦头和领先的标签分配策略 SimOTA.其在众多数据集中均获得了最佳结果。

- 参考实现：

  ```
  url=https://github.com/Megvii-BaseDetection/YOLOX
  commit_id=6880e3999eb5cf83037e1818ee63d589384587bd
  code_path=ACL_PyTorch/contrib/cv/detection/YOLOX
  model_name=YOLOX
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 640 x 640 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小    |
  | -------- | -------- | ------- |
  | output   | FLOAT32  | 1 x 8400 x 85 |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                            | 版本    | 环境准备指导                                                                                          |
  | --------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
  | 固件与驱动                                                      | 23.0.rc1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 7.0.RC1.alpha003 | -                                                                                                     |
  | Python                                                          | 3.9.11  | -                                                                                                     |
  | PyTorch                                                         | 2.0.1   | -                                                                                                     |
  | torch_aie                                                       | 6.3rc2  | 
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |

# 模型部署<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 安装依赖<a name="section4622531142816"></a>

1. 获取源码

   ```shell
   git clone https://github.com/Megvii-BaseDetection/YOLOX
   cd YOLOX
   git reset 6880e3999eb5cf83037e1818ee63d589384587bd --hard
   patch -p1 < ../yolox_coco_evaluator.patch
   pip install -v -e .  # or  python3 setup.py develop
   cd ..
   ```

2. 安装需要的Python Library

   ```shell
   apt-get install libprotobuf-dev protobuf-compiler
   apt-get install libgl1-mesa-glx
   pip install -r requirements.txt
   ```

3. 设置环境变量

   执行环境中推理引擎安装路径下的环境变量设置脚本

   ```shell
   source {aie}/set_env.sh
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   请参考开源代码仓方式获得[COCO2017数据集](https://cocodataset.org/)，并根据需要置于服务器上（如 `datasets_path=/data/dataset/coco`），val2017目录存放coco数据集的验证集图片，annotations目录存放coco数据集的instances_val2017.json，文件目录结构如下：

   ```
    data
    ├── dataset
    │   ├── coco
    │   │   ├── annotations
    │   │   ├── val2017
   ```

## 模型推理<a name="section741711594517"></a>

### 1. 模型转换

   使用PyTorch将模型权重文件.pth转换为torchscript文件

   1. 获取权重文件

       我们利用官方的PTH文件进行验证，官方PTH文件可从原始开源库中获取，我们需要[yolox_x.pth](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth)文件,请将其放在与README.md文件同一目录内。

   2. 导出torchscript文件

      ```shell
      cd YOLOX
      python tools/export_torchscript.py \
         --output-name ../yolox.torchscript.pt \
         -n yolox-x \
         -c ../yolox_x.pth
      cd ..
      ```

      获得yolox.torchscript.pt文件。

      + 参数说明
         + `--output-name`：输出文件名称
         + `-n`：模型名称
         + `-c`：权重文件路径

### 2. 开始推理验证

   1. 执行命令查看芯片名称（$\{chip\_name\}）。

      ```shell
      npu-smi info
      #该设备芯片名为Ascend310P3 （在下一步中赋值给soc_version环境变量）
      回显如下：
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

   2. 对原生ts文件执行torch_aie编译，导出NPU支持的ts文件

      ```shell
      soc_version="Ascend310P3" # User-defined
      python yolox_export_torch_aie_ts.py \
         --torch-script-path ./yolox.torchscript.pt \
         --batch-size 1 \
         --save-path ./yoloxb1_torch_aie.pt \
         --soc-version ${soc_version}
      ```
   + 参数说明
      + `--torch-script-path`：原生ts文件路径
      + `--batch-size`：用户自定义的batch size
      + `--save-path`：AIE编译后的ts文件保存路径
      + `--soc-version`：NPU型号

   3. 执行推理并验证精度与性能
   
      <em>COCO2017数据集需要约5分钟完成所有推理任务，请耐心等待。</em>

      ```shell
      python yolox_eval.py \
         --dataroot /data/dataset/coco \
         --batch 1 \
         --ts ./yoloxb1_torch_aie.pt
      ```

      - 参数说明：
         -   --dataroot：COCO数据集的路径，同上
         -   --batch：用户自定义的batch size
         -   --ts：AIE编译后的ts文件路径

      运行成功后将打印该模型在NPU推理结果的精度信息与性能信息。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

基于推理引擎完成推理计算，精度与性能可参考下列数据：

| Soc version | Batch Size | Dataset | Accuracy ||
| ----------  | ---------- | ---------- | ---------- | ---------- |
| 310P3精度   | 1  | coco2017 | Average Precision(IoU=0.50:0.95): 0.498 | Average Precision(IoU=0.50): 0.678 |
| 310P3精度   | 4  | coco2017 | Average Precision(IoU=0.50:0.95): 0.498 | Average Precision(IoU=0.50): 0.678 |
| 310P3精度   | 8  | coco2017 | Average Precision(IoU=0.50:0.95): 0.498 | Average Precision(IoU=0.50): 0.678 |
| 310P3精度   | 20 | coco2017 | Average Precision(IoU=0.50:0.95): 0.498 | Average Precision(IoU=0.50): 0.678 |

| Soc version | Batch Size | Dataset | Performance |
| -------- | ---------- | ---------- | ---------- |
| 310P3    | 1          | coco2017 | 35.65 ms/pic |
| 310P3    | 4          | coco2017 | 33.09 ms/pic |
| 310P3    | 8          | coco2017 | 33.63 ms/pic |
| 310P3    | 20         | coco2017 | 32.67 ms/pic |

# FAQ
1. 若遇到类似报错：ImportError: /lib/aarch64-linux-gnu/libGLdispatch.so.0: cannot allocate memory in static TLS block

   解决方法：
   export LD_PRELOAD=$LD_PRELOAD:{报错信息中的路径}

# Citations 数据集引用

```
https://cocodataset.org/#termsofuse
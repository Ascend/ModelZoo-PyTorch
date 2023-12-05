# MobileNetV3-推理指导

- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理性能&精度](#模型推理性能&精度)

******


# 概述
MobileNetV3，是谷歌在2019年3月21日提出的网络架构。网络架构是基于NAS实现的MnasNet（效果比MobileNetV2好），结合资源受限的NAS（platform-aware NAS）与NetAdapt两种技术进行网络搜索。
MobileNetV3引入了MobileNetV1的深度可分离卷积，MobileNetV2的具有线性瓶颈的倒残差结构和基于squeeze and excitation结构的轻量级注意力模型(SE)，
同时使用了一种新的激活函数h-swish(x)，在ImageNet数据集上取得了很好的结果。

- 版本说明：
  ```
  url=https://github.com/xiaolai-sqlai/mobilenetv3
  commit_id=adc0ca87e1dd8136cd000ae81869934060171689
  model_name=MobileNetV3
  ```

### 输入输出数据

- 输入数据

  | 输入数据 |  数据类型   |            大小             | 数据排布格式 | 
  |:-------:|:-------------------------:|:------:|:----------:| 
  | input    | FLOAT16 | batchsize x 3 x 224 x 224 |  NCHW  |


- 输出数据

  | 输出数据  |  数据类型   |        大小        | 数据排布格式 |
  |:-------:|:----------------:|:----------------:|:----------:|
  | output | FLOAT16 | batchsize x 1000 |   ND       |


# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

| 配套                                                     | 版本      | 环境准备指导                                                 |
| ------------------------------------------------------- |---------| ------------------------------------------------------------ |
| 固件与驱动                                                | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                    | 6.0.RC1 | -                                                            |
| Python                                                  | 3.7.5   | -                                                            |
| PyTorch                                                 | 1.10.1  | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |


# 快速上手

## 获取源码

1. 获取`Pytorch`源码  
   ```
   git clone https://github.com/xiaolai-sqlai/mobilenetv3
   cd mobilenetv3
   git reset --hard adc0ca87e1dd8136cd000ae81869934060171689
   mkdir -p output imagenet # 新建output文件夹，作为模型结果的默认保存路径
   ```

2. 获取`OM`推理代码  
   将推理部署代码放在`mobilenetv3`源码仓目录下。
   ```
    MobileNetV3_for_PyTorch
    ├── data               放到mobilenetv3下
    ├── pth2onnx.py        放到mobilenetv3下
    ├── atc.sh             放到mobilenetv3下
    ├── om_val.py          放到mobilenetv3下
    └── requirements.txt   放到mobilenetv3下
   ```   
   
3. 安装依赖  
   ```
   pip3 install -r requirements.txt
   ```


## 准备数据集
- 该模型使用`ImageNet 2012`数据集进行精度评估，下载[ImageNet 2012数据集的验证集](https://image-net.org)， 原始数据集下载解压后直接是图像，没有按照类别区分，可参考[该链接](https://zhuanlan.zhihu.com/p/370799616)进行预处理，处理后的数据放到新建的`imagenet`文件下，文件结构如下：
   ```
    imagenet
    └── val
      ├── n01440764
        ├── ILSVRC2012_val_00000293.jpeg
        ├── ILSVRC2012_val_00002138.jpeg
        ……
        └── ILSVRC2012_val_00048969.jpeg
      ├── n01443537
      ……
      └── n15075141
    └── val_label.txt
   ```


## 模型推理
### 1 模型转换  
将模型权重文件`.pth`转换为`.onnx`文件，再使用`ATC`工具将`.onnx`文件转为离线推理模型`.om`文件。

1. 获取权重文件  
   权重文件已包含在`git clone`下载下来的`pytorch`源码里。

2. 导出`ONNX`模型  
  运行`pth2onnx.py`导出onnx模型，默认保存在`output`文件夹下，可设置参数`--dynamic`支持导出动态batch的onnx，`--simplify`简化导出的onnx。
   ```
   python3 pth2onnx.py --pth=mbv3_small.pth.tar --onnx=mbv3_small.onnx --batch=1 --dynamic
   ```

3. 使用`ATC`工具将`ONNX`模型转为`OM`模型  
   3.1 配置环境变量  
   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
   > **说明：**  
     该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

   3.2 执行命令查看芯片名称（得到`atc`命令参数中`soc_version`）
   ```
   npu-smi info
   #该设备芯片名为Ascend310P3 （自行替换）
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

   3.3 执行ATC命令  
   运行`atc.sh`导出`OM`模型，默认保存在`output`文件夹下。
   ```
   bash atc.sh --model mbv3_small --bs 1 --soc Ascend310P3
   ```
      - `atc`命令参数说明（参数见`atc.sh`）：
        -   `--model`：ONNX模型文件
        -   `--framework`：5代表ONNX模型
        -   `--output`：输出的OM模型
        -   `--input_format`：输入数据的格式
        -   `--input_shape`：输入数据的shape
        -   `--log`：日志级别
        -   `--soc_version`：处理器型号
        -   `--input_fp16_nodes`：指定输入数据类型为FP16的输入节点名称
        -   `--output_type`：指定网络输出数据类型或指定某个输出节点的输出类型

    
### 2 开始推理验证

1. 安装`ais_bench`推理工具  
   请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

2. 执行推理 & 精度验证  
   运行`om_val.py`推理OM模型，得到模型精度结果。
   ```
   python3 om_val.py --dataset=imagenet --checkpoint=output/mbv3_small_bs1.om --batch=1
   ```

3. 性能验证  
   可使用`ais_bench`推理工具的纯推理模式验证不同`batch_size`的`OM`模型的性能，参考命令如下：
   ```
   python3 -m ais_bench --model output/mbv3_small_bs${bs}.om --loop 1000 --batchsize ${bs}
   ```
   其中，`bs`为模型`batch_size`。

# 模型推理性能&精度

调用ACL接口推理计算，性能&精度参考下列数据。

|   芯片型号   | Batch Size |    数据集     |         精度           |      性能      |
|:-----------:|:----------:|:----------:|:-----------------------:|:------------:|
| Ascend310P3 |     1      |  ImageNet  | 65.094/Top1 85.432/Top5 | 2769.64 fps  |
| Ascend310P3 |     4      |  ImageNet  | 65.094/Top1 85.432/Top5 | 7743.44 fps  |
| Ascend310P3 |     8     |  ImageNet  | 65.094/Top1 85.432/Top5 | 10965.24 fps  |
| Ascend310P3 |     16      |  ImageNet  | 65.094/Top1 85.432/Top5 | 14284.28 fps  |
| Ascend310P3 |     32      |  ImageNet  | 65.094/Top1 85.432/Top5 | 15442.12 fps  |
| Ascend310P3 |     64     |  ImageNet  | 65.079/Top1 85.417/Top5 | 14863.88 fps |


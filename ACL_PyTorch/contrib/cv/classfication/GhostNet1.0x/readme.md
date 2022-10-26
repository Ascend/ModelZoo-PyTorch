# GhostNet1.0x模型-推理指导


- [概述](#概述)

- [推理环境准备](#推理环境准备)

- [快速上手](#快速上手)

  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)

- [模型推理性能](#模型推理性能)


# 概述<a name="概述"></a>

 GhostNet是华为诺亚方舟实验室提出的一个新型神经网络结构，其中的Ghost Module和深度分离卷积就很类似，不同之处在于先进行PointwiseConv，后进行DepthwiseConv，另外增加了DepthwiseConv的数量，包括一个恒定映射。

- 参考实现：

  ```
  url=https://github.com/huawei-noah/CV-Backbones.git
  branch=master
  commit_id=5a06c87a8c659feb2d18d3d4179f344b9defaceb
  model_name=GhostNet1.0x
  ```

  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

## 输入输出数据<a name="输入输出数据"></a>

- 输入数据

| 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
| -------- | -------- | ------------------------- | ------------ |
| input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

| 输出数据 | 大小     | 数据类型 | 数据排布格式 |
| -------- | -------- | -------- | ------------ |
| output1  | 1 x 1000 | FLOAT32  | ND           |


# 推理环境准备\[所有版本\]<a name="推理环境准备"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本      | 环境准备指导                                                 |
| ------------------------------------------------------------ |---------| ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.8.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="快速上手"></a>


1. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="准备数据集"></a>

1. 获取原始数据集（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/root/datasets/imagenet/val与/root/datasets/imagenet/val_label.txt。

2. 数据预处理

   1.预处理脚本imagenet_torch_preprocess.py

    2.如验证的数据集在目录imageNet下
    ```
    ├── imageNet    
           └── val       // 验证集文件夹
    ├── val_label.txt    //验证集标注信息      
    ```
    执行预处理脚本，生成数据集预处理后的bin文件，将原始数据（.jpg）转化为二进制文件（.bin）存放在prep_dataset文件目录下
    ```
    python3.7 imagenet_torch_preprocess.py ghostnet /root/datasets/imageNet/val ./prep_dataset
    ```

## 模型推理<a name="模型推理"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       1.下载pth权重文件[GhostNet预训练pth权重文件](https://github.com/huawei-noah/CV-Backbones/raw/master/ghostnet_pytorch/models/state_dict_73.98.pth)或是执行下述命令获取
       ```
       wget http://github.com/huawei-noah/CV-Backbones/raw/master/ghostnet_pytorch/models/state_dict_73.98.pth
       ```
       2.下载开源仓
       ```
       git clone https://github.com/huawei-noah/CV-Backbones.git
       cd CV-Backbones
       git reset --hard 5a06c87a8c659feb2d18d3d4179f344b9defaceb
       cd ..
       ```

   2. 导出onnx文件。

      1. 使用ghostnet_pth2onnx.py导出onnx文件。

         运行ghostnet_pth2onnx.py脚本。

         ```
         python3.7 ghostnet_pth2onnx.py state_dict_73.98.pth ghostnet.onnx
         ```

         获得ghostnet.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称。

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

      3. 执行ATC命令。
         ```
          atc --framework=5 --model=./ghostnet.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=ghostnet_bs1 --log=debug --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --insert\_op\_conf=aipp\_resnet34.config:  AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。

           运行成功后生成ghostnet_bs1.om模型文件。



2. 开始推理验证。

   a.  使用ais-infer工具进行推理。
      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   b.  执行推理。
      ```
      python ${ais_infer_path}/ais_infer.py --model ./ghostnet_bs1.om --input ./prep_dataset/ --output ./ --outfmt NPY --batchsize 1
      ```
      - 参数说明：
           - model：为OM模型文件
           - input：为数据路径
           - output：输出推理结果
           - outfmt：输出结果的格式
           - input_shape：输入数据的shape
           - batchsize：模型接受的bs大小
        ...
      推理后的输出默认在当前目录result下。
      >**说明：** 
      >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

   c.  精度验证。

      调用imagenet_acc_eval.py脚本推理结果与label比对，可以获得精度结果数据，显示在控制台。
      ```
      python3.7 imagenet_acc_eval.py ./lcmout/2022_xx_xx-xx_xx_xx/sumary.json /home/HwHiAiUser/dataset/imageNet/val_label.txt
      ```
      - 参数说明：
         - val_label.txt：为标签数据
         - sumary.json：为生成结果文件

   d.  性能验证。

      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
       python3.7 ${ais_infer_path}/ais_infer.py --model=${om_model_path} --loop=20 --batchsize=${batch_size}
      ```


# 模型推理性能&精度<a name="模型推理性能"></a>

调用ACL接口推理计算，性能参考下列数据。
以下为不同芯片型号和bs下的精度和性能表现
GhostNet在310上的精度复现与性能表现如下表1，表2所示

表1-精度对比

| 芯片型号 | top1 | top5 |
|:------|:---:|:------:|
| 310 | 0.7398 | 0.9146 |
| 310P | 0.7398 | 0.9146 |

表2-性能对比

| Batch Size | 310 | 310P | t4 | 310P/310| 310P/t4|
|:------|:------:|:------:|:------:|:------:|:------:|
| 1 | 	1348.024 | 1502.4291 | 219.2172| 1.1145 | 6.8536|
| 4 | 2233.9991 | 2317.6152 | 701.0072 | 1.0374| 3.3061|
| 8 | 2463.9302 | 3739.9555| 1032.52| 1.5179 | 3.6222|
| 16 | 2624.8900 | 3438.7936| 924.992| 1.3101| 3.7176|
| 32 | 2689.0490 | 3020.9916| 447.872| 1.1234| 6.7452|

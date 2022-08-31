# ResNeXt-50模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  ******




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>
ResNeXt50是一种用于图像分类的卷积神经网络，这个模型的默认输入尺寸是224×224，有三个通道。通过利用多路分支的特征提取方法，提出了一种新的基于ResNet残差模块的网络组成模块，并且引入了一个新的维度cardinality。该网络模型可以在于对应的ResNet相同的复杂度下，提升模型的精度（相对于最新的ResNet和Inception-ResNet)）同时，还通过实验证明，可以在不增加复杂度的同时，通过增加维度cardinality来提升模型精度，比更深或者更宽的ResNet网络更加高效。



- 参考实现：

  ```
    url=https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    branch=master
    commit_id=78ed10cc51067f1a6bac9352831ef37a3f842784
  ```




  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 1 x 1000 | FLOAT32  | ND           |




# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.6.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>



1. 安装依赖。

   ```
   pip3 install -r requirment.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
在官网http://image-net.org/下载ILSVRC2012数据集。
使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。
图片与标签分别存放在/root/datasets/imagenet/val与/root/datasets/imagenet/val_label.txt。


2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行preprocess_resnext50_pth.py和get_info.py脚本，完成预处理。

   ```
   python3 preprocess_resnext50_pth.py dataset/ImageNet/val_union/ pre_bin
   python3 get_info.py bin pre_bin resnext50_val.info 224 224
   ```




## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       下载链接：https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth

   2. 导出onnx文件。

      1. 使用resnext50_pth2onnx.py导出onnx文件。

         运行resnext50_pth2onnx.py脚本。

         ```
         python3.7 resnext50_pth2onnx.py ./resnext50_32x4d-7cdf4587.pth ./resnext50.onnx
         ```

         获得XXX.onnx文件。



   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

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
         atc --model=./resnext50.onnx --framework=5 --output=resnext50_bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --log=info --soc_version=Ascend${chip_name}
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

           运行成功后生成<u>***XX.om***</u>模型文件。



2. 开始推理验证。

a.  使用ais-infer工具进行推理。

   执行命令增加工具可执行权限，并根据OS架构选择工具

   ```
   chmod u+x 
   ```

b.  执行推理。

    ```
    python ais_infer.py --model "/home/zzy/resnext50_bs16_310.om" --input /home/zzy/prep_bin/  --output "/home/zzy/output/" --outfmt  TXT  --batchsize 16 
    ```

    -   参数说明：

        -   --model：模型类型。
        -   --input：模型需要的输入，支持bin文件和目录。
        -   --output：推理结果输出路径。默认会建立日期+时间的子文件夹保存输出结果 如果指定output_dirname 将保存到output_dirname的子文件夹下。
        -   --outfmt：输出数据的格式，默认”BIN“，可取值“NPY”、“BIN”、“TXT”。
        -   --batchsize ：模型batchsize，默认为1 。
		

        推理后的输出默认在当前目录output下。

      
        >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

c.  精度验证。

    调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

    ```
    python3.7 vision_metric_ImageNet.py output/2022_08_04-17_21_14/ ./val_label.txt ./ result.json
    ```

    output/2022_08_04-17_21_14/ ：为生成推理结果所在路径  
    
    val_label.txt：为标签数据
    
    result.json：为生成结果文件

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| 310       |  1               | ILSVRC2012 |77.61%      | 642.98          |
| 310       |  16              | ILSVRC2012 |77.61%      | 2070.524        |
| 310p      |  1               | ILSVRC2012 |77.62%      | 1413.46         |
| 310p      |  32              | ILSVRC2012 |77.62%      | 4012.15         |
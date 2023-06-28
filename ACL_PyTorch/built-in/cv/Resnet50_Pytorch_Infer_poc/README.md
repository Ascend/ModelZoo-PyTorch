# Resnet50-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Resnet是残差网络(Residual Network)的缩写,该系列网络广泛用于目标分类等领域以及作为计算机视觉任务主干经典神经网络的一部分，典型的网络有resnet50, resnet101等。Resnet网络的证明网络能够向更深（包含更多隐藏层）的方向发展。


- 参考实现：

  ```
  url=https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
  ```

  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone https://gitee.com/ascend/ModelZoo-PyTorch.git       # 克隆仓库的代码
  cd /ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer             # 切换到模型的代码仓目录
  git checkout master        # 切换到对应分支
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



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 23.0.RC2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.3.203 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | >1.5.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>



1. 安装依赖。

   ```
   pip3 install -r requirment.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型使用ImageNet 50000张图片的验证集，请前往ImageNet官网下载数据集

    ```
    ├── ImageNet
    |   ├── val
    |   |    ├── ILSVRC2012_val_00000001.JPEG
    │   |    ├── ILSVRC2012_val_00000002.JPEG
    │   |    ├── ......
    |   ├── val_label.txt
    ```

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   执行imagenet_torch_preprocess.py脚本，完成预处理。

   ```
   python3 imagenet_torch_preprocess.py --src_path ./ImageNet/val --save_path ./prep_dataset
   ```
   每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成prep_dataset二进制文件夹

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       前往[Pytorch官方文档](https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet50)下载对应权重，参考下载权重如下：
   
      [权重](https://download.pytorch.org/models/resnet50-0676ba61.pth)
   
   2. 导出onnx文件。
   
      1. 使用pth2onnx.py导出onnx文件。
   
         运行pth2onnx.py脚本。
   
         ```
         python3 pth2onnx.py ./resnet50-0676ba61.pth
         ```
   
         获得resnet50_official.onnx文件。

      2. 模型量化。

         请访问[昇腾社区](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/devtools/auxiliarydevtool/atlasamctonnx_16_0012.html)，跟据安装指南安装amct_onnx工具。

         生成校准数据，进行模型量化
         ```shell
         python3 imagenet_torch_preprocess.py --src_path ./ImageNet/val --save_path ./amct_bin --amct
         amct_onnx calibration --model resnet50_official.onnx --save_path amct_model/resnet50 --input_shape "input:1,3,224,224" --data_dir amct_bin --data_types "float32" --calibration_config quant.cfg
         ```
         在amct_model文件夹下得到resnet50_deploy_model.onnx。
   
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
         #该设备芯片名为Ascend310P3 （请根据实际芯片填入）
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

         ```shell
         atc --model=amct_model/resnet50_deploy_model.onnx --framework=5 --output=resnet50_bs${bs} --input_format=NCHW --input_shape="input:${bs},3,224,224" --enable_small_channel=1 --log=error --soc_version=Ascend${chip_name} --insert_op_conf=aipp_resnet50.config
         ```
         备注：Ascend${chip_name}请根据实际查询结果填写
   
         - 参数说明：
         
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --insert\_op\_conf:  AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。
         
           运行成功后生成`resnet50_bs${bs}.om`模型文件。
         

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。
      ```shell
      python3 -m ais_bench --model ./resnet50_bs${bs}.om --input ./prep_dataset/ --output ./ --output_dirname result
      ```

      - 参数说明：   
         - --model：模型地址
         -  --input：预处理完的数据集文件夹
         -  --output：推理结果保存地址
         -  --output_dirname: 推理结果保存文件夹
         
      运行成功后会在result下生成推理输出的bin文件。


   3. 精度验证。

      统计推理输出的Top 1-5 Accuracy
      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据。
      ```
      python3 vision_metric_ImageNet.py ./result ./ImageNet/val_label.txt
      ```
      - 参数说明：
         - val_label.txt：为标签数据

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```shell
      python3 -m ais_bench --model=./resnet50_bs${bs}.om --loop=50 
      ```

      -   参数说明：

          -   --model：om模型路径。
          -   --loop：推理次数。

      `${bs}`表示不同batch的om模型。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| 310P3 | 1  | ImageNet | top-1: 75.31% <br>top-5: 92.51% | 2113 |
| 310P3 | 4  | ImageNet | top-1: 75.31% <br>top-5: 92.51% | 5452 |
| 310P3 | 8  | ImageNet | top-1: 75.31% <br>top-5: 92.51% | 6993 |
| 310P3 | 16 | ImageNet | top-1: 75.31% <br>top-5: 92.51% | 7289 |
| 310P3 | 32 | ImageNet | top-1: 75.31% <br>top-5: 92.51% | 7444 |
| 310P3 | 64 | ImageNet | top-1: 75.31% <br>top-5: 92.51% | 7529 |

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
  url=https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection
  ```

  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone https://gitee.com/ascend/ModelZoo-PyTorch.git       # 克隆仓库的代码
  cd /ACL_PyTorch/built-in/cv/Resnet50_mlperf             # 切换到模型的代码仓目录
  git checkout master        # 切换到对应分支
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FLOAT32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 |
  | -------- | -------- | -------- |
  | output   | batchsize   | INT64    |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.3.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>



1. 安装依赖。

   1. 安装基础环境
    ```bash
    pip3 install -r requirements.txt
    ```
    说明：某些库如果通过此方式安装失败，可使用pip单独进行安装。

    2. 安装量化工具

        参考[AMCT(ONNX)](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/70RC1alpha001/developmenttools/devtool/atlasamctonnx_16_0004.html)主页安装量化工具。


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

   执行preprocess.py脚本，完成预处理。

   ```
   python3 preprocess.py \
       --src_path ./ImageNet/val \
       --save_path ./prep_dataset
   ```

   - 参数说明
        - --src_path: 测试数据集地址
        - --save_path: 生成预处理数据bin文件地址

   每个图像对应生成一个npy文件。运行成功后，在当前目录下生成prep_dataset npy文件夹

## 模型推理<a name="section741711594517"></a>

### 1. 模型转换。

   1. 获取.onnx模型文件
   
      下载.pb文件转换为.onnx文件

      - 获取pb文件。

         前往[mlperf官方文档](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection)下载对应[pb文件](https://zenodo.org/record/2535873/files/resnet50_v1.pb)：
        
        ```
        wget https://zenodo.org/record/2535873/files/resnet50_v1.pb
        ```
   
   
      - 导出onnx文件。
   
         在和pb文件同目录下，使用convert_to_onnx.sh导出onnx文件。
   
         运行convert_to_onnx.sh脚本。
   
         ```
         bash convert_to_onnx.sh
         ```
   
         获得resnet50.onnx文件。
   2. 模型量化 
   
       在量化前，我们先生成校验数据，以确保量化后模型精度不会损失：
       ```bash
        python3 preprocess.py \
            --src_path ./ImageNet/val \
            --save_path ./amct_bin_data \
            --amct
        ```
       - 参数说明
        - --src_path: 测试数据集地址
        - --save_path: 生成校验数据bin文件地址
        - --amct: 说明是生成amct量化的校验数据

        然后使用`amct`工具，对ONNX模型进行量化，以进一步提升模型性能：
        ```bash
        amct_onnx calibration \
            --model ./models/resnet50.onnx \
            --save_path ./models/resnet50_quant \
            --input_shape "dummy_input:64,3,224,224" \
            --data_dir "./amct_bin_data/" \
            --data_types "float32" \
            --calibration_config ./quant.cfg
        ```
        - 参数说明
          - --model: onnx模型
          - --save_path: 保存量化后onnx模型文件地址
          - --input_shape: 模型输入shape
          - --data_dir: 校验数据
          - --data_types: 数据类型
          - -calibration_config: 量化配置文件
        
        量化后的模型存放路径为 `models/resnet50_quant_deploy_model.onnx`。


   2. 使用ATC工具将ONNX模型转OM模型。
   
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
   
         ```
         atc --model=models/resnet50_quant_deploy_model.onnx \
         --framework=5 \
         --output=resnet50_bs64 \
         --input_format=NCHW \
         --input_shape="dummy_input:64,3,224,224" \
         --log=error \
         --insert_op_conf=./aipp_resnet50.conf \
         --enable_small_channel=1 \
         --soc_version=Ascend${chip_name} 
         ```
   
            备注：Ascend${chip_name}请根据实际查询结果填写    
         - 参数说明：
         
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input_format：输入数据的格式。
           -   --input_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc_version：处理器型号。
           -   --enable_small_channel:是否使能small_channel优化。
           -   --insert_op_conf：使能AIPP，使用该参数后，则输入数据类型为uint8。
         
           运行成功后生成resnet50_bs64.om模型文件。
           
           

### 2.开始推理验证。

a.  安装ais_bench推理工具。

   请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

b.  执行推理。
   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh 
   python3 -m ais_bench --model ./resnet50_bs64.om --input ./prep_dataset/ --output ./ --output_dirname result
   ```

   - 参数说明：   
      - --model：模型地址
      -  --input：预处理完的数据集文件夹
      -  --output：推理结果保存地址
      -  --output_dirname: 推理结果保存文件夹
        
   运行成功后会在result目录下生成推理输出的bin文件。


c.  精度验证。

统计推理输出的Accuracy
调用脚本与数据集标签val_map.txt比对，可以获得Accuracy数据，结果保存在result.json中。
   ```
   python3 accuracy.py ./result ./val_map.txt ./ result.json
   ```
   - 参数说明
     - result：为推理结果保存文件夹   
     - val_map.txt：为标签数据
     - result.json：为生成结果文件

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| 310P3 | 1 | ImageNet | 76.38% | 4089|
| 310P3 | 2 | ImageNet | 76.38% | 6323|
| 310P3 | 4 | ImageNet | 76.38% | 9492|
| 310P3 | 8 | ImageNet | 76.38% | 9403|
| 310P3 | 16 | ImageNet | 76.38% | 7748|
| 310P3 | 32 | ImageNet | 76.38% | 7020|
| 310P3 | 64 | ImageNet | 76.38% | 12141|


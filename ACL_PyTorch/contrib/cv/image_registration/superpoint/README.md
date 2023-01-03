# SuperPoint模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

  - [输入输出数据](#section4622531142820)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994678)
  - [模型推理](#section183221994799)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


    ******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

我们设计了一种称为SuperPoint的全卷积神经网络架构，该架构对全尺寸图像进行操作，并在单次前向传递中产生伴随固定长度描述符的兴趣点检测。该模型有一个单一的共享编码器来处理和减少输入图像的维数。在编码器之后，该架构分成两个解码器“头”，它们学习任务特定权重——一个用于兴趣点检测，另一个用于感兴趣点描述。大多数网络参数在两个任务之间共享，这与传统系统不同，传统系统首先检测兴趣点，然后计算描述符，并且缺乏跨两个任务共享计算和表示的能力。



- 参考实现：

  ```
  url= https://github.com/eric-yyjau/pytorch-superpoint
  commit_id= 5eb75d74df27c07f6e7311df8f167e2a9c01a798
  model_name= SuperPointNet_gauss2
  ```


## 输入输出数据<a name="section4622531142820"></a>


- 输入数据

  | 输入数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- |---------------------------| ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 1 x 240 x 320 | NCHW         |


- 输出数据

  | 输出数据    | 数据类型                | 大小 | 数据排布格式 |
  | -------- |---------------------------|--------| ------------ |
  | output1 | FLOAT32 | 1 x 256 x 30 x 40  | NHCW   |
  | output2 | FLOAT32  | 1 x 65 x 30 x 40  | NHCW   |



# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.6.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>
1. 获取源码。
   ```
   git clone https://github.com/eric-yyjau/pytorch-superpoint.git
   cd pytorch-superpoint
   git reset --hard 5eb75d74df27c07f6e7311df8f167e2a9c01a798
   patch -p3 < sp.patch
   ```

2. 安装依赖  
   ```
   pip install -r requirements.txt 
   ```

## 准备数据集<a name="section183221994678"></a>（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
1. 获取原始数据集

   数据集名称: [HPatches](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz)

   数据集存放在datasets文件夹中

   ```
   目录结构
   datasets/ ($DATA_DIR)
   `-- HPatches
   |   |-- i_ajuntament
   |   `-- ...
   ```

2. 数据预处理

   数据预处理在代码主目录将原始数据集转换为模型输入的数据。
   ```
   python superpoint_preprocess.py --img_path $img_path --result_path $result_path
   ```
   详见下表

    | 参数        | 说明                                          |
    | ----------- | --------------------------------------------- |
    | img_path    | 数据集路径(./datasets/hpatches/)              |
    | result_path | 数据预处理得到的bin文件保存位置(./pre_result) |

## 模型推理<a name="section183221994799"></a>。

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       [下载链接](https://github.com/eric-yyjau/pytorch-superpoint/tree/master/logs/superpoint_coco_heat2_0/checkpoints) 采用名称为superPointNet_170000_checkpoint.pth.tar的权重文件

   2. 导出onnx文件。

      在代码主目录使用superpoint_pth2onnx.py导出onnx文件。
      
      ```
      python superpoint_pth2onnx.py --model_path=$model_path --batch_size=$batch_size
      ```
      
      | 参数           | 说明                                                 
      |----------------------------------------------------|---------------------------|                               
      | model_path    | pth模型路径(./superPointNet_170000_checkpoint.pth.tar) |
      | batch_size    | batch大小(1、4、8、16、32、64)                            |

       获得sp-{batch_size}.onnx文件。
      
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

         退出虚拟环境，进行onnx到om的转换
         ```
         atc --framework=5 \
             --model=superpoint.onnx \
             --output=sp \
             --input_format=NCHW \
             --input_shape="image:1,1,240,320" \
             --soc_version=${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --soc\_version：处理器型号。
           



2. 开始推理验证。

   1. 安装ais_bench推理工具。
   
      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      ```
      python -m ais_bench --model sp-${batchsize}.om --input ${bin_path} --output ${result_path} --batchsize ${batchsize}
      ```

      - 参数说明：
           - --model：om模型路径。
           - --input：数据预处理得到的bin文件。
           - --output：推理结果保存的目录。
           - --batchsize： batchsize的大小 
          
        
      

   
   3. 精度验证。
   
       在代码主目录进行精度计算
   
       ```
       python superpoint_postprocess.py --img_path=$config_path --result_path=$result_path
       ```
       | 参数           | 说明 |
       | -------- |---------------------------|
       | img_path | config路径(./configs/magicpoint_repeatability_heatmap.yaml) |
       | result_path    | 推理文件保存的位置 |
   4. 性能验证。
   
      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
   
      ```
      python -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
      ```
        | 参数           | 说明 
        | -------- |---------------------------|                               
        | ais_infer_path    | ais_infer文件路径 |
        | om_model_path    | 模型文件保存的位置 |
        |batchsize         | batchsize大小  |


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号        | Batch Size | 数据集 | 精度 | 性能 |
|-------------|------------|----| ---------- |--|
| Ascend310P3 | 1          | hpatches   |     80.6%      | 2124 |
| Ascend310P3 | 4          | hpatches   |     80.6%       | 2437 |
| Ascend310P3 | 8          | hpatches   |     80.6%       | 2528 |
| Ascend310P3 | 16         | hpatches   |     80.6%       | 2515 |
| Ascend310P3 | 32         | hpatches   |     80.6%       | 2515 |
| Ascend310P3 | 64         | hpatches   |     80.6%       | 1787 |
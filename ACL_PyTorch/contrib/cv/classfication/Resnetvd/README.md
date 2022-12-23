#  Resnetvd模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******

  



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

SE_ResNet50模型是在ResNet网络的基础上，增加了Squeeze-and-Excitation(SE)模块，通过显式建模通道之间的相互依赖性，自适应地重新校准通道特征响应。 


- 参考实现：

  ```
  url=https://github.com/Cadene/pretrained-models.pytorch
  commit_id=ea25bb0c6a05a36c7a7ae145d5114ad3ee6048b9
  code_path=ACL_PyTorch/contrib/cv/classfication/
  model_name=Resnetvd
  ```
  





## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | 1 x 1000 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

- 该模型需要以下依赖 

  **表 2**  依赖列表

  | 依赖名称         | 版本     |
  | ---------------- | -------- |
  | onnx             | 1.10.2   |
  | Torch            | 1.8.0    |
  | TorchVision      | 0.9.0    |
  | numpy            | 1.21.4   |
  | Pillow           | 9.3.0    |
  | opencv-python    | 4.5.4.58 |
  | pretrainedmodels | 0.7.4    |
  | protobuf         | 3.20.0   |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

    ```
   git clone https://github.com/Cadene/pretrained-models.pytorch
   cd pretrained-models.pytorch
   git reset ea25bb0c6a05a36c7a7ae145d5114ad3ee6048b9 --hard
   cd ..
   ```
   
2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持[ImageNet 50000](https://gitee.com/link?target=http%3A%2F%2Fwww.image-net.org)张图片的验证集 上传数据集到服务器任意目录（如：*/home/HwHiAiUser/dataset*）。图片与标签分别存放在*/home/HwHiAiUser/dataset*/ImageNet/val_union与*/home/HwHiAiUser/dataset*/ImageNet/val_label.txt位置。目录结构如下：

   ```
   ├── ImageNet
     ├── ILSVRC2012_img_val
     ├── val_label.txt 
   ```

2. 数据预处理。

   将原始数据集转换为模型输入的数据。

   将原始数据（.jpeg）转化为二进制文件（.bin）。

   执行脚本 imagenet_torch_preprocess.py 。
   
   ```
    python3 ./imagenet_torch_preprocess.py /home/HwHiAiUser/dataset/ImageNet/val_union ./data/ImageNet_bin
   ```
   
   + 参数说明：
     + 第一个参数：原始数据集图片 （.jpeg）所在路径。
     + 第二个参数：输出的二进制文件（.bin）所在路径。
   
   每个图像对应生成一个二进制文件。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

        下载[se_resnet50-ce0d4300.pth](https://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth)权重文件   

        ```
       wget https://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth
       ```

   2. 导出onnx文件。

      1. 使用 SE_ResNet50_pth2onnx.py导出onnx文件。

         运行SE_ResNet50_pth2onnx.py脚本,

         ```
          python3 SE_ResNet50_pth2onnx.py ./se_resnet50-ce0d4300.pth ./resnetvd_bs.onnx
         ```
   
          执行后在当前路径下生成se_resnet50_dynamic_bs.onnx模型文件。 

   3. 使用ATC工具将ONNX模型转OM模型。
   
      1. 配置环境变量。
   
         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```
   
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
         atc --model=./resnetvd_bs.onnx --framework=5 --input_format=NCHW --input_shape="image:1,3,224,224" --output=./resnetvd_fp16_bs1 --log=error --soc_version=Ascend${chip_name} --insert_op_conf=./aipp_SE_ResNet50_pth.config --enable_small_channel=1
         ```
         
         - 参数说明：
         
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --insert_op_config：插入算子的配置文件路径与文件名，例如aipp预处理算子 。
           -   --enable_small_channel：Set enable small channel. 0(default): disable; 1: enable。
           
         
         运行成功后生成 resnetvd_fp16_bs1.om 模型文件。
   
2. 开始推理验证。

   1. 使用ais_bench工具进行推理。

      ais_bench工具获取及使用方式请点击查看[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

        ```
        python3 -m ais_bench --model ./resnetvd_fp16_bs1.om --input ./data/ImageNet_bin/ --output ./ --output_dirname bs1 --outfmt BIN --batchsize 1
        ```

        -   参数说明：

             -   --model：om模型。
             -   --input：预处理数据集路径。
             -   --output：推理结果所在路径。
             -   --outfmt：推理结果文件格式。
             -   --output_dirname： 推理结果输出子文件夹。可选参数。与参数output搭配使用，单独使用无效。设置该值时输出结果将保存到 output/output_dirname文件夹中 。
             -   --batchsize：不同的batchsize。
   
        推理后的输出默认在当前目录result下。
   
        >**说明：** 
        >执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见。
   
   3. 精度验证。
   
      调用vision_metric_ImageNet.py工具脚本与数据集标签val_label.txt比对，可以获得Accuracy Top5数据 。 
   
      ```
      python3 ./vision_metric_ImageNet.py ./bs1/ /opt/npu/imageNet/val_label.txt ./ accuracy_result.json
      ```
      
      - 参数说明：
      
        - 第一个参数： 推理结果输出结果目录  。
        - 第二个参数： 数据集标签val_label.txt所在路径。
        - 第三个参数 ：  精度结果输出路径。
        - 第四个参数 ：精度结果输出文件名称。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| 310P3 | 1 | ImageNet 50000 | Acc@1： 77.37  Acc@5： 93.77 | 1134.63 |
| 310P3 | 4 | ImageNet 50000 | Acc@1： 77.37  Acc@5： 93.77 | 2219.723 |
| 310P3 | 8 | ImageNet 50000 | Acc@1： 77.37  Acc@5： 93.77 | 1999.132 |
| 310P3 | 16 | ImageNet 50000 | Acc@1： 77.37  Acc@5： 93.77 | 1700.829 |
| 310P3 | 32 | ImageNet 50000 | Acc@1： 77.37  Acc@5： 93.77 | 2823.656 |
| 310P3 | 64 | ImageNet 50000 | Acc@1： 77.37  Acc@5： 93.77 | 1574.383 |


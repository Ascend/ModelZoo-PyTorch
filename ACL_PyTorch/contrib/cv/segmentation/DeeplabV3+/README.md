# DeepLabV3+模型-推理指导


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

DeepLabV3+就是属于典型的DilatedFCN，它是Google提出的DeepLab系列的第4弹, 它的Encoder的主体是带有空洞卷积的DCNN，可以采用常用的分类网络如ResNet，然后是带有空洞卷积的空间金字塔池化模块（Atrous Spatial Pyramid Pooling, ASPP)），主要是为了引入多尺度信息；相比DeepLabv3，v3+引入了Decoder模块，其将底层特征与高层特征进一步融合，提升分割边界准确度.

- 参考论文：Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. (2018)


- 参考实现：
  ```
  url=https://github.com/jfzhang95/pytorch-deeplab-xception
  branch=master
  commit_id=9135e10
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 513 x 513 | NCHW         |


- 输出数据

  | 输出数据 | 大小                      | 数据类型 | 数据排布格式 |
  | -------- | -------------------------- | -------- | ------------ |
  | output1  | batchsize x 21 x 513 x 513 | FLOAT16  | ND           |



# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2   | -                                                            |
| Python                                                       | 3.8.13  | -                                                            |
| PyTorch                                                      | 1.6.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

代码目录参考。

   ```
   ├── dataset                                       //VOC2012数据集所在文件夹
   ├── prep_bin                                     //输出的二进制文件（.bin）所在路径
   ├── lmcout                                      //推理结果所在路径。
   ├── preprocess_deeplabv3plus_pytorch.py         //数据集预处理脚本，通过均值方差处理归一化图片，生成图片二进制文件
   ├── deeplabV3+_pth2onnx.py                     //用于转换pth模型文件到onnx模型文件
   ├── ais_infer.py                               //ais_infer工具推理文件
   └── post_deeplabv3+_pytorch.py                 //验证推理结果脚本，比对benchmark输出的分类结果和标签，给出Accuracy

   ```

1. 安装依赖。

   因为需要使用numpy_to_ptr函数，该接口即将废弃，若要继续使用该接口，需要运行环境为python>=3.8且numpy>=1.22.0。
   所以为了方便起见，选用python版本为3.8。
   ```
   pip install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。(解压命令参考tar –xvf *.tar与 unzip *.zip)

   请用户需自行获取VOC2012数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）
   VOC2012验证集所需文件目录参考（只列出该模型需要的目录）。

   ```
   ├── ImageSets
      └── Segmentation
         ├── train.txt
         ├── trainval.txt
         └── val.txt            //验证集文件列表
   ├── JPEGImages                 //验证数据集文件夹
   └── SegmentationsClass         //语义分割集
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   在目录下创建prep_bin文件夹。
   ```
   python preprocess_deeplabv3plus_pytorch.py ./dataset/VOCdevkit/VOC2012/JPEGImages/ ./prep_bin/ ./dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt
   ```
   第一个参数为原始数据验证集（.jpeg）所在路径，第二个参数为输出的二进制文件（.bin）所在路径。每个图像对应生成一个二进制文件。第三个为处理的图片的列表。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

   下载源码包：https://www.hiascend.com/zh/software/modelzoo/models/detail/1/76f4e072a489484f98073591b912ad16。
   从源码包中获取训练后的权重文件deeplab-resnet.pth.tar。

   2. 导出onnx文件。

      1. 导出onnx文件。

         下载代码仓。
         ```
         git clone https://github.com/jfzhang95/pytorch-deeplab-xception.git
         ```
         将代码仓上传至服务器路径下。
         进入代码仓目录并将权重文件deeplab-resnet.pth.tar和deeplabV3plus_pth2onnx.py脚本移到当前目录下。
         
         ```
         mv ./deeplab-resnet.pth.tar ./pytorch-deeplab-xception/
         mv deeplabV3plus_pth2onnx.py ./pytorch-deeplab-xception/
         cd pytorch-deeplab-xception/
         ```

         执行deeplabV3plus_pth2onnx.py脚本将.pth.tar文件转换为.onnx文件，执行如下命令。
         ```
         python deeplabV3plus_pth2onnx.py ./deeplab-resnet.pth.tar ./deeplabv3_plus_res101.onnx
         ```
         第一个参数为输入权重文件路径，第二个参数为输出onnx文件路径。

         运行成功后，在当前目录生成deeplabv3_plus_res101.onnx模型文件。

         获得deeplabv3_plus_res101.onnx文件。

      2. 优化ONNX文件。

         简化onnx文件。
         ```
         python -m onnxsim deeplabv3_plus_res101.onnx deeplabv3_plus_res101_sim.onnx --input-shape 1,3,513,513
         ```
         运行成功后生成deeplabv3_plus_res101_sim.onnx。

         将生成的onnx文件移动到从ModelZoo上获得的源码包中。

         获得deeplabv3_plus_res101_sim.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         使用命令激活npu：export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/Ascend/driver/lib64/driver/

         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         回显如下：
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 15.9         50                1127  / 1127          |
         | 0       0         | 0000:86:00.0    | 0            3683 / 21534                            |
         +===================+=================+======================================================+
         ```

      3. 执行ATC命令。

         使用二进制输入时，执行如下命令
         ```
         atc --model=./deeplabv3_plus_res101_sim.onnx --framework=5 --output_type=FP16 --output=deeplabv3_plus_res101-sim_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,513,513" --enable_small_channel=1 --log=error --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成deeplabv3_plus_res101_sim_bs1.om模型文件。
           
        ```
        mv deeplabv3_plus_res101_sim_bs1.om ../
        cd ..
        ```



2. 开始推理验证。

a.  使用ais-infer工具进行推理。

   执行命令增加工具可执行权限，并根据OS架构选择工具。
   ```
   chmod u+x ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py
   ```
b.  执行推理。

    
    python ais_infer.py --model deeplabv3_plus_res101_sim_bs1.om --input ./prep_bin/ --output ./lmcout/ --outfmt BIN --batchsize 1
    

   - 参数说明：

      -   --model：需要进行推理的om模型。
      -   --input：模型需要的输入，支持bin文件和目录，若不加该参数，会自动生成都为0的数据。
      -   --output：推理结果输出路径。默认会建立日期+时间的子文件夹保存输出结果 如果指定output_dirname 将保存到output_dirname的子文件夹下。
      -   --outfmt: 输出数据的格式，默认”BIN“，可取值“NPY”、“BIN”、“TXT”。
      -   --batchsize：模型batch size 默认为1 。当前推理模块根据模型输入和文件输出自动进行组batch。参数传递的batchszie有且只用于结果吞吐率计算。请务必注意需要传入该值，以获取计算正确的吞吐率。

   推理后的输出默认在当前目录result下。

c.  精度验证。
   
   删除sumary.json文件
   ```   
   rm -rf ./lmcout/xxxx/sumary.json
   ```
   使用脚本post_deeplabv3plus_pytorch.py精度测试。
   ```
   python post_deeplabv3plus_pytorch.py --result_path=./lmcout/xxxxx/ --label_images=./dataset/VOCdevkit/VOC2012/SegmentationClass/ --labels=./dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt
   ```
   - 参数说明：

     -   --result_path：推理结果所在路径。
     -   --label_images:标签数据图片文件。
     -   --labels：验证集图像名称列表。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|    310    |        1         |   VOC2012  |    78.44   |     102.32      |
|    310P   |        1         |   VOC2012  |    78.43   |     182.02      |
|    T4     |        1         |   VOC2012  |    78.30   |      92.09      |

备注：此模型只支持bs1。
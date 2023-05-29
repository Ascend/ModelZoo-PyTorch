# SE-ResNet50模型-推理指导


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

   SE-ResNet50是[Squeeze-and-Excitation Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)论文基于ResNet50的实现。

- 参考实现：

  ```
  url=https://github.com/Cadene/pretrained-models.pytorch.git
  branch=master
  commit_id=8aae3d8f1135b6b13fed79c1d431e3449fdbf6e0
  model_name=SE-ResNet50
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input1    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | batchsize x 1000 | FLOAT32  | ND   |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                    | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.5.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本 。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone -b master https://github.com/Cadene/pretrained-models.pytorch.git
   cd pretrained-models.pytorch
   git reset --hard 8aae3d8f1135b6b13fed79c1d431e3449fdbf6e0
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   cd pretrained-models.pytorch
   python3 setup.py install
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，以ILSVRC2012为例，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。解压后数据集结构如下：
   
   ```
   ├── imageNet
       ├── val
           ├── ILSVRC2012_val_00000001.JPEG
               ...
           ├── ILSVRC2012_val_00050000.JPEG
       ├── val_label.txt
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行SE_ResNet50_preprocess.py脚本，完成预处理。

   ```
   python3 SE_ResNet50_preprocess.py resnet ${data_path}/imageNet/val ./pre_data
   ```

   第一个参数为模型类型，第二个参数为测试图像路径，第三个参数为预处理数据保存路径。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

   参考[源码说明](https://github.com/Cadene/pretrained-models.pytorch#senet)获取权重文件，权重文件链接为：http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth。

   在当前目录下可通过`wget`命令获取。

   ```
   wget http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth --no-check-certificate
   ``` 

   2. 导出onnx文件。

      1. 使用SE_ResNet50_pth2onnx.py导出onnx文件。

         运行SE_ResNet50_pth2onnx.py脚本。

         ```
         python3 SE_ResNet50_pth2onnx.py \
             --pth=./se_resnet50-ce0d4300.pth \
             --onnx=./se_resnet50_dybs.onnx
         ```
         
         - 参数说明：

            -   --pth：表示权重文件。
            -   --onnx：表示onnx保存文件。

         在当前目录下获得se_resnet50_dybs.onnx文件。

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
         atc --framework=5 \
             --model=./se_resnet50_dybs.onnx \
             --input_format=NCHW \
             --input_shape="image:${batchsize},3,224,224" \
             --output=./se_resnet50_bs${batchsize} \
             --log=error \
             --soc_version=Ascend${chip_name} 
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

         ${batchsize}表示转出不同batchsize的om模型，当前可支持的batchsize为：1，4，8，16，32，64。
         运行成功后生成`./se_resnet50_bs${batchsize}.om`模型文件。


2. 开始推理验证。

   a.  安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  


   b.  执行推理。

      ```
      python3 -m ais_bench \
          --model=./se_resnet50_bs${batchsize}.om \
          --input=./pre_data \
          --output=./ \
          --outfmt=TXT \
		  --batchsize=${batch_size}
      ```

      -   参数说明：

           -   --model：om模型路径。
           -   --input：bin文件路径。
           -   --output：推理结果保存路径。
           -   --outfmt：推理结果保存格式。
		   -   --batchsize：om模型的batchsize。

      `${batchsize}`表示不同batch的om模型。
      
      推理完成后在当前`SENet`工作目录生成推理结果。其目录命名格式为`xxxx_xx_xx-xx_xx_xx`(`年_月_日-时_分_秒`)，如`2022_08_18-06_55_19`。


   c.  精度验证。

      调用脚本与数据集标签val_label.txt比对，可以获得Accuracy数据，结果保存在result_bs${batchsize}.json中。

      ```
      python3 SE_ResNet50_postprocess.py ${output_path}/xxxx_xx_xx-xx_xx_xx {data_path}/imageNet/val_label.txt ./ result_bs${batchsize}.json
      ```
      
      第一个参考为ais_bench推理工具推理结果路径，第二个参数为标签文件val_label.txt路径，第三个参数为精度结果文件保存路径，第四个参数为不同batchsize精度结果文件名称。

   d.  性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
       python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

|   芯片型号    | Batch Size |  数据集  |            精度             |   性能   |
| ------------ | ---------- | ---------| -------------------------- | -------- |
|  Ascend310P3 | 1          | ImageNet | Acc@1:77.64%; Acc@5:93.74% | 1064.92 |
|  Ascend310P3 | 4          | ImageNet | Acc@1:77.64%; Acc@5:93.74% | 2113.68 |
|  Ascend310P3 | 8          | ImageNet | Acc@1:77.64%; Acc@5:93.74% | 2348.42 |
|  Ascend310P3 | 16         | ImageNet | Acc@1:77.64%; Acc@5:93.74% | 2358.25 |
|  Ascend310P3 | 32         | ImageNet | Acc@1:77.64%; Acc@5:93.74% | 2479.68 |
|  Ascend310P3 | 64         | ImageNet | Acc@1:77.64%; Acc@5:93.74% | 1494.32|
注：模型最优batchsize=4。
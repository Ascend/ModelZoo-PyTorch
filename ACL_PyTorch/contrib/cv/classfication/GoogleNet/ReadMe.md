# ATC GoogleNet (FP16)模型-推理指导


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

GoogleNet是一种用于图像分类的卷积神经网络，这个模型的默认输入尺寸是224×224，有三个通道。该篇论文的作者在ILSVRC 2014比赛中提交的报告中使用了GoogLeNet，这是一个22层的深度网络。在这里面提出了一种新的叫做Inception的结构。该网络具有很大的depth和width，但是参数数量却仅为AlexNet的1/12。



- 参考实现：

  ```
  url=https://github.com/pytorch/vision.git
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
| CANN                                                         | 3.3.0 | -                                                            |
| ONNX                                                       | 1.7.0   | -                                                            |
| PyTorch                                                      | 1.6.0   | 
| TorchVision                                                   | 0.7.0 | -                                                            |
| numpy                                                        | 1.18.5 | -  
| Pillow                                                        | 7.2.0 | -                                                            |
                                                         

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>



1. 单击“立即下载”，下载源码包。
2. 上传源码包到服务器任意目录并解压（如：/home/HwHiAiUser）。


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。模型输入数据有两种格式，分别为二进制输入和jpeg图像输入。

   - 二进制输入
    将原始数据（.jpeg）转化为二进制文件（.bin）。转化方法参考Torchvision训练预处理方法处理数据，以获得最佳精度。通过缩放、均值方差手段归一化，输出为二进制文件。

    执行preprocess_googlenet_pth.py脚本。

    ```
    python3.7 preprocess_googlenet_pth.py ./ImageNet/ILSVRC2012_img_val ./prep_bin
    ```
    第一个参数为原始数据验证集（.jpeg）所在路径，第二个参数为输出的二进制文件（.bin）所在路径。每个图像对应生成一个二进制文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件[googlenet-1378be20.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/GoogleNet/PTH/googlenet-1378be20.pth)转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 导出onnx文件。

     - 获取ONNX模型。
      googlenet_pth2onnx.py脚本将.pth文件转换为.onnx文件，执行如下命令。
      
      
      ```
      python3.7 googlenet_pth2onnx.py ./googlenet-1378be20.pth ./googlenet.onnx
      ```
      第一个参数为输入权重文件路径，第二个参数为输出onnx文件路径。
      
      运行成功后，在当前目录生成googlenet.onnx模型文件。
      
      须知：使用ATC工具将.onnx文件转换为.om文件，需要.onnx算子版本需为11。在googlenet_pth2onnx.py脚本中torch.onnx.export方法中的输入参数    opset_version的值需为11，请勿修改。

   2. 执行命令查看芯片名称（${chip_name})。

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
      atc --model=./googlenet.onnx --framework=5 --output=googlenet_bs1_new --input_format=NCHW --input_shape="actual_input_1:1,3,224,224" --log=info --soc_version=Ascend${chip_name}
      ```


         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc_version:处理器型号。


2. 开始推理验证。

   a.安装ais_bench推理工具。

      1.本推理工具编译需要安装好CANN环境。用户可以设置CANN_PATH环境变量指定安装的CANN版本路径，比如export CANN_PATH=/xxx/nnae/latest/. 如果不设置，本推理工具默认会从 CANN_PATH /usr/local/Ascend/nnae/latest/ /usr/local/Ascend/ascend-toolkit/latest 分别尝试去获取

 
      2.请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   b.执行推理。
   
      ```
      python -m ais_bench --model ./googlenet_bs32.om --input ./prep_bin/ --output ./lmcout/bs32 --outfmt TXT --batchsize 32
      ```
   
   c.精度验证。
   
     调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

     ```
     python googleNet_postprocess.py --result_path=xxx/sumary.json
     ```
       

      - 参数说明：

           -   --xxx/sumary.json：生成推理结果所在路径。
           -   --val_label.txt：标签数据。
           -   --result.json：生成结果文件。
        
       
# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

精度：
| BatchSize | top1   | top5   |
| --------- | ------ | ------ |
|     1     | 69.77% | 89.51% |
|     8     | 69.77% | 89.51% |

性能：

| Batch Size |     310     |      310P      |       T4       |   310P/310  |    310P/T4    |
| ---------- | ----------- | -------------- | -------------- | ----------- | ------------- |
|     1      |   1864.996  |   1923.5063    |   311.16746    |   1.03137   |   6.181579    |
|     4      |   3971.292  |   4487.84812   |   745.90649    |   1.13      |   6.016636    |
|     8      |   4475.56   |   6287.326287  |   878.6491068  |   1.4048    |   7.15567     |
|     16     |   4819.32   |   6017.703841  |   1019.17379   |   1.24866   |   5.9         |
|     32     |   4399.12   |   5567.754933  |   1063.788635  |   1.26565   |   5.23389     |
|     64     |   3970.46   |   4932.890785  |   1065.05367   |   1.2423978 |   4.63159     |
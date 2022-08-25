# VGG16模型-推理指导


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

VGGNet是牛津大学计算机视觉组（Visual Geometry Group）和Google DeepMind公司的研究员一起研发的深度卷积神经网络，它探索了卷积神经网络的深度与其性能之间的关系，通过反复堆叠3*3的小型卷积核和2*2的最大池化层，成功地构筑了16~19层深的卷积神经网络。VGGNet相比之前state-of-the-art的网络结构，错误率大幅下降，VGGNet论文中全部使用了3*3的小型卷积核和2*2的最大池化核，通过不断加深网络结构来提升性能。
VGG16包含了16个隐藏层（13个卷积层和3个全连接层）


- 参考实现：

  ```
  url=https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
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
  | output1  | batchsize x 1000 | FLOAT32  | ND           |



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



1. 安装依赖。

   ```
   pip3 install -r requirment.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。

   数据目录结构请参考：
   ```
   |-- dataset
     |-- ILSVRC2012_val_00000001.JPEG
     |-- ILSVRC2012_val_00000002.JPEG
     |-- ILSVRC2012_val_00000003.JPEG
     |-- ...
   ```

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   执行“preprocess_vgg_pytorch.py”脚本，完成预处理。

   ```
   python3.7 preprocess_vgg_pytorch.py ${dataset_dir} ./prep_dataset
   ```
   ${datasets_path}：原始数据验证集（.jpeg）所在路径。

   “./prep_dataset”：输出的二进制文件（.bin）所在路径。

   每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成“prep_dataset”二进制文件夹。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
       
       从源码包中获取权重文件：“vgg16-397923af.pth”。
       或从这里下载权重文件：https://download.pytorch.org/models/vgg16-397923af.pth

   2. 导出onnx文件。

      1. 使用 pth2onnx.py 导出onnx文件。

         执行“pth2onnx.py”脚本。

         ```
         python3.7 pth2onnx.py ./vgg16-397923af.pth ./vgg16.onnx
         ```

         获得“vgg16.onnx”文件。

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
         atc --framework=5 --model=./vgg16.onnx --output=vgg16_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,224,224" --log=info --soc_version=Ascend${chip_name}
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

           运行成功后生成<u>***vgg16_bs1.om***</u>模型文件。



2. 开始推理验证。

   a. 使用ais-infer工具进行推理。
      
      执行命令增加工具可执行权限，并根据OS架构选择工具

      ```
      chmod u+x ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py
      ```

   b. 执行推理。

   ```
   python ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model ./vgg16_bs1.om --input ./prep_dataset --output ./lcmout/ --outfmt TXT --batchsize 1
   ```

   -   参数说明：

        -   --model：om文件路径。
        -   --input：预处理完的数据集文件夹
        -   --output：推理结果保存地址
        -   --outfmt：推理结果保存格式
        -   --batchsize：batchsize大小

        推理后的输出在--output所指定目录下。

        >**说明：** 
        >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见 [ais_infer推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)。

   c.  精度验证。

   调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据。

    ```
    python vgg16_postprocess.py --gtfile=./val_label.txt --result_path=./lcmout/2022_xx_xx-xx_xx_xx/sumary.json
    ```
   -   参数说明：

        -   --result_path：生成推理结果summary.json所在路径。
        -   --gtfile_path：标签val_label.txt所在路径
    

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

性能参考下列数据:

|	       |  310	 | 710	    | T4	   |710/310	    |710/T4    |
|----------|---------|----------|----------|------------|----------|
|bs1 	   | 460.68	 |467.54	|395.3569  |1.014891031	|1.182577059|
|bs4 	   | 834.48	 |1074.59	|352.7865  |1.287736075	|3.046006579|
|bs8	   | 947.02	 |1370.47	|375.8303  |1.447139448	|3.646512801|
|bs16	   | 1041.48 |1567.42	|820.3655  |1.504992895	|1.910636174|
|bs32	   | 1076.3	 |1473.20   |428.8135  |1.368763356	|3.435526167|
|bs64	   | 936.44	 |1521.60   |772.2594  |1.624877194	|1.970322407|					
|最优batch  | 1076.3  |1567.42	|820.3655  |1.456304004	 |1.910636174|

精度参考下列数据:
|     | top1_acc  | top5_acc |
|-----|---------- |----------|
| 310  |  0.7128  | 0.9038   |
| 310P |  0.7128  | 0.9038   |


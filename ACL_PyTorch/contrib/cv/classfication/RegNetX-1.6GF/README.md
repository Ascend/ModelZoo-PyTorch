# RegNetX-1.6GF模型-推理指导


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

RegNet并不是一个单一的网络，甚至也不是一个像EfficientNets这样的扩展的网络家族。它是一个被量化的线性规则限制的设计空间，期望包含好的模型。

- 参考实现：

  ```
  url=https://github.com/rwightman/pytorch-image-models.git
  commit_id=742c2d524726d426ea2745055a5b217c020ccc72
  model_name=RegNetX-1.6GF
  ```

  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch}         # 切换到对应分支
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
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.8.1   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   RegNetX-1.6GF模型代码在timm里，安装timm，arm下需源码安装，参考https://github.com/rwightman/pytorch-image-models ，若安装过程报错请百度解决。
         
   ```
   git clone https://github.com/rwightman/pytorch-image-models
   cd pytorch-image-models
   python3.7 setup.py install
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。     
      ```
     ├── ImageNet
        ├── ILSVRC2012_img_val
        ├── val_label.txt
      ```
   
2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   执行imagenet_torch_preprocess.py脚本，完成预处理。

   ```
   python3.7 imagenet_torch_preprocess.py /home/HwHiAiUser/dataset/ImageNet/ILSVRC2012_img_val ./prep_dataset
   ```
         
      -   参数说明：

           -   /home/HwHiAiUser/dataset/ImageNet/ILSVRC2012_img_val：原始数据验证集（.jpeg）所在路径。
           -   ./prep_dataset：输出的二进制文件所在路径。
       每个图像对应生成一个二进制文件。
   
   
## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。
   
   1. 获取权重文件。
     [RegNetX-1.6GF预训练pth权重文件](wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_016-65ca972a.pth )
        
   ```
   wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_016-65ca972a.pth 
   ```

   2. 使用RegNetX_onnx.py导出onnx文件。

      运行RegNetX_onnx.py脚本。
 
   ```
   python3.7 RegNetX_onnx.py regnetx_016-65ca972a.pth RegNetX-1.6GF.onnx
   ```
         
     运行成功后生成RegNetX-1.6GF.onnx模型文件，生成不同batch_size的onnx时，需修改参数。在RegNetX_onnx.py脚本中的dummy_input = torch.randn(1, 3, 224, 224)，如需生成batch_size=4的onnx，即将torch.randn(1,3,224,224)中的1改为4即可。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
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
          atc --framework=5 --model=./RegNetX-1.6GF_bs1.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=RegNetX-1.6GF_bs1 --log=debug --soc_version=Ascend${chip_name}
         ```
      -   参数说明：

           -   model：为ONNX模型文件。
           -   framework：5代表ONNX模型。
           -   output:输出的OM模型。
           -   input_format:输入数据的格式。
           -   input_shape：输入数据的shape。
           -   log：日志级别。
           -   soc_version：处理器型号。
              运行成功后生成RegNetX-1.6GF_bs1.om模型文件。

2. 开始推理验证。

   a.  使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]


   b.  执行推理。

      ```
        python3.7 ais_infer.py --model /home/tangxiao/file/RegNetX-1.6GF_bs1.om --input "/home/tangxiao/RegNetX-1.6GF/prep_dataset" --output "/home/tangxiao/RegNetX-1.6GF" --outfmt TXT --batchsize n
      ```

      -   参数说明：

           -   model：om文件路径。
           -   input：数据集预处理后的文件。
           -   output:推理结果输出路径。默认会建立日期+时间的子文件夹保存输出结果。
           -   outfmt:输出结果的格式，指定为txt格式。
           -   batchsize:输入模型的batchsize

      推理后的输出默认在建立日期+时间的子文件夹中，除了图片信息的txt文件外，还有一个.json文件，需手动删除。

   c.  精度验证。

      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

      ```
       python3.7 vision_metric_ImageNet.py ./output_dirname/ ./val_label.txt ./ result.json
      ```
      -   参数说明：

           -   ./output_dirname/：生成推理结果所在路径。
           -   val_label.txt：为标签数据。
           -   result.json:为生成结果文件。

   d.  性能验证。

      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
       python3.7 ${ais_infer_path}/ais_infer.py --model=${om_model_path} --loop=20 --batchsize=${batch_size}
      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。                                                                                                                   
       精度：

| Precision  | mAP  |
|---|---|
| 310精度  | 93.43%  |
| 310P精度  | 93.42%  |                                                                                                                                      

       性能：

| Throughput  | 310  | 310P  | T4  | 310P/310   | 310P/T4  |
|---|---|---|---|---|---|
| RegNetX-1.6GF_bs1  | 930.57  | 1677.04  | 436.27  | 1.802  | 3.844  |
| RegNetX-1.6GF_bs4  | 3355.08  | 4372.53  | 1084.42  | 1.303  | 4.032  |
| RegNetX-1.6GF_bs8  | 3608.97  | 5486.04  | 1532.78  | 1.520  | 3.579  |
| RegNetX-1.6GF_bs16  | 2783.01  | 3934.81  | 1867.64  | 1.413  | 2.107  |
| RegNetX-1.6GF_bs32  | 3170.12  | 3752.48  | 2166.99  | 1.184  | 1.732  |
| RegNetX-1.6GF_bs64  | 3186.38  | 3623.42  | 2276.49  | 1.137  | 1.592  |
| 最优batch  | 3608.97  | 5486.04  | 2276.49  | 1.520  | 2.410  |






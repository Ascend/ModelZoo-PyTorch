# ConvNext模型-推理指导


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

ConvNext是卷积神经网络,有内嵌的感应偏差，使他们更好的适用于各种各样的计算机视觉任务。



- 参考实现：

  ```
  url=https://github.com/facebookresearch/ConvNeXt.git 
  branch=main 
  commit_id=3d444184dc27156e0562133c3b69b56f7efba500
  model_name=models/convnext
  ```



  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone https://github.com/facebookresearch/ConvNeXt.git        # 克隆仓库的代码
  cd ConvNext              # 切换到模型的代码仓目录
  git checkout main         # 切换到对应分支
  git reset --hard 3d444184dc27156e0562133c3b69b56f7efba500      # 代码设置到对应的commit_id
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
| PyTorch                                                      | 1.8.0   | -                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   https://github.com/facebookresearch/ConvNeXt.git 
   branch=main 
   commit_id=3d444184dc27156e0562133c3b69b56f7efba500

   git clone https://github.com/facebookresearch/ConvNeXt.git
   ```
2. 修改源码。
此模型转换为onnx需要修改开源代码仓代码
   ```
   patch -p1 < convnext.patch
   ```

3. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试。

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   执行ConvNext_preprocess.py脚本，完成预处理。

   ```
   python ConvNext_preprocess.py --dataset_root ${dataset_path} --output_dir ${prep_output_dir} --bs ${batch_size}
   ```

  data_root为预处理之前的数据集路径，output_dir为预处理结果的输出路径，bs为生成的二进制文件的batch_size。batch_size=1的二进制文件适用与后面所有batch_size的推理验证。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       ```
         wget https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
         ```


   2. 导出onnx文件。

      1. 使用ConvNext_pth2onnx.py导出onnx文件。

         运行ConvNext_pth2onnx.py脚本。

         ```
         python ConvNext_pth2onnx.py convnext_tiny_1k_224_ema.pth convnext.onnx
         ```

         convnext_tiny_1k_224_ema.pth为获取的权重文件,获得convnext.onnx文件。


   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-lastest/set_env.sh
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
          atc --framework=5 --model=./convnext_bs1.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=convnextbs1 --log=error --soc_version=${chip_name} --keep_dtype=execeotionlist.cfg --op_precision_mode=op_precision.ini
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input_format：输入数据的格式。
           -   --input_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --keep_dtype：保持部分算子fp32格式。
           -   --op_precision_mode：优化性能配置文件的路径。

           运行成功后生成convnextbs1.om模型文件。



2. 开始推理验证。

   a.  安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。  


   b.  执行推理。

      ```
       python -m ais_bench --model ./convnextbs1.om --input ./data/ --output ./output/ --outfmt  BIN --batchsize 1 
      ```

      -   参数说明：

           -   outfmt:推理输出文件的格式。
           -   input:预处理后的数据路径。
           -   batchsize:推理的batchsize。
           -   model：om文件路径。
           -   input:预处理后的数据路径。
           -   output:推理输出路径。
	  

      推理后的输出默认在当前目录result下。


   c.  精度验证。

      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

      ```
       python ConvNext_postprocess.py ./output ./Result result.json ./val_label.txt
      ```
     -   参数说明：

           -   ./output：为生成推理结果所在路径。
           -   val_label.txt：为标签数据。
           -   result.json：生成结果文件。
           -   ./Result：为result.json所在路径。
        
   d.  性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
       python -m ais_bench --model ./convnextbs1.om --output ./output/ --outfmt  BIN --batchsize 1
      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|    310p       |         1         |      ImageNet      |       82.094%      |       262.4          |
|    310p       |         4         |      ImageNet      |       82.093%      |       440.91          |
|    310p       |         8         |      ImageNet      |       82.095%      |       461.9          |
|    310p       |         16         |      ImageNet      |       82.092%      |       449.03          |
|    310p       |         32         |      ImageNet      |       82.094%      |       435.7          |
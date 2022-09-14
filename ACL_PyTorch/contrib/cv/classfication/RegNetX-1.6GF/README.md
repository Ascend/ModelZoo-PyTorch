# {模型名称}模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  ******

  ***<u>斜体带下划线且加粗内容为备注，写作时请删除</u>***

  ***<u>标题请删除模板字眼</u>***



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

MobileNetV2是针对移动端专门定制的轻量级卷积神经网络，该网络将MobileNetV1和ResNet的残差单元结合，相比MobileNetV1网络，一是在Depth-wise卷积之前加入了一个Point-wise卷积，二是将最后输出的ReLU6去掉，直接线性输出，避免特征信息的损失；相比ResNet，采用倒置残差，先升维（6倍）再卷积最后降维。

<u>***简单描述模型的结构、应用、优点等信息。***</u>


- 参考实现：

  ```
  url=branch=master
  commit_id=
  model_name=
  ```

  *<u>**url=参考的模型源代码git地址，强烈建议使用release分支版本的地址**</u>*

  *<u>**commit\_id例如：291f7e20339510cfa956b5782741697eb8e6d554**</u>*
  
  *<u>**model\_name: 子模型名，开源仓往往会提供很多子模型，需说明推理的是哪一个子模型，比如[Segmenter](https://github.com/rstrudel/segmenter)下的Seg-L-Mask/16模型**</u>*


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

<u>***请按照如上表格描述输入/输出数据的大小、数据类型、数据排布格式，若有多条输入请添加多条数据。若只有一条数据，则只有一行描述即可***</u>


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.6.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   <u>***写清楚原始数据集名称、下载链接、所用到的文件、存放路径、目录结构。***</u>

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   执行XXX脚本，完成预处理。

   ```
   
   ```

  <u>***需写明预处理执行命令，包括命令代码、参数说明、运行成功后生成文件，有其他说明请单独写出。***</u>


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       <u>***写清楚权重文件的获取方式：下载链接***</u>

   2. 导出onnx文件。

      1. 使用XXX导出onnx文件。<u>***请用脚本名称替换xxx***</u>

         运行XXX脚本。

         ```
         
         ```

         获得XXX.onnx文件。

      2. 优化ONNX文件。<u>***不存在此步骤可删除***</u>

         ```
         
         ```

         获得xxx.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/......
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

           运行成功后生成<u>***XX.om***</u>模型文件。



2. 开始推理验证。<u>***根据实际推理工具编写***</u>

   a.  使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]


   b.  执行推理。

      ```
        
      ```

      -   参数说明：

           -   xxx：模型类型。
           -   xxx：om文件路径。
           -   xxx：NPU设备编号。
		...

      推理后的输出默认在当前目录result下。

      >**说明：** 
      >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

   c.  精度验证。

      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

      ```
       python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ ./val_label.txt ./ result.json
      ```

      result/dumpOutput_device0/：为生成推理结果所在路径  <u>***参数解释***</u>
    
      val_label.txt：为标签数据
    
      result.json：为生成结果文件


   d.  性能验证。

      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
       python3.7 ${ais_infer_path}/ais_infer.py --model=${om_model_path} --loop=20 --batchsize=${batch_size}
      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|           |                  |            |            |                 |
 

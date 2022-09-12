#  SSD MobileNetV1 模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)







# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

MobileNet网络是由google团队在2017年提出的，专注于移动端或者嵌入式设备中的轻量级CNN网络。相比传统卷积神经网络，在准确率小幅降低的前提下大大减少模型参数与运算量。相比VGG16准确率减少了0.9%，但模型参数只有VGG的1/32。

优点：
·Depthwise Convolution( 大大减少运算量和参数数量)
·增加超参数 增加超参数α 、β
（其中α是控制卷积层卷积核个数的超参数，β是控制输入图像的大小）


- 参考实现：

  ```
  url=https://github.com/qfgaohao/pytorch-ssd
  branch=master
  commit_id=f61ab424d09bf3d4bb3925693579ac0a92541b0d 
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
  | input    | RGB_FP32 | batchsize x 3 x 300 x 300 | NCHW         |


- 输出数据

  | 输出数据 | 大小           | 数据类型 | 数据排布格式 |
  | -------- | -------------- | -------- | ------------ |
  | output1  | 16 x 3000 x 21 | FLOAT32  | ND           |
  | output2  | 16 x 3000 x 4  | FLOAT32  | ND           |




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
   git clone https://github.com/qfgaohao/pytorch-ssd.git -b master
   cd pytorch-ssd
   git reset f61ab424d09bf3d4bb3925693579ac0a92541b0d --hard
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   这里使用VOC2007的测试集作为测试数据集
   URL：https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
   解压后获得VOCdevkit文件，并以VOCdevkit其作为数据集路径.
   
   预训练权重：
   ```bash
   wget https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth
   ```
   下载后重命名为mobilenet-v1-ssd.pth

   数据标签：
   ```bash
   wget https://storage.googleapis.com/models-hao/voc-model-labels.txt
   ```

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   执行"SSD_MobileNet_preprocess.py"脚本，完成预处理。

   ```
   python3.7 SSD_MobileNet_preprocess.py --test_batch_size=X --output_path='bin-bsX/'
   ```

 


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       <u>***写清楚权重文件的获取方式：下载链接***</u>

   2. 导出onnx文件。

      1. 使用SSD_MobileNet_pth2onnx.py导出onnx文件。

         运行SSD_MobileNet_pth2onnx.py脚本。

         ```
         python3.7 SSD_MobileNet_pth2onnx.py mobilenet-v1-ssd.pth mb1-ssd.onnx
         ```

         获得mb1-ssd.onnx文件。


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
         atc --framework=5 --model=mb1-ssd.onnx --output=mb1-ssd_bs1 --input_format=NCHW --input_shape="image:1,3,300,300" --log=debug --soc_version=Ascend${chip_name} 
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

           运行成功后生成<u>***mb1-ssd_bs1.om***</u>模型文件。



2. 开始推理验证。

   a.  使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]


   b.  执行推理。

      ```
      python ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model mb1-ssd_bs1.om --input ./pre_dataset/ --output ./lmcout/bs1/ --outfmt BIN --batchsize 1  
      ```

      -   参数说明：

        -   --model：om文件路径。
        -   --input：预处理完的数据集文件夹
        -   --output：推理结果保存地址
        -   --outfmt：推理结果保存格式
        -   --batchsize：batchsize大小
		

      推理后的输出默认在--output所指定目录下。

      >**说明：** 
      >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见[ais_infer推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)。

   c.  精度验证。

      调用"SSD_MobileNet_postprocess.py"脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据.

      ```
       python SSD_MobileNet_postprocess.py ./VOCdevkit/VOC2007/ voc-model-labels.txt ./lmcout/bs1/xxxx/ ./eval_results1/
      ```
   -   参数说明：

        -   --voc-model-labels.txt：为标签数据
        -   --eval_results1：为生成结果文件夹


   d.  性能验证。

      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
       python3.7 ${ais_infer_path}/ais_infer.py --model=${om_model_path} --loop=20 --batchsize=${batch_size}
      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| Batch Size |     310     |      310P   |       T4     |  310P/310  |    310P/T4   |
| ---------- | ----------- | ------------| ------------ | ---------- | ------------ |
|     1      |   922.640   |   1100.894  |   1256.281   |   1.193    |   0.876      |
|     4      |   1134.212  |   2464.827  |   2300.853   |   2.173    |   1.071      |
|     8      |   1159.360  |   2487.807  |   2663.346   |   2.146    |   0.934      |
|     16     |   1165.876  |   2502.598  |   2880.288   |   2.147    |   0.869      |
|     32     |   1162.018  |   2433.221  |   3124.847   |   2.094    |   0.779      |
|     64     |   1165.688  |   2358.435  |   3208.197   |   2.023    |   0.735      |
精度参考下列数据:
|            | top1_acc     |
| ---------- | ------------ |
|    310     |   0.692555   |
|    310P    |   0.692647   | 


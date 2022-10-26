# ADNet模型-推理指导


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

 ADNet是一种注意引导的，用于图像去噪的卷积神经网络，主要包括用于图像去噪的稀疏块(SB)、特征增强块(FEB)、注意块(AB)和重构块(RB)。具体来说，SB通过使用扩展卷积和普通卷积来消除噪声，在性能和效率之间做出了折衷。该算法通过长路径集成全局和局部特征信息，增强模型的表达能力。该算法用于精细地提取隐藏在复杂背景中的噪声信息，对复杂噪声图像特别是真实噪声图像进行融合降噪是非常有效的。同时，将滤波算法与自适应算法相结合，提高了模型的训练效率，降低了模型训练的复杂度。最后，RB算法的目标是通过获得的噪声映射和给定的噪声图像来构造干净的图像。 


- 论文参考： [Tian, Chunwei, et al. "Attention-guided CNN for image denoising." Neural Networks 124 (2020): 117-129.](https://www.sciencedirect.com/science/article/pii/S0893608019304241) 

- 参考实现：

  ```
  url=https://github.com/cqray1990/ADNet
  branch=master
  commit_id=76560b90045292db020b47901fe5474b84f4c942
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
  | input    | FLOAT32  | batchsize x 1 x 321 x 481 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | output1  | FLOAT32  | batchsize x 1 x 321 x 481 | NCHW         |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| :------------------------------------------------------------: | :-------: | :------------------------------------------------------------: |
| 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   |  \                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

- 该模型需要以下插件与驱动

   **表 2** 依赖配套表

| 配套依赖                 | 版本             |
| :-----------------------: | :----------------: |
| torch                   |  1.12.0          |
| torchvision             |  0.13.1          |
| onnx                    |  1.12.0          |
| onnx-simplifier         |  0.4.8           |
| numpy                   |  1.21.6          |
| Pillow                  |  9.2.0           |
| opencv-python           |  4.6.0.66        |
| scikit-image            |  0.19.3          |
| tqdm                    |  4.64.1          |
| 说明：ADnet模型的适配性较强,<br>未与任何版本有强关联，用户可根据自己环境选选择依赖                    |  \          |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。
   
   源码目录结构如下：(注：可不用下载源码仓代码)

   ```
   ├──ADNet_preprocess.py
   ├──ADNet_postprocess.py
   ├──ADNet_pth2onnx.py
   ├──perf_t4.sh                     //gpu性能测试脚本
   ├──models.py
   ├──utils.py
   ├──LICENCE
   ├──requirements.txt
   ├──modelzoo_level.txt
   ├──prep_dataset
   ├──model_70.pth
   ├──ADNet.onnx
   ├──ADNet_bs1.om
   ├──ADNet_bs16.om
   ```

   
2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```



## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型支持[BSD68 数据集](https://pan.baidu.com/s/1XiePOuutbAuKRRTV949FlQ)共68张图片。用户可自行获取BSD68数据集上传数据集到服务器，可放置于任意路径下，以"/ADNet/dataset"目录为例。可使用百度网盘，提取码：0315。

   数据集目录结构如下:

   ```
      ├──dataset
            ├──BSD68
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行ADNet_preprocess.py脚本，完成预处理。

   ```python
   python3 ADNet_preprocess.py /ADNet/dataset/BSD68 ./prep_dataset
   ```

   预处理prep_dataset目录结构如下：

   ```
      ├──prep_datase
            ├──INoisy         //原图片加入随机噪声处理后文件
            ├──ISoure         //原图片处理后文件
   ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用Torch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       下载权重文件[model_70.pth](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fhellloxiaotian%2FADNet%2Fblob%2Fmaster%2Fgray%2Fg25%2Fmodel_70.pth)至服务器

   2. 导出onnx文件。

      a. 使用ADNet_preprocess.py导出onnx文件。

         运行ADNet_preprocess.py脚本。

         ```python
         python3 ADNet_preprocess.py ./model_70.pth ADNet.onnx
         ```

         获得ADNet.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      a. 配置环境变量。
   
         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```
         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
   
      b. 执行命令查看芯片名称。
   
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
   
      c. 执行ATC命令。

         ```
          atc --framework=5 --model=ADNet.onnx --output=ADNet_bs1 --input_format=NCHW --input_shape="image:1,1,321,481" --log=error --soc_version=Ascend${chip_name}
         ```
   
         * 参数说明：
         -   --model：为ONNX模型文件。
         -   --framework：5代表ONNX模型。
         -   --output：输出的OM模型。
         -   --input\_format：输入数据的格式。
         -   --input\_shape：输入数据的shape。
         -   --log：日志级别。
         -   --soc\_version：处理器型号。
       
         运行成功后生成ADNet_bs1.om模型文件。


2. 开始推理验证。

   a.  使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)。


   b.  执行推理。

      ```
       mkdir -p outbs1/INoisy
       python3 ais_infer.py --model ./ADNet_bs1.om --input ./prep_dataset/INoisy/  --output ./outbs1/INoisy/ --outfmt BIN  --batchsize 1 --device 0
      ```
      - 参数说明：
      -  --model：om文件路径。
      -   --input：输入的bin文件路径。
      -   --output：推理结果文件路径。
      -   --outfmt：输出结果格式。
      -   --device：NPU设备编号。
      -   --batchsize：批大小。

      推理后的输出在推理结果文件路径下的日期+时间的子文件夹(如下文：2022_10_13-03_38_20)。
    
      >**说明：** 
      >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

   c.  精度验证。

      调用脚本与原图片处理后文件比对，可以获得Accuracy数据，结果保存在result.json中。
    
      ```python
       python3 ADNet_postprocess.py ./outbs1/INoisy/2022_10_13-03_38_20/ ./prep_dataset/ISoure > result.json
      ```

      -  参数说明：     
      -   --model：om文件路径。
      -   --input：输入的bin文件路径。
      -   --output：推理结果文件路径。
      -   --outfmt：输出结果格式。
      -   --device：NPU设备编号。
      -   --batchsize：批大小。


   d.  性能验证。

      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
    
      ```python
       python3 ais_infer.py --model ./ADNet_bs1.om --output ./ --batchsize 1 --device 0 --outfmt BIN --loop 5
      ```
      - --loop：推理次数。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| :---------: | :----------------: | :----------: | :----------: | :---------------: |
|    310P       |        1          |     BSD68       |     29.245       |        205         |
|    310P       |        4          |     BSD68       |     29.245       |        183         |
|    310P       |        8          |     BSD68       |     29.245       |        204         |
|    310P       |        16         |     BSD68       |     29.245       |        210         |
|    310P       |        32         |     BSD68       |     29.245       |        214         |
|    310P       |        64         |     BSD68       |     29.245       |        215         |
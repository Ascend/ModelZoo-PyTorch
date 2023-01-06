# SimCLR模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)






# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

SimCLR：一个简单的视觉表示对比学习框架，不仅比以前的工作更出色，而且也更简单，既不需要专门的架构，也不需要储存库。SimCLR利用基于ResNet架构的卷积神经网络变量对图像表示进行计算。SimCLR利用完全连接的网络（即MLP）计算图像表示的非线性投影，实现放大不变的特征并，以及网络识别同一图像的不同变换的能力最大化。SimCLR首先学习未标记数据集上图像的一般表示，然后可以使用少量标记图像对其进行微调，以实现给定分类任务的良好性能。通过采用一种称为对比学习的方法，可以通过同时最大化同一图像的不同变换视图之间的一致性以及最小化不同图像的变换视图之间的一致性来学习通用表示。利用这一对比目标更新神经网络的参数，使得相应视图的表示相互“吸引”，而非对应视图的表示相互“排斥”。




- 参考实现：

  ```
   url=https://github.com/google-research/simclr
   branch=master 
   commit_id=2fc637bdd6a723130db91b377ac15151e01e4fc2
  ```

  
  通过Git获取对应commit\_id的代码方法如下：

  ```
   git clone {repository_url}        # 克隆仓库的代码 
   cd {repository_name}              # 切换到模型的代码仓目录 
   git checkout {branch}             # 切换到对应分支 
   git reset --hard {commit_id}      # 代码设置到对应的commit_id 
   cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ----------------------- | ------------ |
  | input    | RGB_FP16 | batchsize x 3 x 32 x 32 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 1 x 128 | FLOAT32  | ND           |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.8.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>


1. 获取源码
    ```
    git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
    ```

2. 安装依赖。

   ```
   pip install -r requirements.txt 
   ```


## 准备数据集<a name="section183221994411"></a>

获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

获取CIFAR-10数据集
```
#Version：CIFAR-10 python version
```
对压缩包进行解压到/root/datasets文件夹(执行命令：tar -zxvf cifar-10-python.tar.gz -C /root/datasets)，test_batch存放cifar10数据集的测试集图片，文件目录结构如下：
```
root
├── datasets
│   ├── cifar-10-batch-py
│   │   ├── batches.meta
│   │   ├── data_batch_1
|   |   ├── data_batch_2
|   |   ├── data_batch_3
|   |   ├── data_batch_4
|   |   ├── data_batch_5
|   |   ├── test_batch
|   |   ├── readme.html
```


2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行Simclr_preprocess.py脚本，完成预处理。

   ```
   python3.7 Simclr_preprocess.py ./cifar-10-batches-py/test_batch ./prep_data
   ```



## 模型推理<a name="section741711594517"></a>

  1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      ```
       https://pan.baidu.com/s/18sZVnLoQpgIj_nuRpG-XnQ
       提取码：irpw 
      ```

   2. 导出onnx文件。

      1. 使用Simclr_pth2onnx.py导出onnx文件。
         运行Simclr_pth2onnx.py脚本。

         ```
         python3.7 Simclr_pth2onnx.py ./simclr.pth Simclr_model.onnx
         ```

         获得Simclr_model.onnx文件。



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
       atc --framework=5 --model=Simclr_model.onnx --output=Simclr_model_bs1 --input_format=NCHW --input_shape="input:1,3,32,32" --log=info -- 
       soc_version=Ascend${chip_name} --insert_op_conf=aipp.cfg --enable_small_channel=1 --keep_dtype=execeptionlist.cfg
       ```
        - 参数说明:
        
        --model：为ONNX模型文件。
               
        --framework：5代表ONNX模型。
        
        --output：输出的OM模型。
        
        --input\_format：输入数据的格式。
        
        --input\_shape：输入数据的shape。
        
        --log：日志级别。
        
        --soc\_version：处理器型号。
        
        --insert\_op\_conf=aipp\_resnet34.config:  AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。

        --enable_small_channel：Set enable small channel. 0(default): disable; 1: enable。
           
        运行成功后生成Simclr_model_bs1.om模型文件。

  2.开始推理验证  
   a.安装ais_bench推理工具

   请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  
  
   b.执行推理。

```
python3.7 -m ais_bench --model ../Simclr_model_bs1.om --input "../prep_data/" --output ./result/ --outfmt "TXT" --batchsize 1
```
-   参数说明：
     
 -   --input：输入文件夹。
   
 -   --model：om文件路径。
   
 -   --output：输出文件夹。

 -   --outfmt：输出格式。

 -   --batchsize：batchsize.


  c.精度验证
     
调用脚本Simclr_postprocess.py获取，可以获得Accuracy数据，结果保存在log文件中。

```
 python3.7 Simclr_postprocess.py  ./ais_bench/result/2022_07_25-10_41_40/ > result_bs1.log
```
result/2022_07_25-10_41_40/：为生成推理结果所在路径  
    
result_bs1.log：为生成结果文件

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>
精度：
|  设备  |  accuracy1 |
|:-----: |:----------:|
|310     |  65.39   |
|310p    |  65.55   |

调用ACL接口推理计算，性能参考下列数据。

| 模型        |     T4    |     310    | 310p性能  | 310p/310  |  310p/T4  |
| :------:    | :------:  | :------:   | :------: |  :------:  | :------: |
| SimCLR bs1  | 833fps   | 4196fps| 3571fps    |  0.851048  | 4.286914 |
| SimCLR bs4  | 2352fps  | 7772fps| 12903fps   |  1.555970  | 5.485969 |
| SimCLR bs8  | 5333fps  | 11732fps| 17391fps   |  1.482355  | 3.261016 |
| SimCLR bs16 | 6666fps  | 10780fps| 23529fps   |  2.182653  | 3.529702 |
| SimCLR bs32 | 9696fps  | 11320fps| 28070fps   |  2.479681  | 2.895008 |
| SimCLR bs64 | 12075fps | 11644fps| 27586fps   |  2.369117  | 2.284554 |
| 最优 bs | bs64 12075fps | bs64 11732fps| bs32 28070fps   |  2.392601  | 2.324637 |


310p最优batch为:bs32。

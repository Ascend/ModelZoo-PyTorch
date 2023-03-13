# pnasnet5large Onnx模型端到端推理指导
- [概述](#概述)
    - [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)

- [快速上手](#快速上手)

  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)

- [模型推理性能](#模型推理性能)

  ******

  

# 概述<a name="概述"></a>

PNAS是一种学习卷积神经网络（CNN）结构的方法，该方法比现有的基于强化学习和进化算法的技术更有效。使用了基于序列模型的优化(SMBO)策略，在这种策略中，按照增加的复杂性对结构进行搜索，同时学习代理模型（surrogate model）来引导在结构空间中的搜索。
这种方法类似于 A* 算法（也被称为分支限界法），其中从简单到复杂搜索模型空间，并在前进过程中剪枝处理掉没有前途的模型。 这些模型（单元）按照它们所包含的模块的数量进行排序。
从考量带有一个模块的单元开始。评估这些单元（通过训练它们并在一个验证集上计算它们的损失），然后使用观察得到的奖励来训练一个基于 RNN 的启发式函数（也被称为代理函数），其可以预测任何模型的奖励。
接着可以使用这个学习到的启发式函数来决定应该评估哪些带有 2 个模块的单元。在对它们进行了评估之后，再对这个启发式函数进行更新，重复这一过程，直到我们找到带有所想要的模块数量的优良单元。

- 参考论文：[Progressive Neural Architecture Search](https://arxiv.org/pdf/1712.00559.pdf)

- 参考实现：

  ```
  url=https://github.com/rwightman/pytorch-image-models
  branch=master
  commit_id=7096b52a613eefb4f6d8107366611c8983478b19
  ```


## 输入输出数据<a name="输入输出数据"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- |---------------------------| ----------------- | -------- |
  | image   | RGB_FP32 | batchsize x 3 x 331 x 331 | NCHW       |


- 输出数据

  | 输出数据 | 大小                   | 数据类型 | 数据排布格式 |
  |----------------------| -------- |--------| ------------ |
  | class    | batchsize x 1000 | FLOAT32  | ND     |


# 推理环境准备<a name="推理环境准备"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本      | 环境准备指导                                                 |
| ------------------------------------------------------------ |---------| ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fpies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.8.1   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 |         |                                                              |



# 快速上手<a name="快速上手"></a>

## 获取源码<a name="获取源码"></a>

1. 获取源码。

   pnasnet5large模型代码在timm里，安装timm，arm下需源码安装，参考https://github.com/rwightman/pytorch-image-models
，若安装过程报错请百度解决
   ```
   rm -r pytorch-image-models
   git clone https://github.com/rwightman/pytorch-image-models.git
   cd pytorch-image-models
   python3.7.5 setup.py install
   cd ..
   ```

2. 安装依赖。

   ```
   pip3.7.5 install -r requirements.txt
   ```

​		


## 准备数据集<a name="准备数据集"></a>

1. 获取原始数据集。

   该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在 /home/datasets/imagenet/val 与 /home/datasets/imagenet/val_label.txt。

   解压后数据集目录结构：
   ```
   imagenet
   ├── val_label.txt    //验证集标注信息       
   └── val             // 验证集文件夹
   ```

2. 数据预处理。

   将原始数据转化为二进制文件（.bin）。

   执行imagenet_torch_preprocess.py脚本，生成数据集预处理后的bin文件，存放在当前目录下的prep_dataset文件夹中。

   ```
   python3.7.5 imagenet_torch_preprocess.py /home/datasets/imagenet/val ./prep_dataset
   ```
   
   - 参数说明
     - /home/datasets/imagenet/val：数据集的路径。
     - ./prep_dataset：生成的bin文件路径。


## 模型推理<a name="模型推理"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       由于源代码问题，加载下载好的[权重文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/PNASNet5Large/PTH/pnasnet5large-bf079911.pth)会报错，所以选择根据脚本自动下载权重文件。

   2. 导出onnx文件。

      1. 使用pnasnet5large_onnx.py导出onnx文件，脚本会自动下载权重文件。

         运行使用pnasnet5large_onnx.py脚本。

         ```
         python3.7.5 pnasnet5large_onnx.py pnasnet5large.onnx
         ```

         获得pnasnet5large.onnx文件。
         - 参数说明：
             - pnasnet5large.onnx：生成的onnx文件。

      2. 优化ONNX文件。

         ```
         python3.7.5 -m onnxsim  --overwrite-input-shape="-1,3,331,331" pnasnet5large.onnx pnasnet5large_sim.onnx
         ```

         获得pnasnet5large_sim.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```
         
         > **说明：** 
         该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
      
      2. 执行命令查看芯片名称（$\{chip\_name\}）。
      
         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         回显如下：
         +-------------------|-----------------|------------------------------------------------------+
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
      
         使用atc将onnx模型转换为om模型文件，工具使用方法可以参考《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。生成转换batch size为16的om模型的命令如下，对于其他的batch size，可作相应的修改。
         
         ```
         atc --framework=5 --model=./pnasnet5large_sim.onnx --input_format=NCHW --input_shape="image:4,3,331,331" \
         --output=pnasnet5large_bs4 --log=error --soc_version=Ascend${chip_name}
         ```
      
         - 参数说明：
           -   --framework：5代表ONNX模型。
           -   --model：为ONNX模型文件。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --output：输出的OM模型。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
 
         运行成功后生成pnasnet5large_bs4.om模型文件。
         
2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        python3.7.5 -m ais_bench --model pnasnet5large_bs4.om --batchsize 4 --input ./prep_dataset --output ./result --outfmt "TXT" --device 0
        ```
        - 参数说明：
            - --model: 需要进行推理的om离线模型文件。
            - --batchsize: 模型batchsize。
            - --input: 模型需要的输入，指定输入文件所在的目录即可。
            - --output: 推理结果保存目录。结果会自动创建”日期+时间“的子目录，保存输出结果。可以使用--output_dirname参数，输出结果将保存到子目录output_dirname下。
            - --outfmt: 输出数据的格式。设置为"TXT"用于后续精度验证。
            - --device: 指定NPU运行设备。取值范围为[0,255]，默认值为0。

            推理后的输出默认在当前目录result下。

   3. 精度验证。

        调用RefineDet_postprocess.py脚本，可以获得Accuracy数据，精度结果保存在result_bs4.json中。

        ```
        python3.7.5 vision_metric_ImageNet.py result/2023_01_06-02_56_00 /home/datasets/imagenet/val_label.txt ./ result_bs4.json
        ```

        - 参数说明：
          - 第一个参数为生成推理结果所在路径,请根据ais_bench推理工具自动生成的目录名进行更改。
          - 第二个参数为数据集配套标签。
          - 第三个参数为生成文件的保存目录。
          - 第四个参数为生成的精度结果文件名。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3.7.5 -m ais_bench --model pnasnet5large_bs4.om --batchsize 4 --output ./result --loop 1000 --device 0
        ```

      - 参数说明：
        - --model：需要进行推理的om模型。
        - --batchsize：模型batchsize。不输入该值将自动推导。当前推理模块根据模型输入和文件输出自动进行组batch。参数传递的batchszie有且只用于结果吞吐率计算。请务必注意需要传入该值，以获取计算正确的吞吐率。
        - --output: 推理结果输出路径。默认会建立"日期+时间"的子文件夹保存输出结果。
        - --loop: 推理次数。默认值为1，取值范围为大于0的正整数。
        - --device: 指定NPU运行设备。取值范围为[0,255]，默认值为0。

   ​	

# 模型推理性能&精度<a name="模型推理性能&精度"></a>

调用ACL接口推理计算，精度和性能参考下列数据。

|   芯片型号   | Batch Size |    数据集     | 精度acc1 |   性能    |
|:--------:|:----------:|:----------:|:------:|:-------:|
|  310P3   |     1      |  ImageNet  | 81.76% | 66.329  |
|  310P3   |     4      |  ImageNet  | 81.90% | 203.256 |
|  310P3   |     8      |  ImageNet  | 81.14% | 177.719 |
|  310P3   |     16     |  ImageNet  | 81.32% | 164.340 |
|  310P3   |     32     |  ImageNet  | 81.20% | 154.361 |
|  310P3   |     64     |  ImageNet  | 81.16% | 148.531 |

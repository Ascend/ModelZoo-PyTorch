# CTPN模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

******





# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

CTPN是一种文字检测算法，它结合了CNN与LSTM深度网络，能有效的检测出复杂场景的横向分布的文字CTPN。作者开发了一种垂直锚定机制，可以联合预测每个固定宽度提议的位置和文本/非文本得分，大大提高了定位精度。序列提议通过循环神经网络自然连接，并与卷积网络无缝结合，形成一个端到端的可训练模型，这使得CTPN可以探索丰富的图像上下文信息，使其强大的检测极其模糊的文本。CTPN可以在多尺度和多语言文本上可靠地工作，而无需进一步的后处理，这与以前自下而上的方法需要多步后处理不同。CTPN只预测文本的竖直方向上的位置，水平方向的位置不预测，从而检测出长度不固定的文本。

- 论文参考： [Tian Z, Huang W, He T, et al. Detecting text in natural image with connectionist text proposal network[C]//European conference on computer vision. Springer, Cham, 2016: 56-72.](https://www.semanticscholar.org/paper/Detecting-Text-in-Natural-Image-with-Connectionist-Tian-Huang/b620548e06b03e3cc2e9e775c39f5b3d5a4eb19a) 


- 参考实现：

  ```
  url=https://github.com/CrazySummerday/ctpn.pytorch
  branch=master
  commit_id=99f6baf2780e550d7b4656ac7a7b90af9ade468f
  ```
  

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32  | batchsize x 3 x h x w | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | output1  | FLOAT32  | batchsize x ((h // 16) * (w // 16)) x 2 | ND         |
  | output2  | FLOAT32  | batchsize x ((h // 16) * (w // 16)) x 2 | ND         |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

   | 配套                                                         | 版本    | 环境准备指导                                                 |
   | :------------------------------------------------------------: | :-------: | :------------------------------------------------------------: |
   | 固件与驱动                                                   | 1.0.16  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
   | CANN                                                         | 5.1.RC2 | -                                                            |
   | Python                                                       | 3.7.5   |  \                                                            |
   | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/CrazySummerday/ctpn.pytorch.git -b master
   cd ctpn.pytorch
   git reset 99f6baf2780e550d7b4656ac7a7b90af9ade468f –hard
   cd ..
   ```

   源码目录结构如下：

   ```
   ├──ctpn.pytorch                      //开源仓目录
   ├──ctpn_preprocess.py
   ├──ctpn_postprocess.py
   ├──ctpn_pth2onnx.py
   ├──config.py
   ├──image_kmeans.py
   ├──task_process.py
   ├──LICENCE
   ├──requirements.txt
   ├──README.md
   ├──modelzoo_level.txt
   ├──performance_gpu.py
   ```

   
2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型支持[ICDAR2013 数据集](https://gitee.com/link?target=https%3A%2F%2Frrc.cvc.uab.es%2F%3Fch%3D2)及相应[精度评测代码](https://gitee.com/link?target=https%3A%2F%2Frrc.cvc.uab.es%2Fstandalones%2Fscript_test_ch2_t1_e2-1577983067.zip)。用户可自行获取ICDAR2013数据集及评测方法代码上传到服务器，可放置于任意路径下，以"./datasets"和"./script"目录为例。

   ```
   ├──datasets
         ├──Challenge2_Test_Task12_Images
   ├──script                                 //精度验证时会用到
         ├──gt.zip
         ├──readme.txt
         ├──rrc_evaluation_funcs_1_1.py
         ├──script.py 
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   因为该模型根据图片输入形状采用分档输入，一共分为了10档，因此需要生成不同分辨率的预处理文件，为简化步骤、避免浪费不必要的时间，直接将相应的预处理程序放在任务处理的"task_process.py"脚本中，该脚本会自动删除和创建数据预处理的文件夹，以及调用预处理“ctpn_preprocess.py”程序。执行task_process.py脚本，完成预处理。

   ```python
   python3 task_process.py --interpreter=python3 --mode=preprocess --src_dir=./datasets/Challenge2_Test_Task12_Images --res_dir ./pre_bin/images_bin
   ```
   - 参数说明：
      - --interpreter:解释器路径。
      - --mode：脚本处理的方式。
      - --src_dir：输入文件的目录。
      - --res_dir：得到的文件的目录。

   预处理后生成结果目录结构如下：

   ```
   ├──pre_bin
         ├──images_bin_248x360        
         ├──images_bin_1000x462
         ├──images_bin_280x550
         ├──images_bin_650x997
         ├──images_bin_458x440
         ├──images_bin_319x973
         ├──images_bin_997x744
         ├──images_bin_631x471
         ├──images_bin_477x636
         ├──images_bin_753x1000        
   ```

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用Torch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      权重文件在./ctpn.pytorch/weights/ 目录下，文件名称为ctpn.pth。

   2. 导出onnx文件。

      1. 使用ctpn_pth2onnx.py导导出onnx文件。

         运行ctpn_pth2onnx.py导脚本。

         ```python
         python3 ctpn_pth2onnx.py --pth_path=./ctpn.pytorch/weights/ctpn.pth --onnx_path=./
         ```

         - 参数说明
            - --pth_path：权重文件路径。
            - --onnx_path：生成的onnx文件路径


         获得onnx文件如下：
         ```
         ├──./
            ├──ctpn_280x550.onnx        
            ├──ctpn_248x360.onnx
            ├──ctpn_319x973.onnx
            ├──ctpn_458x440.onnx
            ├──ctpn_477x636.onnx
            ├──ctpn_631x471.onnx
            ├──ctpn_650x997.onnx
            ├──ctpn_753x1000.onnx
            ├──ctpn_997x744.onnx
            ├──ctpn_1000x462.onnx      
         ```

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称。

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
         atc --framework=5 --model=ctpn_1000x462.onnx --output=ctpn_bs1 --input_format=NCHW --input_shape="image:1,3,-1,-1" --dynamic_image_size="248,360;280,550;319,973;458,440;477,636;631,471;650,997;753,1000;997,744;1000,462" --log=error --soc_version=Ascend${chip_name}
         ```

         - 参数说明：
            -   --model：为ONNX模型文件。
            -   --framework：5代表ONNX模型。
            -   --output：输出的OM模型。
            -   --input\_format：输入数据的格式。
            -   --input\_shape：输入数据的shape。
            -   --log：日志级别。
            -   --soc\_version：处理器型号。
            -   --dynamic\_image\_size：设置输入图片的动态分辨率参数。

         运行成功后生成ctpn_bs1.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      ```
      python3 task_process.py --interpreter="python3 -m ais_bench" --om_path=./ctpn_bs1.om --src_dir=./pre_bin/images_bin --res_dir=./result --batch_size=1 --device=0
      ```
      - 参数说明：
         - --interpreter:推理工具。
         - --model：om文件路径。
         - --input：输入的bin文件路径。
         - --output：推理结果文件路径。
         - --outfmt：输出结果格式。
         - --device：NPU设备编号。
         - --res_dir：得到的结果文件夹。

      推理后的输出在推理结果文件路径下result文件夹。

      性能计算方式：

      设输入数据根据宽高的不同分为 $n$ 组，第 $i$ 组的性能为 $f_i$，第 $i$ 组的数据集大小为 $s_i$，则模型的综合性能的计算公式为：
      $$
      performance = \frac{\sum_i^n f_i*s_i}{\sum_i^ns_i}
      $$


   3. 精度验证。

      调用脚本与原图片处理后文件比对，可以获得Accuracy数据，结果保存在result.json中。
    
      ```python
      python3 ctpn_postprocess.py --imgs_dir=./datasets/Challenge2_Test_Task12_Images --bin_dir=./result --predict_txt=./result/predict_txt
      zip -j ./script/predict_txt.zip ./result/predict_txt/*
      python3 script/script.py -g=./script/gt.zip –s=./script/predict_txt.zip > result.json
      ```

      -  参数说明：     
         - --model：om文件路径。
         - --input：输入的bin文件路径。
         - --output：推理结果文件路径。
         - --outfmt：输出结果格式。
         - --device：NPU设备编号。
         - --batchsize：批大小。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| :---------: | :----------------: | :----------: | :----------: | :---------------: |
|    Ascend310P       |        1          |     ICDAR2013       |     precision: 86.84%;recall: 75.05%;hmean: 80.51%       |        141.11         |
|    Ascend310P       |        4          |     ICDAR2013       |     precision: 86.84%;recall: 75.05%;hmean: 80.51%       |        153.24         |
|    Ascend310P       |        8          |     ICDAR2013       |     precision: 86.84%;recall: 75.05%;hmean: 80.51%       |        167.68         |
|    Ascend310P       |        16          |     ICDAR2013       |     precision: 86.84%;recall: 75.05%;hmean: 80.51%       |        162.42         |
|    Ascend310P       |        32          |     ICDAR2013       |     precision: 86.84%;recall: 75.05%;hmean: 80.51%       |        164.31         |
|    Ascend310P       |        64          |     ICDAR2013       |     precision: 86.84%;recall: 75.05%;hmean: 80.51%       |        160.14         |
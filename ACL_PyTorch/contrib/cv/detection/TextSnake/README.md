# TextSnake模型-推理指导


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

[TextSnake论文](https://arxiv.org/abs/1807.01544)
论文主要提出了一种能够灵活表示任意弯曲形状文字的数据结构——TextSnake，主要思想是使用多个不同大小，带有方向的圆盘(disk)对标注文字进行覆盖，并使用FCN来预测圆盘的中心坐标，大小和方向进而预测出场景中的文字


- 参考实现：

  ```
  url=https://github.com/princewang1994/TextSnake.pytorch
  commit_id=b4ee996d5a4d214ed825350d6b307dd1c31faa07
  model_name=contrib/cv/detection/TextSnake
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 512 x 512 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | 1 x 7 x 512 x 512 | ND           |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.6.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码

   ```
   git clone https://github.com/princewang1994/TextSnake.pytorch
   cd TextSnake.pytorch
   git reset --hard b4ee996d5a4d214ed825350d6b307dd1c31faa07
   cd ..
   ```

2. 安装依赖

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

  首先进入TextSnake.pytorch/dataset/total-text文件夹中,根据[源码仓](https://github.com/princewang1994/TextSnake.pytorch/tree/master/dataset/total_text/download.sh)的方式下载数据集并整理成gt文件夹和Images文件夹。回到TextSnake目录下新建data文件夹，进入data文件夹，创建total-text文件夹，将第一步生成的Images/Test移动到total-text中，将gt/Test移动到total-text中。目录如下：

   ```
      data
         |——total-text
         ├── gt
            |——Test     
         └── Images
            |——Test 
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行TextSnake_preprocess.py脚本，完成预处理

   ```
   python3 TextSnake_preprocess.py --src_path ./data/total-text/Images/Test --save_path ./total-text-bin
   ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件

      ```
      wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/TextSnake/PTH/textsnake_vgg_180.pth
      ```

   2. 导出onnx文件。

      1. 使用TextSnake_pth2onnx.py导出onnx文件。

         运行TextSnake_pth2onnx.py脚本。

         ```
         python3 TextSnake_pth2onnx.py --input_file './textsnake_vgg_180.pth'  --output_file './TextSnake.onnx'
         ```

         获得TextSnake.onnx文件。


   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

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
          atc --model=TextSnake.onnx \
            --framework=5 \
            --output=TextSnake_bs1 \
            --input_format=NCHW \
            --input_shape="image:1,3,512,512" \
            --log=error \
            --soc_version=Ascend${chip_name} \
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
          

           运行成功后生成`TextSnake_bs1.om`模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
         python3 -m ais_bench --model TextSnake_bs1.om \
	      --input ./total-text-bin \ 
	      --output ./ \
	      --output_dirname result \
	      --outfmt TXT \
        ```

        -   参数说明：

             - --model：om模型
             - --input：输入文件
             - --output：输出路径
             - --output_dirname: 输出结果文件夹
             - --outfmt: 输出格式
                  	

        推理后的输出默认在当前目录result下。


   3. 精度验证。

      调用TextSnake_postprocess.py，精度会打屏显示

      ```
       python TextSnake_postprocess.py result
      ```

   4. 性能验证

   可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
	```
	python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size} 
	```
	- 参数说明
		- --model：om模型
		- --loop：循环次数
		- --batchsize：推理张数



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度 | 性能 |
| -------- | ---------- | ------ | ---- | ---- |
|     310P3     |    1        |  Total-Text-Dataset      | tr0.7 & tp0.6:f1:0.59<br>tr0.8 & tp0.4:f1:0.78     |  180.36    |
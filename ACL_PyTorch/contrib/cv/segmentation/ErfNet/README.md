# ErfNet模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

  - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&性能](#ZH-CN_TOPIC_0000001172201573)

  ******

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

ErfNet(Efficient Residual Factorized Network)是一个能够实现准确和快速的像素级别语义分割的架构。该架构采用了重新设计的残差层，提升了效率，使自身在可靠性和速度之间获得了一个很好的权衡。ErfNet适用于如自动驾驶汽车中的场景理解这种需要稳健性和实时操作的应用。

- 参考实现：

  ```
  url=https://github.com/Eromera/erfnet_pytorch
  branch=master
  commit_id=d4a46faf9e465286c89ebd9c44bc929b2d213fb3
  model_name=ErfNet
  ``` 
 
  通过Git获取对应commit_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据        | 数据类型  | 大小                       | 数据排布格式  |
  | -------------- | -------- | -------------------------- | ------------ |
  | actual_input_1 | FLOAT32  | batchsize x 3 x 512 x 1024 | NCHW         |


- 输出数据

  | 输出数据  | 数据类型  | 大小                        | 数据排布格式  |
  | -------- | -------- | --------------------------- | ------------ |
  | output1  | FLOAT32  | batchsize x 20 x 512 x 1024 | NCHW         |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套           | 版本    | 环境准备指导              |
  | ------------  | ------- | ------------------------ |
  | 固件与驱动     | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN          | 6.0.0   | -                        |
  | Python        | 3.7.5   | -                        |
  | PyTorch       | 1.8.0   | -                        |  

- 该模型需要以下依赖   

  **表 2**  依赖列表

  | 依赖名称               | 版本                    |
  | --------------------- | ----------------------- |
  | torchvision           | >= 0.6.0                |
  | onnx                  | >= 1.7.0                |
  | numpy                 | 1.20.2                  |
  | Pillow                | 7.2.0                   |
  | opencv-python         | 4.5.2.52                |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

    ```
    git clone https://github.com/Eromera/erfnet_pytorch   
    cd erfnet_pytorch  
    git reset d4a46faf9e465286c89ebd9c44bc929b2d213fb3 --hard
    cd .. 
    ```

2. 安装依赖。
  
    ```
    pip3 install -r requirements.txt
    ```

    **说明：** 
    >   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3 install 包名 安装
    >
    >   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3 install 包名 安装

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

    - 下载[Cityscapes dataset](https://www.cityscapes-dataset.com/)数据集。
      - 下载名为"leftImg8bit"的RGB图像和名为"gtFine"的标签。
      - 注意：在训练时，应该使用"_labelTrainIds"而非"_labelIds"，可以下载[cityscapes scripts](https://github.com/mcordts/cityscapesScripts)和[conversor](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py)，通过两者将labelIds转换为trainIds。 

2. 数据预处理。

    1. 执行“ErfNet_preprocess.py”脚本将原始数据集转换为模型输入的数据。

        ```
        python3 ErfNet_preprocess.py ${datasets_path}/cityscapes/leftImg8bit/val ./prep_dataset ${datasets_path}/cityscapes/gtFine/val ./gt_label
        ```

      - 参数说明：
        - ${datasets_path}/cityscapes/leftImg8bit/val：数据集路径。（请用数据集准确路径替换{datasets_path}）
        - ./prep_dataset：输出文件路径。
        - ${datasets_path}/cityscapes/gtFine/val：数据集路径。（请用数据集准确路径替换{datasets_path}）
   
## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      下载权重文件[erfnet_pretrained.pth](https://github.com/Eromera/erfnet_pytorch/blob/master/trained_models/erfnet_pretrained.pth)，放到当前目录。
       
   2. 导出onnx文件。

      执行ErfNet_pth2onnx.py脚本，生成onnx模型文件。由于使用原始的onnx模型转出om后，精度有损失，故添加了modify_bn_weights.py来修改转出onnx模型bn层的权重。

        ```
        python3 ErfNet_pth2onnx.py erfnet_pretrained.pth ErfNet_origin.onnx
        python3 modify_bn_weights.py ErfNet_origin.onnx ErfNet.onnx
        ```

        获得ErfNet.onnx文件。

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
             +--------------------------------------------------------------------------------------------+
             | npu-smi 22.0.0                       Version: 22.0.2                                       |
             +-------------------+-----------------+------------------------------------------------------+
             | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
             | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
             +===================+=================+======================================================+
             | 0       310P3     | OK              | 17.0         56                0    / 0              |
             | 0       0         | 0000:AF:00.0    | 0            934  / 23054                            |
             +===================+=================+======================================================+
         ```

      3. 执行ATC命令。
         ```
         atc --framework=5 --model=ErfNet.onnx --output=ErfNet_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,512,1024" --log=error --soc_version=${chip_name} --output_type=FP16
         ```

         - 参数说明：

            - --model：为ONNX模型文件
            - --framework：5代表ONNX模型
            - --output：输出的OM模型
            - --input\_format：输入数据的格式
            - --input\_shape：输入数据的shape
            - --log：日志级别
            - --soc\_version：处理器型号
            - --output_type: 网络输出类型

          运行成功后生成ErfNet_bs1.om模型文件。

2. 开始推理验证。
   
   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。
      ```
      python3 -m ais_bench --model=${user_path}/ErfNet/ErfNet_bs1.om --input ${user_path}/ErfNet/prep_dataset/ --output ${user_path}/output/ --outfmt BIN --batchsize 1
      ```

      - 参数说明：

         - --model: om模型的路径

         - --input: 输入的bin文件目录
       
         - --output: 推理结果输出路径
      
         - --outfmt: 输出数据的格式

         - --batchsize: 模型输入批次大小

         - ${user_path}: 用户个人文件准确路径替换

   3. 精度验证。

      后处理统计精度， 执行后处理脚本进行精度验证。

      ```
      python3 ErfNet_postprocess.py ${user_path}/output/2022_07_15-14_16_46/sumary.json ${user_path}/ErfNet/gt_label/
      ```

      - 参数说明：

        - “${user_path}/output/2022_07_15-14_16_46/sumary.json”：ais_bench推理结果汇总数据保存路径。

        - “${user_path}/ErfNet/gt_label/”：合并后的验证集路径。
   
   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model=ErfNet_bs1.om --loop 1000 --batchsize 1
      ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

1. 性能对比
    
    T4、310、310P3性能对比参考下列数据。

    | batchsize | T4       | 310          | 310P    | 310P/310 | 310P/T4 | 310P-AOE | 310P-AOE/310 | 310P-AOE/T4 |
    | --------- | -------- | ------------ | ------- | -------- | ------- | -------- | ------------ | ----------- |
    | 1         | 215.8135 | 220.2016     | 292.137 | 1.327    | 1.35    | 380.0822 | 1.73         | 1.76        |
    | 4         | 226.745  | 176.2568     | 192.398 | 1.09     | 0.849   | 379.2337 | 2.15         | 1.67        |
    | 8         | 237.8234 | 176.6912     | 210.707 | 1.1925   | 0.886   | 381.9549 | 2.16         | 1.6         |
    | 16        | 250.2417 | 175.8732     | 211.125 | 1.2      | 0.84    | 379.9660 | 2.16         | 1.51        |
    | 32        | 222.312  | 181.5056     | 215.234 | 1.186    | 0.9682  | 380.1598 | 2.09         | 1.71        |
    | 64        | 226.2459 | 内存分配失败 | 216.937 | /        | 0.96    | 226.5988 | /            | 1.001       |
    | 最优batch | 250.2417 | 220.2016     | 292.137 | 1.33     | 1.17    | 381.9545 | 1.73         | 1.53        |

    经过对比，AOE调优后的性能结果已经达到交付要求。

2. 精度对比

    310、310P3精度参考下列数据。

    | batch | 310   | 310P  |
    | ----- | ----- | ----- |
    | 1     | 72.20 | 72.20 |
    | 4     | 72.20 | 72.20 |
    | 8     | 72.20 | 72.20 |
    | 16    | 72.20 | 72.20 |
    | 32    | 72.20 | 72.20 |

    [官网pth精度](https://github.com/Eromera/erfnet_pytorch)

    ```
    iou:72.20
    ```

    将得到的om离线模型推理IoU精度与该模型github代码仓上公布的精度对比，310与710上的精度下降在1%范围之内，故精度达标。


备注：
1. 由于使用原始的onnx模型转出om后，精度有损失，故添加了modify_bn_weights.py来修改转出onnx模型bn层的权重。
2. 由于tensorRT不支持部分算子，故gpu性能数据使用在线推理的数据。
# 3D_ResNet_ID0421模型-推理指导  


- [概述](#ZH-CN_TOPIC_0000001172161501)
   
   - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

3D-CNN网络在传统卷积网络的基础上引入了3D卷积，使得模型能够提取数据的时间维度和空间维度特征，从而能够完成更复杂的图像识别或者动作识别任务。3D-ResNets将经典的残差网络结构Resnets与3D卷积结合，在动作识别领域达到了SOA水平。


- 参考实现：

  ```
  url=https://github.com/kenshohara/3D-ResNets-PyTorch
  branch=master
  commit_id=540a0ea1abaee379fa3651d4d5afbd2d667a1f49
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据  | 大小      | 数据类型                | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    |  10 x 3 x 16 x 112 x 112 | FLOAT32|  NCDHW|


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 10 x 51 | FLOAT32  | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.16（NPU驱动固件版本为5.1.RC2）  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |                                                           |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>


1. 获取源码。

    源码目录结构：
    ``` 
    ├── 3D-ResNets_postprocess.py              //模型后处理脚本  
    ├── 3D-ResNets_preprocess.py               //模型前处理脚本 
    ├── 3D-ResNets_pth2onnx.py                 //用于转换pth文件到onnx文件  
    ├── eval_accuracy.py                          //模型精度输出文件
    ├── resnet.patch                            //修改开源仓resnet.py文件的patch文件
    ├── modelzoo_level.txt                          //模型精度性能结果
    ├── requirements.txt                            //依赖库和版本号
    ├── LICENSE                                     //Apache LICENCE                            
    ├── README.md                                   //模型离线推理说明README
    ```
2. 获取开源代码仓并整理代码结构。
   ```
   git clone https://github.com/kenshohara/3D-ResNets-PyTorch.git
   mv 3D-ResNets_postprocess.py 3D-ResNets_preprocess.py 3D-ResNets_pth2onnx.py eval_accuracy.py 3D-ResNets-PyTorch/
   ```


3. 修改resnet文件。在源码路径3d_resnets目录下执行patch命令。
    ```
    patch -p0 < resnet.patch
    ```

4. 安装依赖。

   ```
   pip3 install -r requirements.txt
   cd 3D-ResNets-PyTorch/
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型基于hmdb51数据集训练和推理，hmdb51是一个轻量的动作识别数据汇集，包含51种动作的短视频。hmdb51数据集的获取及处理参考[源代码仓](https://github.com/kenshohara/3D-ResNets-PyTorch)中Preparation和HMDB-51小节。
   数据目录结构请参考：
    ```
    ├──hmdb51
        ├──brush_hair
        |  ├──April_09_brush_hair_u_nm_np1_ba_goo_0
        |  |  ├──image_00001.jpg
        |  |  ├──image_00002.jpg
        |  |  ...
        |  ├──April_09_brush_hair_u_nm_np1_ba_goo_1
        |  |  ├──image_00001.jpg
        |  |  ├──image_00002.jpg
        |  |  ...
        |  ...
        ├──cartwheel
        |  ├──(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0
        |     ├──image_00001.jpg
        |     ├──image_00002.jpg
        |     ...
        |  ├──Acrobacias_de_un_fenomeno_cartwheel_f_cm_np1_ba_bad_8
        |     ├──image_00001.jpg
        |     ├──image_00002.jpg
        |     ...
        |  ...
        ...
    ```

2. 数据预处理。

   数据预处理将图片数据转换为模型输入的二进制数据，将原始数据（.jpg）转化为二进制文件（.bin）。
   在3D-ResNets-PyTorch目录下，执行3D-ResNets_postprocess.py脚本。

   ```
   python3 3D-ResNets_preprocess.py \
      --video_path=hmdb51 \
      --annotation_path=../hmdb51_1.json \
      --output_path=Binary_hmdb51 \
      --dataset=hmdb51 \
      --inference_batch_size=1
   ```
    - 参数说明：  
       
      - --video_path：原始数据集的路径。
      - --annotation_path：数据集信息路径。
      - --output_path：输出目录。
      - --dataset：数据集类型，默认hmdb51。
      - --inference_batch_size：推理数据batch_size。

   运行完预处理脚本会在当前目录输出hmdb51.info文件和Binary_hmdb51二进制文件夹，包含视频片段名字和长度信息，用于后处理。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件 [save_700.pth](https://pan.baidu.com/s/1Hcgd8AAObItOO5-H3n5vdQ)，提取码：jctn。

   2. 导出onnx文件。

      将模型权重文件.pth转换为.onnx文件。

         在3D-ResNets-PyTorch目录下，执行3D-ResNets_pth2onnx.py脚本将.pth文件转换为.onnx文件，执行如下命令。
         ```
         python3 3D-ResNets_pth2onnx.py \
            --root_path=./ \
            --video_path=hmdb51 \
            --annotation_path=../hmdb51_1.json \
            --result_path=./ \
            --dataset=hmdb51 \
            --model=resnet \
            --model_depth=50 \
            --n_classes=51 \
            --resume_path=save_700.pth
         ```
         - 参数说明：
             - --root_path：工作目录。
             - --video_path：原始数据验证集所在路径。
             - --annotation_path：hmdb51_1.json文件所在目录。
             - --result_path：生成的中间文件所在目录。
             - --dataset：数据集类型。
             - --model：模型类型。
             - --model_depth：resnet模型的深度。
             - --n_classes：数据集类型数。
             - --resume_path：权重文件所在路径。

        运行成功后，在当前目录生成3D-ResNets.onnx模型文件。


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

      3. 执行ATC命令。此模型当前仅支持batch_size=10。

         ```
         atc --model=3D-ResNets.onnx \
            --framework=5 \
            --output=3D-ResNets \
            --input_format=NCHW \
            --input_shape="input:10,3,16,112,112" \
            --log=info \
            --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           
           运行成功后生成 3D-ResNets.om 模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。
      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 创建推理结果保存的文件夹。
      ```
      mkdir result
      ```

   3. 执行推理命令。

      ```
      python3 -m ais_bench --model=3D-ResNets.om --input=Binary_hmdb51 --output=result --batchsize=10
      ```
        -  参数说明：
    
           - --model：模型地址。
           - --input：预处理完的数据集文件夹。
           - --output：推理结果保存路径。
           - --batchsize：om模型的batchsize。
    
        推理后的输出默认在当前目录result下。
    

    4. 精度验证。

       1. 运行后处理脚本3D-ResNets_postprocess.py将推理结果处理成json文件。
    
          ```
          python3 3D-ResNets_postprocess.py result/${result_path} 1
          ```
          -  参数说明：
    
              - result/${result_path}：ais_bench工具生成的推理结果的路径。
              - 1：选择统计精度的topK的K值，如1表示统计top 1精度。

            运行成功后生成val.json文件。

       2. 运行eval_accuracy.py脚本与数据集标签hmdb51_1.json比对，可以获得Accuracy数据。
    
          ```
          python3 eval_accuracy.py ../hmdb51_1.json val.json --subset=validation -k=1 --ignore
          ```
          -  参数说明：
    
             - ../hmdb51_1.json：数据集的标签文件。
             - val.json 是后处理输出的json文件。
             - --subset：选择评测的子集，默认为validation。
             - -k：选择统计精度的topK的K值，如1表示统计top 1精度。
             - --ignore：忽略缺失数据。

    5. 性能验证。

       可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

       ```
       python3 -m ais_bench --model=3D-ResNets.om --loop=20 --batchsize=10
       ```
       - 参数说明：
            - --model：om模型的路径
            - --loop: 推理次数
            - --batchsize：om模型的batchsize


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| 310P3         | 10               | hmdb51  | 0.6222     | 830.7165  |
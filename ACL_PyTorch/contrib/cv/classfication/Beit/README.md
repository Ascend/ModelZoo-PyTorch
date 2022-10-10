# beit模型-推理指导


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
beit模型离线推理，采用imagenet数据集中的val数据，以及val_label.txt文件进行精度测试。


- 参考实现：

  ```
  url=git clone https://github.com/microsoft/unilm.git
  branch=master
  model_name=beit
  commit_id=35d21904a9b5beca074b085869d06b9583db2e81
  ```
  
  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径
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

| 配套            | 版本     | 环境准备指导                                                 |
|---------------| -------- | ------------------------------------------------------------ |
| 固件与驱动        | 1.0.15   | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN          | 5.1.RC2  | -                                                            |
| Python        | 3.7.5    | -                                                            |
| PyTorch       | 1.12.0   | -                                                            |
| torchvision   | 0.13.0   | -                                                            |
| Numpy         | 1.21.2   | -                                                            |
| Pillow        | 9.2.0    |                                                              |
| Onnx-simplifier | 0.4.1    |                                                              |
| Pillow        | 9.2.0    |                                                              |
| onnxruntime   | 1.12.0   |                                                              |
| Magiconnx     | 0.1.0    | 获取工具及使用方法可以参考 `https://gitee.com/Ronnie_zheng/MagicONNX/tree/master#1-magiconnx%E7%AE%80%E4%BB%8B` |
| opencv-python | 4.6.0.66 |                                                              |
| timm          | 0.4.12   |                                                              |
| decorator     |          |                                                              |
| tqdm          |          |                                                              |


说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>
0. 获取源码
    ```
    git clone https://github.com/microsoft/unilm.git   
   ```

1. 安装依赖。

   ```
   pip3 install -r requirment.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   数据集选择使用imagenet数据集中的val数据集（ILSVRC2012_img_val.tar）以及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。
   [下载链接]( https://www.image-net.org)

   数据结构如下：

    ```
    ├── ImageNet
        ├── ILSVRC2012_img_val
            ├── val_label.txt
    ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行*beit_preprocess.py*，完成预处理。

   ```
   python3 beit_preprocess.py --image_path="/opt/npu/imageNet/val" --prep_image="./prep_image_bs8" --batch_size=8
   ```

    其中三个参数为：imageNet/val数据集路径、预处理结果输出路径、batch size


## 模型推理<a name="section741711594517"></a>

一. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      ```
      wget https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22kto1k.pth
      ```

   2. 导出onnx文件。

      1. 使用beit_pth2onnx.py出onnx文件。

         运行beit_pth2onnx.py脚本。

         ```
         python3 beit_pth2onnx.py --source="beit_base_patch16_224_pt22k_ft22kto1k.pth" --target="beit_base_patch16_224.onnx" --batch_size=8
         ```

         获得beit_base_patch16_224.onnx文件。

      2. 使用onnxsim优化ONNX文件。

         ```
         python3 -m onnxsim --input-shape="8,3,224,224" beit_base_patch16_224.onnx beit_onnxsim_bs8.onnx
         ```

         获得beit_onnxsim_bs8.onnx文件。

         其中，input-shape需要根据所使用的batch size进行修改 第二个参数为onnx模型，第三个参数为输出模型名称

      3. 使用magiconnx优化onnx文件。

         - 获取magiconnx

           获取工具及使用方法可以参考 `https://gitee.com/Ronnie_zheng/MagicONNX/tree/master#1-magiconnx%E7%AE%80%E4%BB%8B`

           ```
           git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
           cd MagicONNX
           pip install .
           ```

         - 运行graph_modify.py对onnx模型结构图进行优化

           ```
           python3 graph_modify.py  beit_onnxsim_bs8.onnx  beit_mg_bs8.onnx  8
           ```

           其中三个参数分别为：输入onnx模型，输出onnx模型，batch size

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
   
         ```

         > **说明：** 
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

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
         atc --framework=5 --model=beit_mg_bs8.onnx \
         --output=beit_mg_bs8 \
         --input_format=NCHW \
         --input_shape="image:8,3,224,224" \
         --log=error \
         --soc_version={chip_name} \
         --optypelist_for_implmode="Gelu" \
         --op_select_implmode=high_performance
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --optypelist_for_implmode: 选择算子 
           -   --op_select_implmode: 更改上述算子的运行模式

           运行成功后生成<u>***beit_mg_bs8.om***</u>模型文件。



二. 开始推理验证。

1.  使用ais-infer工具进行推理。

    执行命令增加工具可执行权限，并根据OS架构选择工具
    ```
    chmod u+x 
    ```


2. 执行推理。 

    1. 测试性能数据
    ```
    python3 ais_infer.py --model "beit_mg_bs8.om"  --loop 100 --batchsize 8
    ```
        
   - 参数说明：
            
     - model：om模型路径。
     - loop：循环次数
     - batchsize：om模型batch size大小

    2. 测试精度数据
        ```
        python3 ais_infer.py --model "beit_mg_bs8.om" \
        --input "./prep_image_bs8" \
        --output ./ais_out/ \
        --outfmt TXT  
        --batchsize 8
       ```
   
       - 参数说明：
            
         - model：om模型路径。
         - input：数据集路径。
         - outfmt：输出数据格式。
         - output：输出路径
         - batchsize: om模型batch size大小
       > **说明：** 
        > 执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见


3. 精度验证。

    调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。
    
    ```
    python3 beit_eval_acc.py --folder_davinci_target="./ais_out/{具体输出文件夹}" \
        --annotation_file_path="/opt/npu/imageNet/val_label.txt" \
        --result_json_path="./" \
        --json_file_name="acc_bs8.json" \
        --batchsize=8
    ```
    
   - 参数说明：
       - folder_davinci_target：为生成推理结果所在路径  
    
       - annotation_file_path：标签数据
    
       - json_file_name：为生成结果文件

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

1. 精度
   
    | om model | Acc@1  |
    | :------: | :----: |
    |  office  | 85.27% |
    |   BS1    | 84.68% |
    |   BS4    | 84.68% |
    |   BS8    | 84.68% |
    |   BS16   | 84.68% |
    |   BS32   | 84.67% |
    
    84.67% / 85.27% = 99.30%
    精度误差保持在1%以内，精度达标


2. 性能

    | gpu | batch size |   fps   |
    |:----------:|:-------:| :------: |
    |  T4  |    bs1     | 188.917 |
    |  T4  |    bs4     | 267.900 |
    |  T4  |    bs8     | 290.474 |
    |  T4  |    bs16    | 287.486 |
    |  T4  |    bs32    |  285.54  |
    
    | npu  | batch size | fps/card |
    |:----------:|:--------:| :------: |
    | 310P |    bs1     | 295.691  |
    | 310P |    bs4     |  349.66  |
    | 310P |    bs8     |  516.00  |
    | 310P |    bs16    | 390.817  |
    | 310P |    bs32    |  361.19  |
    
    npu在bs=8时性能最佳，gpu在bs=8时性能最佳，两者对比：

    Ascend 310P/ gpu t4 = 512.137 / 290.474 = 1.763
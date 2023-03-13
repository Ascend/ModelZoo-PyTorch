# EfficientDetD7模型-推理指导


- [概述](#概述)

  - [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)

  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型推理](#准备数据集)
- [模型推理性能&精度](#模型推理性能&精度)

# 概述

EfficientDet该论文首先提出了一种加权双向特征金字塔网络（BiFPN），它允许简单、快速的多尺度特征融合；其次，提出了一种复合特征金字塔网络缩放方法，统一缩放所有backbone的分辨率、深度和宽度、特征网络和box/class预测网络。

当融合不同分辨率的特征时，一种常见的方法是首先将它们调整到相同的分辨率，然后将它们进行总结。金字塔注意网络global self-attention上采样恢复像素定位。所有以前的方法都一视同仁地对待所有输入特征。 然而，论文中认为由于不同的输入特征在不同的分辨率，他们通常贡献的输出特征不平等。为了解决这个问题，论文建议为每个输入增加一个权重，并让网络学习每个输入特性的重要性。


- 参考实现：

  ```
  ur=https://github.com/rwightman/efficientdet-pytorch
  branch=master
  commit_id=c5b694aa34900fdee6653210d856ca8320bf7d4e
  model_name=EfficientDetD7
  ```
  



## 输入输出数据

- 输入数据

  | 输入数据 | 数据类型                   | 大小     | 数据排布格式 |
  | -------- | -------------------------- | -------- | ------------ |
  | input    | batchsize x 3 x 1536x 1536 | RGB_FP32 | NCHW         |


- 输出数据

  | 输出数据  | 数据类型                    | 大小    | 数据排布格式 |
  | --------- | --------------------------- | ------- | ------------ |
  | classout1 | batchsize x 810 x 192 x 192 | FLOAT32 | NCHW         |
  | boxout1   | batchsize x 36 x 192x 192   | FLOAT32 | NCHW         |
  | classout2 | batchsize x 810 x 96 x 96   | FLOAT32 | NCHW         |
  | boxout2   | batchsize x 36 x 96 x 96    | FLOAT32 | NCHW         |
  | classout3 | batchsize x 810 x 48 x 48   | FLOAT32 | NCHW         |
  | boxout3   | batchsize x 36 x 48 x 48    | FLOAT32 | NCHW         |
  | classout4 | batchsize x 810 x 24 x 24   | FLOAT32 | NCHW         |
  | boxout4   | batchsize x 36 x 24 x 24    | FLOAT32 | NCHW         |
  | classout5 | batchsize x 810 x 12 x 12   | FLOAT32 | NCHW         |
  | boxout5   | batchsize x 36 x 12 x 12    | FLOAT32 | NCHW         |


# 推理环境准备

- 该模型需要以下插件与驱动 

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手

## 获取源码

1. 获取源码。

   ```
   git clone https://github.com/rwightman/efficientdet-pytorch   # 克隆仓库的代码
   cd efficientdet-pytorch             						  # 切换到模型的代码仓目录
   git checkout c5b694aa34900fdee6653210d856ca8320bf7d4e         # 切换到对应分支
   patch -p1 < ../EfficientDetD7.patch							  # 添加模型补丁
   cd ..
   git clone https://gitee.com/zheng-wengang1/onnx_tools.git	  # 下载onnx模型修改工具
   cd onnx_tools
   git checkout cbb099e5f2cef3d76c7630bffe0ee8250b03d921		  # 切换到对应分支
   cd ..
   sed -i -e 's/onnx.onnx_ml_pb2/onnx/g' onnx_tools/OXInterface/OXInterface.py  # 对onnx模型修改工具进行修改
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集

1. 获取原始数据集。

   本模型支持coco2017 val 5000张图片的验证集。请用户自行获取数据集，上传数据集到代码仓目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到coco val2017.zip验证集及~/annotations中的instances_val2017.json数据标签。
   
   数据目录结构请参考：

    ```
   coco_data
       ├──val2017
       ├── annotations
       ├── instances_val2017.json
    ```
   
   请将coco_data文件夹放在EfficientDetD7目录下。
   
2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行“EfficientDetD7_preprocess.py”脚本，完成预处理。

   ```
   mkdir bin_save
   python3 EfficientDetD7_preprocess.py --root=coco_data --bin-save=bin_save
   ```

   参数说明：

   - root：coco数据集文件

    - bin-save：输出的二进制文件（.bin）所在路径

   每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成bin_save文件夹。



## 模型推理

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      [d7.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/EffcientDet-D7/PTH/d7.pth)

       请在EfficientDetD7目录下创建model文件夹，并将d7.pth文件移入model文件夹中。

   2. 导出onnx文件。

      1. 使用EfficientDetD7_pth2onnx.py导出onnx文件。

         运行EfficientDetD7_pth2onnx.py脚本。
         ```
         python3 EfficientDetD7_pth2onnx.py --batch_size=1 --checkpoint=./model/d7.pth --out=./model/d7_bs1.onnx 
         ```

         参数说明：
         
         -   --batch_size：转出模型的batchsize，目前只支持1
         -   --checkpoint：待转模型的参数文件
         -   --out：输出的onnx模型文件名。
         
         获得d7_bs1.onnx 文件。
         
         
         
      2. 优化并修改ONNX文件。

         ```
         python3 -m onnxsim --input-shape="1,3,1536,1536" --dynamic-input-shape ./model/d7_bs1.onnx ./model/d7_bs1_sim.onnx --skip-shape-inference
         python3 modify_onnx.py --model=./model/d7_bs1_sim.onnx --out=./model/d7_bs1_modify.onnx
         ```

         获得d7_bs1_modify.onnx文件。

      

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3
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
         atc --framework=5 --model=./model/d7_bs1_modify.onnx --output=./model/d7_bs1 --input_format=NCHW --input_shape="x.1:1,3,1536,1536" --log=debug --soc_version=Ascend310P3
         ```

         参数说明：

         -   --model：为ONNX模型文件。
         -   --framework：5代表ONNX模型。
         -   --output：输出的OM模型。
         -   --input\_format：输入数据的格式。
         -   --input\_shape：输入数据的shape。
         -   --log：日志级别。
         -   --soc\_version：处理器型号。

         运行成功后生成<u>***d7_bs1.om***</u>模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        在执行推理前，请创建result文件夹
        
        ```
        mkdir result
        ```
        
        请用下列语句执行推理流程：
        
        ```
        python3 -m ais_bench --model model/d7_bs1.om --input ./bin_save --output ./ --output_dirname=result --outfmt BIN --batchsize=1 --infer_queue_count=1
        ```
        
         参数说明：
        
        - model：模型类型。
        - input：经过预处理后的bin文件路径。
        - output：输出文件路径。
        - output_dirname：输出文件目录
        - outfmt：输出文件格式。
        - batchsize：批次大小。
        - infer_queue_count: 推理队列的数据最大数量。
        
        推理后的输出默认在当前目录result下。
     

3. 精度验证。

     调用“EfficientDetD7_postprocess.py”脚本即可获得最终mAP精度：

     ```
     python3 EfficientDetD7_postprocess.py --root=./coco_data --omfile=./result
     ```

     参数说明:

     - root：coco数据集路径。
     - omfile：模型推理结果。

4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model=model/d7_bs1.om --loop=20 --batchsize=1
      ```

      参数说明：
      - model：模型路径。
      - batchsize：性能测试时所用的batch_size，本模型仅支持1。




# 模型推理性能&精度

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| 310P3 | 1 | coco_data | mAP:53.0 | 6.18fps |
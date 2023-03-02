# DB_Dynamic 模型PyTorch离线推理指导


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

在基于分割的文本检测网络中，最终的二值化map都是使用的固定阈值来获取，并且阈值不同对性能影响较大。而在DB中会对每一个像素点进行自适应二值化，二值化阈值由网络学习得到，彻底将二值化这一步骤加入到网络里一起训练，这样最终的输出图对于阈值就会非常鲁棒。 


- 参考实现：

  ```
  url=https://github.com/MhLiao/DB 
  commit_id=e5a12f5c
  model_name=DB_Dynamic_for_PyTorch
  ```
  

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                       | 数据排布格式 |
  | -------- | -------- | -------------------------- | ------------ |
  | input    | RGB_FP32 | 1 x 3 x h x w | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小                       | 数据排布格式 |
  | -------- | -------- | -------------------------- | ------------ |
  | output1  | FLOAT32  | 1 x 1 x h x w | NCHW           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.2.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码

   ```shell
   git clone https://github.com/MhLiao/DB 
   cd DB
   git reset 4ac194d0357fd102ac871e37986cb8027ecf094e --hard
   patch -p1 < ../DB.patch
   cd ..
   ```

2. 安装依赖

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集

   本模型支持icdar2015验证集。用户需自行获取数据集解压并上传数据集到DB/datasets路径下。目录结构如下：

   ```
   datasets/icdar2015/  
   ├── test_gts  
   ├── test_images  
   ├── test_list.txt  
   ├── train_gts  
   └── train_list.txt  
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   在`DB`目录下执行db_preprocess.py脚本，完成预处理

   ```shell
   python3 ../db_preprocess.py --image_src_path=./datasets/icdar2015/test_images --npy_file_path=./input_npy
   ```
   
   结果存在 ./input_npy 中


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       ```sh
       wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/DBnet/PTH/ic15_resnet50 -O DB/ic15_resnet50
       ```

   2. 导出onnx文件。


      1. 在DB目录中使用convert_to_onnx.py导出onnx文件

         ```shell
         python3 ../convert_to_onnx.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml resume ./ic15_resnet50 dbnet.onnx
         ```
         
         获得`dbnet.onnx`文件 
      
      2. 修改onnx文件

         ```
         python3 ../modify.py dbnet.onnx dbnet_fix.onnx

         mv dbnet_fix.onnx ..
         ```

         得到`dbnet_fix.onnx`, 并移到`DB`外
      
   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。
   
         ```sh
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```
   
      2. 执行命令查看芯片名称（$\{chip\_name\}）。
   
         ```sh
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
   
         ```sh
         atc --framework=5 --model=./dbnet_fix.onnx --input_format=ND --input_shape_range="input:[1,3,640~4096,640~4096]" --output=om/dbnet_dy --log=error --soc_version=Ascend${chip_name}
         ```
      
         运行成功后生成<u>***om/dbnet_dy.om***</u>模型文件。
         
         - 参数说明
           
              - --model：为ONNX模型文件
              
              - --framework：5代表ONNX模型
              
              - --output：输出的OM模型
              
              - --input_format：输入数据的格式
              
              - --input_shape_range：输入数据的shape范围
              
              - --log：日志级别

              - --soc_version：处理器型号
              
                
   
2. 开始推理验证

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。
   

        ```shell
        python3 -m npu_end2end.py --data_path DB/input_npy --onnx_path ./dbnet_fix.onnx  --device 0 --om_path om/dbnet_fix.om
        ```

        -   参数说明：

             -   --data_path: 输入数据
             -   --onnx_path: onnx文件
             -   --device： npu device id
             -   --om_path: om文件

         npu E2E性能和onnxruntime运行结果的余弦相似度会打屏显示


   3. GPU 推理

      运行以下脚本确保有GPU环境，以及对应的cuda和TensorRT

      ```shell
      python3 gpu_end2end.py --data_path DB/input_npy --onnx_path dbnet_fix.onnx --engine_path ./dbnet.trt
      ```

      - 参数说明：

        - --data_path: 输入数据
        - --onnx_path：onnx文件
        - --engine_path：trt文件
      
      gpu E2E性能会打屏显示


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 |  数据集   | 精度 | E2E(s)  |
| :------: | :-------: | :--: | :---: |
|  310P3   | icdar2015 | cosineSimilarity:0.99 | 9.3s左右 |

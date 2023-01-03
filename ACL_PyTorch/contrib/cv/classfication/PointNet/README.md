# PointNet模型-推理指导


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

PointNet是针对3D点云进行分类和分割的模型。该网络包含了三维的STN模块，三维的STN可以通过学习点云本身的位姿信息学习到一个最有利于网络进行分类或分割的DxD旋转矩阵（D代表特征维度，pointnet中D采用3和64）。pointnet采用了两次STN，第一次input transform是对空间中点云进行调整，直观上理解是旋转出一个更有利于分类或分割的角度，比如把物体转到正面；第二次feature transform是对提取出的64维特征进行对齐，即在特征层面对点云进行变换。 


- 参考实现：

  ```
  url=https://github.com/fxia22/pointnet.pytorch 
  commit_id=f0c2430b0b1529e3f76fb5d6cd6ca14be763d975
  code_path=/ACL_PyTorch/contrib/cv/classfication/PointNet
  model_name=PointNet
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                 | 数据排布格式 |
  | -------- | -------- | -------------------- | ------------ |
  | input    | FP32     | batchsize x 3 x 2500 | BCN          |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | output1  | FLOAT32  | batchsize x 16 | BD         |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.6.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/fxia22/pointnet.pytorch        
   cd pointnet.pytorch              # 切换到模型的代码仓目录 
   git checkout f0c2430b0b1529e3f76fb5d6cd6ca14be763d975
   patch -p1 < ../modify.patch    # 打上补丁
   cd ..  
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 数据预处理。 （解压命令参考tar –xvf *.tar与 unzip *.zip） 

   本模型是对点云数据进行分类，使用的数据集是shapenet的子集，用户需要自行获取数据集，数据集包含了16个类别文件夹和用于对应类别和训练集测试集划分的功能文件，下载完成后请将数据存放到模型根目录下自行创建的data文件夹中。 数据集目录如下：

   ```
   shapenet
   ├──02691156
   │   ├── points
   │   ├── points_label
   │   ├── seg_img
   ├──02773838
   │   ├── points
   │   ├── points_label
   │   ├── seg_img
   ......#省略8个文件夹
   │   ├── README.txt
   │   ├── README.txt~
   │   ├── synsetoffset2category.txt
   │   ├── train_test_split
   ```
   
2.  数据预处理，将原始数据集转换为模型输入的数据。 

    执行预处理脚本“ pointnet_preprocess.py”。 

      ```
      python3 pointnet_preprocess.py ./data/shapenetcore_partanno_segmentation_benchmark_v0 ./bin_file
      ```

   + 参数说明：
     + ./data/：数据集所在路径；
     + bin_file：是预处理后的数据文件的相对路径；

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1.  获取权重文件。 

      从代码仓中获取权重文件“[checkpoint_79_epoch.pkl](https://pan.baidu.com/s/168Vk3C60iZOWrgGIBNAkjw)” ，提取码：lmwa，下载好后请放到PointNet工程目录下。

   2.  导出onnx文件。

       1. 使用pointnet_pth2onnx.py导出onnx文件。
   
            ```
            python3 pointnet_pth2onnx.py
            ```
      
            获得pointnet.onnx文件。
   
       2. 简化ONNX文件。 
   
            ```
            python3 -m onnxsim pointnet.onnx pointnet_bs1_sim.onnx --input-shape="1, 3, 2500" --dynamic-input-shape
            ```
      
            获得pointnet_bs1_sim.onnx文件。
   
       3. 优化ONNX文件
   
            ```
            git clone https://gitee.com/zheng-wengang1/onnx_tools.git
            cd onnx_tools && git checkout cbb099e5f2cef3d76c7630bffe0ee8250b03d921
            cd ..
            python3 fix_conv1d.py pointnet_bs1_sim.onnx pointnet_bs1_sim_fixed.onnx
            ```
   
            获得pointnet_bs1_sim_fixed.onnx文件。
   
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
            atc --framework=5 --model=pointnet_bs1_sim_fixed.onnx --output=pointnet_bs1_fixed --input_shape="image:1, 3, 2500" --soc_version=Ascend${chip_name} --log=error --out_nodes "LogSoftmax_84:0"
            ```
   
            + 参数说明：
              + --model：为ONNX模型文件；
              + --framework：5代表ONNX模型；
              + --output：输出的OM模型；
              + --input\_format：输入数据的格式；
              + --input\_shape：输入数据的shape；
              + --log：日志级别；
              + --out_nodes：固定输出节点；
              + --soc\_version：处理器型号；
   
            运行成功后在--output指定地址生成pointnet_bs1_fixed.om模型文件。
   
2. 开始推理验证。

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]


   2. 执行推理。

        ```
        python3 ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model ./pointnet_bs1_fixed.om --input ./bin_file --output ./result/bs1 --outfmt TXT --batchsize 1 
        ```

        -   参数说明：

             -   --model：om文件路径；
             -   --input：预处理完的数据集文件夹；
             -   --output：推理结果保存地址；
             -   --outfmt： 输出数据的格式，默认”BIN“，可取值“NPY”、“BIN”、“TXT” ；
             -   --batchsize： 模型batch size ;

       

         >**说明：** 
         >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]。

   3. 精度验证。

       调用“pointnet_postprocess.py”脚本与数据集标签“name2label.txt”比对，可以获得Accuracy数据。 

      ```
       python3 pointnet_postprocess.py ./name2label.txt ./result/bs1/xxxx
      ```
      
      - 参数说明：
        - ./name2label.txt：为标签数据路径；
        - ./result/bs1/xxxx： 推理结果所在路径 ;
     
      



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

精度：
| Batch Size |   310      |   310P     |  github开源仓结果（官方)|  pth模型推理结果|
| ---------- | ---------- | ---------- | --------------------|-----------|
|   1        |  0.974243  | 0.973895   |         0.981          |       0.9742|           



性能：
| 芯片型号| Batch Size |   310      |   310P     |      310P/310   | 
| --------| ---------- | ---------- | ----------  | ---------------| 
| 310P3   |   1        | 1199.43986 | 2132.781818 | 1.7781482      | 
| 310P3 |   4        | 1343.31039 | 1937.487074   | 1.4423227      | 
| 310P3 |   8        | 1483.77947 | 2160.862723     | 1.45632337     | 
| 310P3 |   16       | 1426.54631 | 2251.604186 | 1.57836039     |
| 310P3 |   32       | 1456.63772 | 2211.745445 | 1.51839089     | 
| 310P3 |   64       | 1484.46742 | 2220.06382 |  1.49552883    |  
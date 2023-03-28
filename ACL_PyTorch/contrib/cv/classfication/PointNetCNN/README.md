# PointNetCNN模型-推理指导

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

PointNetCNN是一个简单而通用的从点云中学习特征的框架。在图像处理中卷积可以很好的处理数据中局部空间相关性。然而点云是不规则和无序的，点云里面的点的输入顺序，是阻碍Convolution操作的主要问题，PointNetCNN提出X-transformation，对输入点云进行变换矩阵处理，得到一个与顺序无关的特征，实现了点云数据上的卷积操作。


- 参考实现：

  ```
  url=https://github.com/hxdengBerkeley/PointCNN.Pytorch
  branch=master
  commit_id=6ec6c291cf97923a84fb6ed8c82e98bf01e7e96d
  ``` 


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  |  input1  |   FP32   |   1 x 1024 x 3   |   ND       |
  |  input2  |   FP32   |   1 x 1024 x 3   |   ND       |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | 1 x 40   | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |




# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取开源代码仓。

   ```
   git clone https://github.com/hxdengBerkeley/PointCNN.Pytorch -b master   
   cd PointCNN.Pytorch  
   git reset 6ec6c291cf97923a84fb6ed8c82e98bf01e7e96d --hard 
   patch -p1 < ../PointNetCNN.patch
   cd ..
   cp -r PointCNN.Pytorch/utils PointCNN.Pytorch/provider.py ./
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
   本模型使用经过处理的modelnet40_ply_hdf5_2048数据集，请用户自行获取并将数据集放在./data/下
   
   ```
   mkdir data
   ```
   
   目录结构如下：

   ```
   PointNetCNN
   ├── data
      └── modelnet40_ply_hdf5_2048
         ├── ply_data_test0.h5
         ├── ...
   ```

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   执行PointNetCNN_preprocess.py脚本，完成预处理。

   ```
   python3 PointNetCNN_preprocess.py ./prep_dataset ./labels
   ```
   "./prep_dataset"：输出的二进制文件（.bin）所在路径。

   "./labels"：标签文件目录。

   每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成"prep_dataset"二进制文件夹。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
   
      [PointNetCNN权重文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/PointNetCnn/PTH/pointcnn_epoch240.pth)

   2. 导出onnx文件。

      1. 使用PointNetCNN_pth2onnx.py导出onnx文件。

         运行PointNetCNN_pth2onnx.py脚本。

         ```
         python3 PointNetCNN_pth2onnx.py  pointcnn_epoch240.pth pointnetcnn.onnx 
         ```

         获得pointnetcnn.onnx文件。

      2. 优化onnx模型

         请访问[aoto-optimizer优化工具](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)代码仓，根据readme文档进行工具安装。

         ```
         python3 PointNetCNN_modify_onnx.py pointnetcnn.onnx pointnetcnn_new.onnx
         ```
         - 参数说明：
            - pointnetcnn.onnx：输入模型路径。
            - pointnetcnn_new.onnx：修改后的模型路径。

         得到pointnetcnn_new.onnx文件。

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
         ```shell
         atc --framework=5 --model=./pointnetcnn_new.onnx --input_format=ND --output=pointnetcnn_bs1 --log=error --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成pointnetcnn_bs1.om模型文件。



2. 开始推理验证。

   a.  安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。


   b.  执行推理。

      ```
      python3 -m ais_bench --model ./pointnetcnn_bs1.om --input "prep_dataset,prep_dataset" --output ./ --output_dirnam result --outfmt "TXT"
      ```

      -   参数说明：

           -   model：输入的om文件。
           -   batchsize：批大小，即1次迭代所使用的样本量。
           -   input：输入的bin数据文件。
           -   output：结果保存路径。
           -   output_dirnam：结果保存子目录。
           -   outfmt：输出数据格式。

      输出结果保存在当前目录result文件夹下。


   c.  精度验证。

      调用PointNetCNN_postprocess脚本与数据集标签labels/label*.npy比对，可以获得Accuracy数据，结果保存在result_bs1.json中。

      ```
      python3 PointNetCNN_postprocess.py ./labels/label ./result
      ```

      ./labels/label：标签文件路径 
    
      ./result：ais_bench推理结果
    


   d.  性能验证。

      可使用ais_bench推理工具的纯推理模式验证om模型的性能，参考命令如下：

      ```shell
      python3 -m ais_bench --model=${om_model_path} --loop=20
      ```

      - 参数说明：
        - --model：om模型文件路径。
        - --loop：推理次数。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>


| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | -------------- | ---------- | ---------- | --------------- |
|   310P3   |     1          | modelnet40 |   83.06%   |    273.37       |

仅支持batch size为1
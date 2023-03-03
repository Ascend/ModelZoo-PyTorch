# PointNetCNN模型-推理指导


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

PointNetCNN是一个简单而通用的从点云中学习特征的框架。在图像处理中卷积可以很好的处理数据中局部空间相关性。然而点云是不规则和无序的，点云里面的点的输入顺序，是阻碍Convolution操作的主要问题，PointNetCNN提出X-transformation，对输入点云进行变换矩阵处理，得到一个与顺序无关的特征，实现了点云数据上的卷积操作。


- 参考实现：

  ```
  url=https://github.com/hxdengBerkeley/PointCNN.Pytorch
  branch=master
  commit_id=6ec6c291cf97923a84fb6ed8c82e98bf01e7e96d
  ``` 

  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  |  input1  |   FP32   |   batchsize x 1024 x 3   |   ND       |
  |  input2  |   FP32   |   batchsize x 1024 x 3   |   ND       |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 40 x 1 | FLOAT32  | ND           |




# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                   |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                    | 22.0.2 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| PyTorch                                                      | 1.8.0   | -                                                            |

  **表 2**  环境依赖表

| 依赖                                                         | 版本    | 
| ------------------------------------------------------------ | ------- | 
| Python                                                       | 3.7.5   | 
| PyTorch                                                      | 1.8.0   |
| TorchVision                                                       | 0.9.0   | 
| onnx                                                      | 1.10.2   |
| onnxruntime                                                       | 1.21.1   | 
| onnx-simplifier                                                      | 0.3.6   |
| numpy                                                       | 1.21.5   | 
| h5py                                                      | 3.7.0  |
| scipy                                                       | 1.2.0   | 
| sklearn                                                      | 0.0   |
| Pillow                                                       | 9.2.0   | 
| six                                                      | 1.16.0   |
| decorator                                                       | 5.1.1   | 
| tqdm                                                      | 4.64.1   |



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
   cd data
   wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/PointNetCNN/modelnet40_ply_hdf5_2048.zip
   unzip -d  modelnet40_ply_hdf5_2048 modelnet40_ply_hdf5_2048.zip
   cd ..
   ```

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   执行PointNetCNN_preprocess.py脚本，完成预处理。

   ```
   python3.7 PointNetCNN_preprocess.py ./prep_dataset ./labels
   ```
   "./prep_dataset"：输出的二进制文件（.bin）所在路径。
   "./labels"：标签文件目录。
   每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成"prep_dataset"二进制文件夹。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
   
      从源码包中获取权重文件："[pointcnn_epoch240.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/PointNetCnn/PTH/pointcnn_epoch240.pth)"

   2. 导出onnx文件。

      1. 使用PointNetCNN_pth2onnx.py导出onnx文件。

         运行PointNetCNN_pth2onnx.py脚本。

         ```
         python3.7 PointNetCNN_pth2onnx.py  pointcnn_epoch240.pth pointnetcnn.onnx 
         ```

         获得pointnetcnn.onnx文件。
      2. 删除onnx的split和concat算子

         ```
         git clone https://gitee.com/Ronnie_zheng/MagicONNX.git -b dev
         cd MagicONNX
         pip3 install .
         cd ..
         python3.7 PointNetCNN_DelConcatSplit.py
         ```

      3. 优化ONNX文件。

         ```
         python3.7 -m onnxsim pointnetcnn_modify.onnx pointnetcnn_sim.onnx  --input-shape P_sampled:1,1024,3 P_patched:1,1024,3
         ```

         获得pointnetcnn_sim.onnx文件。

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

      3. 执行ATC命令。
         ```
         atc --framework=5 --model=./pointnetcnn_sim.onnx --input_format=ND --input_shape="P_sampled:1,1024,3;P_patched:1,1024,3" --output=pointnetcnn_bs1 --log=error --soc_version=${chip_name}
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
      mkdir ais_result
      python3.7 -m ais_bench --model ./pointnetcnn_bs1.om --batchsize 1 --input "prep_dataset,prep_dataset" --output ./ais_result --outfmt "TXT"
      ```

      -   参数说明：

           -   model：输入的om文件。
           -   batchsize：批大小，即1次迭代所使用的样本量。
           -   input：输入的bin数据文件。
           -   output：结果保存路径。
           -   outfmt：输出数据格式。
		...

      输出结果保存在当前目录ais_result/X(X为执行推理的时间)文件夹下。


   c.  精度验证。

      调用脚本与数据集标签labels/label*.npy比对，可以获得Accuracy数据，结果保存在result_bs1.json中。

      ```
      python3.7 PointNetCNN_postprocess.py ./labels/label ./ais_result > result_bs1.json
      ```

      ./labels/label：标签文件路径 
    
      ./ais_result：ais_bench推理结果
    
      result_bs1.json：为生成结果文件


   d.  性能验证。

      可使用ais_bench推理工具的纯推理模式验证om模型的性能，参考命令如下：

      ```
       python3.7 -m ais_bench --model=./pointnetcnn_bs1.om --loop=20 --batchsize=1
      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>


| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|     310*4     |     1             |     modelnet40       |     82.37%       |       32.5866          |
|     310P     |     1             |     modelnet40       |     82.82%       |       112.383          |
|     T4     |     1             |     modelnet40       |     82.37%       |       137.044          |
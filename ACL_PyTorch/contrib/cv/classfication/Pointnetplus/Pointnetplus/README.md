# PointNet++模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

PointNet++是PointNet的续作，在一定程度上弥补了PointNet的一些缺陷，表征网络基本和PN类似，还是MLP、1*1卷积、pooling那一套，核心创新点在于设计了局部邻域的采样表征方法和这种多层次的encoder-decoder结合的网络结构。
- 参考实现：

  ```
  url=https://github.com/yanx27/Pointnet_Pointnet2_pytorch
  commit_id=e365b9f7b9c3d7d6444278d92e298e3f078794e1
  code_path=https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Pointnetplus/Pointnetplus
  model_name=PointNet++
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据
  
- part1

  | 输入数据 | 数据类型 | 大小                       | 数据排布格式 |
  | -------- |--------------------------| -------- | ------------ |
  | xyz  | FP32 | batchsize x 512 x 3      | ND     |
  | samp_points    | FP32 | batchsize x 3 x 32 x 512 | ND     |

- part2

  | 输入数据 | 数据类型 | 大小                         | 数据排布格式 |
  | -------- |----------------------------| -------- | ------------ |
  | l1_xyz  | FP32 | batchsize x 3 x 128        | ND     |
  | l1_point    | FP32 | batchsize x 131 x 64 x 128 | ND     |

- 输出数据

- part1

  | 输出数据 | 数据类型 | 大小               | 数据排布格式 |
  | -------- |--------------------------| -------- | ------------ |
  | l1_xyz    | FLOAT32  | batchsize x 512 x 3 | ND           |
  | l1_point    | FLOAT32  | batchsize x 128 x 512 | ND           |

- part2

  | 输入数据     | 数据类型 | 大小                    | 数据排布格式 |
  | -------- |-----------------------| -------- | ------------ |
  | class        | FP32 | batchsize x 40        | ND     |
  | l3_point | FP32 | batchsize x 1024 x -1 | ND     |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                         | 版本      | 环境准备指导                                                 |
  |---------| ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.9.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
    git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch models
    cd models
    git checkout e365b9f7b9c3d7d6444278d92e298e3f078794e1
    patch -p1 < ../models.patch
    cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）


   该模型使用[官网提供的modelnet40](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)，保存在data/modelnet40_normal_resampled/。

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行pointnetplus_preprocess.py脚本，完成预处理。需要注意的是执行part2的预处理需要先完成part1的数据推理。

   ```
   python3.7 pointnetplus_preprocess.py --preprocess_part 1 --save_path ./modelnet40_processed/bs1/pointset_chg_part1 --save_path2 ./modelnet40_processed/bs1/xyz_chg_part1 --data_loc data/modelnet40_normal_resampled/
   python3.7 pointnetplus_preprocess.py --preprocess_part 2 --save_path ./modelnet40_processed/bs1/pointset_chg_part2 --save_path2 ./modelnet40_processed/bs1/xyz_chg_part2 --data_loc ./output/subdir1 --data_loc2 ./modelnet40_processed/bs1/xyz_chg_part1

   ```
   
   - 参数说明：
   
     --preprocess_part，选择进行那一部分模型的预处理。
         
     --save_path，输出的二进制文件（.bin）所在路径1。
   
     --save_path2，输出的二进制文件（.bin）所在路径2。

     --data_loc，需要处理的数据集所在路径。



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      pth采用开源仓自带的权重，权重位置：
      ```
      ./models/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth
      ```
      
   2. 导出onnx文件。

      1. 使用pointnetplus_pth2onnx.py脚本。

         运行pointnetplus_pth2onnx.py脚本。

         ```
         python3.7 pointnetplus_pth2onnx.py --target_model 1 --pth_dir './models/log/classification/pointnet2_ssg_wo_normals/checkpoints' 
         python3.7 pointnetplus_pth2onnx.py --target_model 2 --pth_dir './models/log/classification/pointnet2_ssg_wo_normals/checkpoints'
         ```

         获得Pointnetplus_part1.onnx和Pointnetplus_part2.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/......
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
         atc --framework=5 --model=Pointnetplus_part1.onnx --output=Pointnetplus_part1_bs{batch size} --input_format=ND --input_shape="samp_points:{batch size},3,32,512;xyz:{batch size},512,3" --log=debug --soc_version=Ascend310P3
         atc --framework=5 --model=Pointnetplus_part2.onnx --output=Pointnetplus_part2_bs{batch size} --input_format=ND --input_shape="l1_points:{batch size},131,64,128;l1_xyz:{batch size},3,128" --log=debug --soc_version=Ascend310P3
         示例
         atc --framework=5 --model=Pointnetplus_part1.onnx --output=Pointnetplus_part1_bs1 --input_format=ND --input_shape="samp_points:1,3,32,512;xyz:1,512,3" --log=debug --soc_version=Ascend310P3
         atc --framework=5 --model=Pointnetplus_part2.onnx --output=Pointnetplus_part2_bs1 --input_format=ND --input_shape="l1_points:1,131,64,128;l1_xyz:1,3,128" --log=debug --soc_version=Ascend310P3
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成Pointnetplus_part1_bs1.om和Pointnetplus_part2_bs1.om模型文件，batch size为4、8、16、32、64的修改对应的batch size的位置即可。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        python3 -m ais_bench --model ./Pointnetplus_part1_bs{batch size}.om --input './modelnet40_processed/bs1/xyz_chg_part1,./modelnet40_processed/bs1/pointset_chg_part1' --output ./output --output_dirname subdir1 --outfmt 'BIN' --batchsize {batch size}
        python3 -m ais_bench --model ./Pointnetplus_part2_bs{batch size}.om --input './modelnet40_processed/bs1/xyz_chg_part2,./modelnet40_processed/bs1/pointset_chg_part2' --output ./output --output_dirname subdir2 --outfmt 'BIN' --batchsize {batch size}
        示例
        python3 -m ais_bench --model ./Pointnetplus_part1_bs1.om --input './modelnet40_processed/bs1/xyz_chg_part1,./modelnet40_processed/bs1/pointset_chg_part1' --output ./output --output_dirname subdir1 --outfmt 'BIN' --batchsize 1
        python3 -m ais_bench --model ./Pointnetplus_part2_bs1.om --input './modelnet40_processed/bs1/xyz_chg_part2,./modelnet40_processed/bs1/pointset_chg_part2' --output ./output --output_dirname subdir2 --outfmt 'BIN' --batchsize 1
        ```

        -   参数说明：

             -   model：需要推理om模型的路径。
             -   input：模型需要的输入bin文件夹路径。
             -   output：推理结果输出路径。
             -   outfmt：输出数据的格式。
             -   output_dirname:推理结果输出子文件夹。

        推理后的输出默认在当前目录output的subdir1和subdir2下。需要注意的是需要先执行完part1的推理，进行part2的数据预处理，再进行part2的推理。

   3. 精度验证。

      调用pointnetplus_postprocess.py脚本推理结果，结果保存在result.json中。

      ```
      python3.7 pointnetplus_postprocess.py --target_path ./output/subdir2 --data_loc data/modelnet40_normal_resampled/ >result.json
      ```

      - 参数说明：

        - --target_path：为生成推理结果所在路径  

        - --data_loc：为标签数据所在路径

   4. 性能验证。

     可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

       ```
       python3.7 -m ais_bench --model=./Pointnetplus_part1_bs{batch size}.om --loop=1000 --batchsize={batch size} 
       python3.7 -m ais_bench --model=./Pointnetplus_part2_bs{batch size}.om --loop=1000 --batchsize={batch size}
       示例
       python3.7 -m ais_bench --model=./Pointnetplus_part1_bs1.om --loop=1000 --batchsize=1 
       python3.7 -m ais_bench --model=./Pointnetplus_part2_bs1.om --loop=1000 --batchsize=1
       ```

     - 参数说明：
       - --model：需要验证om模型所在路径
       - --batchsize：验证模型的batch size，按实际进行修改



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

part1(无精度)

| 芯片型号 | Batch Size | 数据集 | 精度  | 性能   |
| --------- |------------| ---------- |-----|------|
|   310P3        | 1          |  modelnet40          | -   | 6960 |
|   310P3        | 4          |  modelnet40          | -   | 7825 |
|   310P3        | 8          |  modelnet40          | -   | 7187 |
|   310P3        | 16         |  modelnet40          | -   | 6862 |
|   310P3        | 32         |  modelnet40          | -   | 7011 |
|   310P3        | 64         |  modelnet40          | -   | 6894 |

part2

| 芯片型号 | Batch Size | 数据集 | 精度                  | 性能   |
| --------- |------------| ---------- |---------------------|------|
|   310P3        | 1          |  modelnet40          | 88.4/class 92.4/ins | 5127 |
|   310P3        | 4          |  modelnet40          | 88.4/class 92.4/ins | 2725 |
|   310P3        | 8          |  modelnet40          | 88.4/class 92.4/ins | 4383 |
|   310P3        | 16         |  modelnet40          | 88.4/class 92.4/ins | 4481 |
|   310P3        | 32         |  modelnet40          | 88.4/class 92.4/ins | 4628 |
|   310P3        | 64         |  modelnet40          | 88.4/class 92.4/ins | 4641 |
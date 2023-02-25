# 3DUNet 模型-推理指导


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

3DUNet模型一般用于3D语义分割。


- 参考实现：

  ```
  url=https://github.com/mlcommons/inference.git
  commit_id=74353e3118356600c1c0f42c514e06da7247f4e8
  model_name=3DUNet
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                            | 数据排布格式 |
  | -------- | -------- | ------------------------------- | ------------ |
  | input    | FLOAT32  | batchsize x 4 x 224 x 224 x 160 | ND           |


- 输出数据

  | 输出数据 | 数据类型 | 大小                            | 数据排布格式 |
  | -------- | -------- | ------------------------------- | ------------ |
  | output   | FLOAT32  | batchsize x 4 x 224 x 224 x 160 | ND           |




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
   git clone --recurse-submodules https://github.com/mlcommons/inference.git
   cd inference
   git reset 74353e3118356600c1c0f42c514e06da7247f4e8 --hard
   cd vision/medical_imaging/3d-unet
   git clone https://github.com/MIC-DKFZ/nnUNet.git
   cd nnUNet/
   git reset b38c69b345b2f60cd0d053039669e8f988b0c0af --hard
   cd ../
   rm -rf nnUnet
   mv nnUNet nnUnet
   ```
   打补丁

   (1). 修改运行脚本Task043_BraTS_2019.py，在main函数中添加以下内容

           ```
           nnUNet_raw_data="./build/raw_data/nnUNet_raw_data"
           maybe_mkdir_p(nnUNet_raw_data)
           ```

   (2). 修改onnxruntime_SUT.py

      import头文件

      ```
      from ais_bench.infer.interface import InferSession 
      ```

      __init__函数中增加

      ```
      self.model = InferSession(0, model_path)
      ```

      注释self.sess:

      ```
      #self.sess = onnxruntime.InferenceSession(model_path)
      ```

      issue_queries函数中修改output

      ```
      output = self.model.infer([data[np.newaxis, ...]])[0].squeeze(0).astype(np.float16)
      ```
   
      或者直接通过patch文件进行修改：
      
       ```
       cd ../../../
       patch -p1 < ../3DUnet.patch
       cd ..
       ```
   
2. 安装依赖

   ```
   pip3 install -r requirements.txt
   ```

3. 编译环境，进入inference/loadgen目录，执行以下命令

   ```
   cd inference/loadgen
   CFLAGS="-std=c++14 -O3" python3 setup.py develop
   ```

   若失败，考虑升级gcc版本

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   该模型使用MICCAI_BraTS2019训练集中得部分数据进行测试，官方下载链接(需注册)：

   [BraTS2019数据集](https://www.med.upenn.edu/cbica/brats2019/data.html)
   
   将下载的训练集解压，将其放在inference/vision/medical_imaging/3d_unet/目录下，目录如下
   
    ```
   └─build
       └─MICCAI_BraTS_2019_Data_Training
           ├──HGG 
           ├──LGG
    ```
   
2. 数据预处理
   
   ```
   cd build
   mkdir postprocessed_data
   mkdir raw_data
   cd raw_data
   mkdir nnUNet_raw_data
   cd ../../
   python3 Task043_BraTS_2019.py
   python3 preprocess.py
   ```
   说明：运行preprocess.py的脚本前，需要先下载权重，即完成下模型转换的第一条


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   1. 下载权重链接网址：https://zenodo.org/record/3903982#.YL9Ky_n7SUk

   进入页面，下载fold_1.zip，在3d-unet目录下创建build/result目录，并将下载的fold_1.zip文件解压，将nnUNet目录放在result目录下，文件目录为：

   ```
   3d-unet/build/result/nnUNet/3d_fullres/Task043_BraTS2019/nnUNetTrainerV2__nnUNetPlansv2.mlperf.1/
   ```
   
   2. 运行unet_pytorch_to_onnx.py脚本。

   ```
   python3.7 unet_pytorch_to_onnx.py
   ```
   运行脚本导出onnx，onnx默认保存在build/model下，模型生成在build/model/目录下，分别是224_224_160.onnx单batch模型和224_224_160_dynamic_bs.onnx动态batch onnx

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
         atc --model=./build/model/224_224_160_dyanmic_bs.onnx --framework=5 --output=./build/model/3DUnet_bs{batch_size} --input_format=ND --input_shape="input:{batch_size},4,224,224,160" --log=info --soc_version={chip_name} --out_nodes="Conv_80:0"
         示例
         atc --model=./build/model/224_224_160_dyanmic_bs.onnx --framework=5 --output=./build/model/3DUnet_bs1 --input_format=ND --input_shape="input:1,4,224,224,160" --log=info --soc_version=Ascend310P3 --out_nodes="Conv_80:0"
         ```
   
            - 参数说明：
   
              -   --model：为ONNX模型文件。
              -   --framework：5代表ONNX模型。
              -   --output：输出的OM模型。
              -   --input\_format：输入数据的格式。
              -   --input\_shape：输入数据的shape。
              -   --log：日志级别。
              -   --soc\_version：处理器型号。
              -   --out_nodes：指定OM的输出节点。
           

2. 开始推理验证

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

        源代码已有相关数据处理代码封装，不容易转bin文件，故选择使用ais_bench的接口使用脚本的形式进行推理。
        ```
        python3 run.py --accuracy --backend onnxruntime --model ./build/model/3DUnet_${batch_size}
        示例
        python3 run.py --accuracy --backend onnxruntime --model ./build/model/3DUnet_bs1.om 
        ```
        - 参数说明
        - accuracy: 进行精度推理
        - backend: 使用框架
        - model: 使用的om模型

   3. 精度验证。

      上述代码运行完成后，会将结果进行打屏。

   4. 性能验证
      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model=${om_model_path} --loop=1000 --batchsize=${batch_size}
      示例
      python3 -m ais_bench --model=./build/model/3DUnet_bs1.om --loop=1000 --batchsize=1
      ```

      - 参数说明：
           - --model：om模型
           - --batchsize：模型batchsize
           - --loop: 循环次数



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度               | 310P性能  |
| -------- |------------| ------ |------------------|---------|
|     310P3     | 1          | BraTS2019 | mean tumor:0.853 | 6.26fps |
|     310P3     | 4          | BraTS2019 | -                | 5.99fps |
|     310P3     | 8          | BraTS2019 | -                | 5.62fps |

说明：模型源代码仅支持单batch推理，batch大于8,ais_bench工具由于超过内存无法进行测试
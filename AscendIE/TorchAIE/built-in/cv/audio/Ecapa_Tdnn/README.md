# ECAPA_TDNN模型-推理指导

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

ECAPA-TDNN基于人脸验证和计算机视觉相关领域的最新趋势，对传统的TDNN引入了多种改进。其中包括一维SE blocks，多层特征聚合（MFA）以及依赖于通道和上下文的统计池化。

- 参考实现：

  ```shell
  url=https://github.com/Joovvhan/ECAPA-TDNN.git
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                   | 数据排布格式 |
  | -------- |----------------------|--------| ------------ |
  | input   | FP32 | batchsize x 80 x 200 | ND     |

  - 输出数据

  | 输出数据   | 数据类型 | 大小               | 数据排布格式 |
  |--------| -------- |--------------------|--------|
  | output1      | FLOAT32 | batchsize x 192 | ND     |
  | output2 | FLOAT32 | batchsize x 200 x 1536 | ND     |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下依赖

  **表 1**  版本配套表


  | 配套                                                            | 版本     | 环境准备指导                                                                                          |
  |--------| ------- | ----------------------------------------------------------------------------------------------------- |
  | CANN                                                            | 7.1.T5.1.B113:7.0.0  | -                                                                                                     |
  | Python                                                          | 3.9.0  | -                                                                                                     |
  | PyTorch                                                         | 2.0.1  | -                                                                                                     |
  | 说明：芯片类型：Ascend310P3 | \      | \   


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

      ```
       获取推理部署代码
        git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
        cd ModelZoo-PyTorch/ACL_PyTorch/contrib/audio/Ecapa_Tdnn/ECAPA_TDNN
       获取源码
        git clone --recursive https://github.com/Joovvhan/ECAPA-TDNN.git
        mv ECAPA-TDNN ECAPA_TDNN
        export PYTHONPATH=$PYTHONPATH:./ECAPA_TDNN
        export PYTHONPATH=$PYTHONPATH:./ECAPA_TDNN/tacotron2
    ```

2. 安装依赖。

      ```
       pip install -r requirements.txt
      ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

用户需自行获取VoxCeleb1数据集中测试集（无需训练集），上传数据集到服务器中,必须要与preprocess.py同目录。目录结构如下：
   ```
    VoxCeleb1
    ├── id10270
       ├── 1zcIwhmdeo4
          ├── 00001.wav 
          ├── ... 
    ├── id10271
    ├── ...
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。
    
    在当前工作目录下，执行以下命令行,其中VoxCeleb为数据集相对路径，input/为模型所需的输入数据相对路径，speaker/为后续后处理所需标签文件的相对路径
   ```
    python3 preprocess.py VoxCeleb1 input/ speaker/
   ```

## 模型推理<a name="section741711594517"></a>
1. 获取权重文件。
    ```
    wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Ecapa_tdnn/PTH/checkpoint.zip
    unzip checkpoint.zip
    ```
   获取基准精度，作为精度对比参考， checkpoint为权重文件相对路径， VoxCeleb为数据集相对路径
    ```
    python3 get_originroc.py checkpoint VoxCeleb1
    ```

2. 生成trace模型(ts)
    ```
     使用与本README同目录的pytorch2onnx.py替换掉原工程同名文件
     python3 pytorch2onnx.py checkpoint ecapa_tdnn.onnx 
    ```

3. 保存编译优化模型（非必要，可不执行。后续执行的推理脚本包含编译优化过程）

    ```
     python export_torch_aie_ts.py --batch_size=1
    ```
   命令参数说明（参数见onnx2om.sh）：
    ```
     --torch_script_path：编译前的ts模型路径
     --soc_version：处理器型号
     --batch_size：模型batch size
     --save_path：编译后的模型存储路径
    ```


4. 执行推理脚本（包括性能验证）

    将pt_val.py放在./yolov3下，model_pt.py放在./yolov3/common/util下
     ```
      python pt_val.py --batch_size=64 --model="ecapa_tdnn_torch_aie_bs64.pt"
     ```
   命令参数说明（参数见onnx2om.sh）：
    ```
     --data_path：验证集数据根目录，默认"VoxCeleb1"
     --soc_version：处理器型号
     --model：输入模型路径
     --need_compile：是否需要进行模型编译（若使用export_torch_aie_ts.py输出的模型，则不用选该项）
     --batch_size：模型batch size
     --device_id：硬件编号
    ```
# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

 精度验证
   ```
     python postprocess.py result/output_bs1 speaker
   ```
命令参数说明（参数见onnx2om.sh）：
```
    --result/output_bs1：为推理结果所在路径
    --speaker：为标签数据所在路径
    --4：batch size
    --4648：样本总数
```

**表 2** ecapa_tdnn模型精度

| batchsize                                      | aie性能     | aie精度  |
|------------------------------------------------|-----------|--------|
| bs1                                            | 894.4216  | 0.9856 |
| bs4                                            | 2674.6597 | 0.9865 |
| bs8                                            | 3686.8627 | /      |
| bs16                                           | 692.3013  | /      |
| bs32                                           | 1358.1562 | /      |
| bs64                                           | 2645.4167 | /      |

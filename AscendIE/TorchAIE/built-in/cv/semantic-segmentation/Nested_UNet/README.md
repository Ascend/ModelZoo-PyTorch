# Nested_UNet模型-推理指导

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

UNet++由不同深度的U-Net组成，其解码器通过重新设计的跳接以相同的分辨率密集连接，主要用于医学图像分割任务。

- 参考实现：

  ```shell
  url=https://github.com/4uiiurz1/pytorch-nested-unet
  branch=master
  commit_id=557ea02f0b5d45ec171aae2282d2cd21562a633e
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- |-------------------------| ------------------------- | ------------ |
  | actual_input_1   | RGB_FP32 | batchsize x 3 x 96 x 96 | NCHW         |

- 输出数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- |-------------------------| ------------------------- | ------------ |
  | output_1   | RGB_FP32 | batchsize x 1 x 96 x 96 | NCHW         |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下依赖

  **表 1**  版本配套表


  | 配套                                                            | 版本    | 环境准备指导                                                                                          |
  | --------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
  | 固件与驱动                                                      | 23.0.rc1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 7.0.RC1.alpha003 | -                                                                                                     |
  | Python                                                          | 3.9.11  | -                                                                                                     |
  | PyTorch                                                         | 2.0.1   | -                                                                                                     |
  | 说明：芯片类型：Ascend310P3 | \       | \   


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

      ```
       1. 获取昇腾ModelZoo-PyTorch仓源码     
        git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
       2. 在同级目录下，下载第三方源码并打补丁
        git clone https://github.com/4uiiurz1/pytorch-nested-unet
        cd pytorch-nested-unet
        git reset --hard 557ea02f0b5d45ec171aae2282d2cd21562a633e
        patch -p1 < ../nested_unet.diff
        cd ..
    ```

2. 安装依赖。

      ```
       pip install -r requirements.txt
      ```

## 准备数据集<a name="section183221994411"></a>

1. 本模型使用2018 Data Science Bowl数据集进行推理测试。下载[地址](https://gitee.com/link?target=https%3A%2F%2Fwww.kaggle.com%2Fc%2Fdata-science-bowl-2018)。
2. 用户自行获取 stage1_train.zip 后，将文件解压并上传数据集到第三方源码的 inputs/data-science-bowl-2018 目录下。数据集及第三方源码的目录结构关系如下所示：
    
    ```
    pytorch-nested-unet/
    |-- LICENSE
    |-- README.md
    |-- archs.py
    |-- dataset.py
    |-- inputs
    |   `-- data-science-bowl-2018 
    |       `-- stage1_train # 解压后数据集
    |			|-- xxx
    |			|-- yyy
    |   		`-- ...
    |-- ...
    |-- preprocess_dsb2018.py # 数据集预处理脚本
    |-- ...
    `-- val_ids.txt
   ```

2. 执行原代码仓提供的数据集预处理脚本，生成处理后的数据集文件夹dsb2018_96。
   ```
    cd pytorch-nested-unet
    python3 preprocess_dsb2018.py
    cd ..
   ```

3. 将第2步得到的数据集转换为模型的输入数据。 执行 nested_unet_preprocess.py 脚本，完成数据预处理。
   ```
    python3 nested_unet_preprocess.py ./pytorch-nested-unet/inputs/dsb2018_96/images ${prep_data} ./pytorch-nested-unet/val_ids.txt

   ```
    参数说明：
    
    --参数1：原数据集所在路径。

    --参数2：生成数据集的路径。

    --参数3：验证集图像id文件。

## 模型推理<a name="section741711594517"></a>
1. 获取权重文件。
    ```
     wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Unet%2B%2B/PTH/nested_unet.pth
    ```


2. 生成trace模型(onnx, om, ts)
    ```
   首先使用本代码提供的nested_unet_pth2onnx.py替换原代码的同名脚本
   
     python3 nested_unet_pth2onnx.py ${pth_file} ${onnx_file}
   参数说明：
    --pth_file：权重文件。
    --onnx_file：生成 onnx 文件。
   
     source /usr/local/Ascend/ascend-toolkit/set_env.sh
   
     # bs = [1, 4, 8, 16, 32, 64]
     atc --framework=5 --model=./nested_unet.onnx --input_format=NCHW --input_shape="actual_input_1:${bs},3,96,96" --output=nested_unet_bs${bs} --log=error --soc_version=Ascend${chip_name}
    ```
    
    atc命令参数说明（参数见onnx2om.sh）：
    ```
    --model：为ONNX模型文件。
    --framework：5代表ONNX模型。
    --output：输出的OM模型。
    --input_format：输入数据的格式。
    --input_shape：输入数据的shape。
    --log：日志级别。
    --soc_version：处理器型号。
    ```

3. 保存编译优化模型（非必要，可不执行。若不执行，后续执行推理脚本时需要包含编译优化过程，入参加上--need_compile）

    ```
     python export_torch_aie_ts.py
    ```
   命令参数说明（参数见onnx2om.sh）：
    ```
     --torch_script_path：编译前的ts模型路径
     --soc_version：处理器型号
     --batch_size：模型batch size
     --save_path：编译后的模型存储路径
    ```


4. 执行推理脚本

    （1）推理脚本，包含性能测试。
     ```
      python3 pt_val.py --model nested_unet_torch_aie_bs4.pt --batch_size=4
     ```
   命令参数说明：
    ```
     --data_path：验证集数据根目录，默认"prep_data"
     --result_root_path：推理结果根目录，默认"result"
     --val_ids_file：val.txt文件路径，默认"./pytorch-nested-unet/val_ids.txt"
     --soc_version：处理器型号
     --model：输入模型路径
     --need_compile：是否需要进行模型编译（若参数model为export_torch_aie_ts.py输出的模型，则不用选该项）
     --batch_size：模型batch size。注意，若该参数不为1，则不会存储推理结果，仅输出性能
     --device_id：硬件编号
     --multi：将数据扩展多少倍进行推理。注意，若该参数不为1，则不会存储推理结果，仅输出性能
    ```
5. 精度验证

    （1）调用脚本与真值比对，可以获得精度结果。
     ```
      python3 nested_unet_postprocess.py ./result/result_bs${bs} ./pytorch-nested-unet/inputs/dsb2018_96/masks/0/
     ```
   命令参数说明：
    ```
    --参数1：推理输出目录。
    --参数2：真值所在目录。
    ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>



芯片型号 Ascend310P3。
dataloader生成未drop_last，已补满尾部batch

模型精度 bs1 = 0.8385

**表 2** 模型推理性能

| batch_size              | 性能（fps）   | 数据集扩大倍数 |
|-------------------------|-----------|---------|
| 1                       | 928.5740  | 1       |
| 4                       | 1883.2271 | 1       |
| 8                       | 2359.5540 | 1       |
| 16                      | 2239.7244 | 1       |
| 32                      | 1364.0031 | 200     |
| 64                      | 1703.1738 | 400     |

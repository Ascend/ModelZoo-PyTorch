# GaitSet模型-推理指导


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

本项目利用昇腾推理引擎`AscendIE`和框架推理插件`TorchAIE`，基于`pytorch框架`实现[GaitSet](https://github.com/AbnerHqC/GaitSet)模型在昇腾设备上的高性能推理。



- 参考实现：

  ```
  url=https://github.com/AbnerHqC/GaitSet
  commit_id=14ee4e67e39373cbb9c631d08afceaf3a23b72ce
  model_name=GaitSet
  ```
  

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 100 x 64 x 44 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型        | 大小 | 数据排布格式 |
  | -------- | ------------ | -------- | ------------ |
  | output1  | FLOAT32 | batchsize x 62 x 256  | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套  | 版本  | 环境准备指导  |
  |---------| ------- | ------------------------------------------------------------ |
  | 固件与驱动 | 23.0.rc1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN | 7.0.RC1.alpha003 | - |
  | Python | 3.9.11 | - |
  | PyTorch | 2.0.1 | - |
  | Torch_AIE | 6.3.rc2 | - |

- 安装依赖

   ```
   pip install -r requirements.txt
   ```


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   本项目基于Ascend ModelZoo中对应的离线推理工程完成，所以公用的数据前后处理脚本从离线推理工程中直接获取。
   
   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cp ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/classfication/GaitSet/GaitSet_post* .
   cp ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/classfication/GaitSet/GaitSet_pre* .
   cp ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/classfication/GaitSet/change.patch .
   git clone https://github.com/AbnerHqC/GaitSet.git
   cd GaitSet
   git reset --hard 14ee4e67e39373cbb9c631d08afceaf3a23b72ce
   git apply ../change.patch
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
   本模型支持CASIA-B图片的验证集。下载地址http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp  ，只下载DatasetB数据集。

   下载后的数据集内的压缩文件需要全部解压，解压后数据集内部的目录应为（`GaitDatasetB-silh`数据集）：数据集路径/对象序号/行走状态/角度，如
   ```
   GaitDatasetB-silh
   ├── 001      
   └── 002
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行GaitSet_preprocess_step1.py脚本

   ```
   python GaitSet_preprocess_step1.py --input_path=./GaitDatasetB-silh --output_path=./predata
   ```
   -   参数说明：

         -   input_path：数据集地址
         -   output：初步预处理保存地址

   执行GaitSet_preprocess_step2.py脚本，完成预处理
   ```
   mkdir CASIA-B-bin
   python GaitSet_preprocess_step2.py --data_path=./predata --bin_file_path=./CASIA-B-bin/
   ```   
   -   参数说明：

         -   data_path：初步预处理结果
         -   bin_file_path：预处理数据地址



## 模型推理<a name="section741711594517"></a>

  1. 执行推理。

    ```
    python inference.py
    ```
    推理结果保存在./result目录，推理结束后会输出当前batch的推理性能。


  2. 精度验证。

    调用脚本GaitSet_postprocess.py，可以获得Accuracy数据。

    ```
    python GaitSet_postprocess.py --output_path=./result
    ```

    - 参数说明：

      - output_path：推理结果保存路径
  
  3. 性能测试。

    ```
    python inference.py --input_shape 4 100 64 44
    ```
    - 参数说明：
      - --input_shape ：推理时输入shape, 可通过修改第1维测试不同batch的推理性能。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| Ascend310P3 | 1 | GaitDatasetB-silh | 95.512% | 796 FPS |
| Ascend310P3 | 4 | GaitDatasetB-silh | 95.512% | 877 FPS |
| Ascend310P3 | 8 | GaitDatasetB-silh | 95.512% | 900 FPS |
| Ascend310P3 | 16 | GaitDatasetB-silh | 95.512% | 922 FPS |
| Ascend310P3 | 32 | GaitDatasetB-silh | 95.512% | 941 FPS |
| Ascend310P3 | 64 | GaitDatasetB-silh | 95.512% | 925 FPS |

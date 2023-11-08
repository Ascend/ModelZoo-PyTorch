# RetinaFace模型-TorchAIE推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

本项目利用昇腾推理引擎`AscendIE`和框架推理插件`TorchAIE`，基于`pytorch框架`实现[RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)模型在昇腾设备上的高性能推理。


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                          | 数据排布格式 |
  | -------- |-----------------------------| ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 1000 x 1000 | NCHW         |


- 输出数据

  | 输出数据    | 数据类型  | 大小                     | 数据排布格式  |
  |---------|------------------------|----------|------------| 
  | output0 | FLOAT32 | batchsize x 41236 x 4  | ND         |
  | output1 | FLOAT32 | batchsize x 41236 x 2  | ND         |
  | output2 | FLOAT32 | batchsize x 41236 x 10 | ND         |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动 

  **表 1**  版本配套表

  | 配套                                                         | 版本      | 环境准备指导                                                 |
  |---------| ------- | ------------------------------------------------------------ |
  | 固件与驱动 | 23.0.rc1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN | 7.0.RC1.alpha003 | - |
  | Python | 3.9.11 | - |
  | PyTorch | 2.0.1 | - |
  | Torch_AIE | 6.3.rc2 | - |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>
1. 获取源码。

   ```
   git clone https://github.com/biubug6/Pytorch_Retinaface
   cd Pytorch_Retinaface
   git reset b984b4b775b2c4dced95c1eadd195a5c7d32a60b --hard
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   该模型使用[WiderFace](http://shuoyang1213.me/WIDERFACE/index.html)
   的3226张验证集进行测试。获取数据集并解压后将images文件夹放在/data/widerface/val文件夹下。目录结构如下：

   ```
   retinaface
   ├── data
      ├── widerface
         ├── val
            ├── images
               ├── 0-Parade
               ├── 1-Handshaking
               ├── ...
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行预处理脚本 data_preprocess.py，完成预处理。

   ```
   python3 data_preprocess.py
   ```

   运行成功后，生成的npy格式图片和预处理信息二进制文件默认放在./widerface文件夹下。目录结构如下：
      ```
   retinaface
      ├── widerface
         ├── prep
         └── prep_info
      ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为torchscript文件。

   1. 获取权重文件，存放在当前目录即可。

       [Retinaface基于mobilenet0.25的预训练权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Retinaface/PTH/mobilenet0.25_Final.pth)

   2. 导出ts文件。

      1. 使用pth2ts.py导出torchscript文件。

         ```
         python3 pth2ts.py -m ./mobilenet0.25_Final.pth
         ```

         获得retinaface.ts文件。

2. 开始推理验证。

   1. 执行推理。

        ```
         python3 inference.py
        ```
      执行结束可直接通过输出日志得到推理性能。

   3. 精度验证。
      1. 数据后处理

         ```
         python3 postprocess.py
         ```

      2. 计算精度

         如果是第一次运行精度计算需要运行第二步，编译评估文件，之后运行可直接执行第三步中的精度计算
         ```
         cd Pytorch_Retinaface/widerface_evaluate
         python3 setup.py build_ext --inplace
         python3 evaluation.py -p ../../widerface_result/
         ```
         注意：如果numpy版本不匹配，评价时可能遇到`AttributeError: module 'numpy' has no attribute 'float'.`
         错误，需要修改`retinaface/Pytorch_Retinaface/widerface_evaluate/box_overlaps.pyx`文件第12行为
         ```
         DTYPE = np.float64
         ```
         并重新执行
         ```
         python3 setup.py build_ext --inplace
         python3 evaluation.py -p ../../widerface_result/
         ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| 310P | 1 | WiderFace | 90.06%（Easy） 86.90%（Medium） 72.04%（Hard） | 48.77 it/s |

# Pelee模型-推理指导

## 概述

Pelee是基于PeleeNet和Single Shot MultiBox Detector方法结合提出的实时对象检测系统，用于移动设备端高性能检测模型。

参考论文：
Pelee: A Real-Time Object Detection System on Mobile Devices. (2018)

参考实现：
```shell
url=https://github.com/yxlijun/Pelee.Pytorch 
branch=master 
commit_id=1eab4106330f275ab3c5dfb910ddd79a5bac95ef
```

### 输入输出数据

- 输入数据

  | 输入数据 | 大小                      | 数据类型 | 数据排布格式 |
  | -------- | ------------------------- | -------- | ------------ |
  | image    | batchsize x 3 x 304 x 304 | RGB_FP32 | NCHW         |

- 输出数据

  | 输出数据 | 大小                  | 数据类型 | 数据排布格式 |
  | -------- | --------------------- | -------- | ------------ |
  | output1  | batch_size x 2976 x 4 | FLOAT32  | ND           |
  | output2  | 2976 x 21             | FLOAT32  | ND           |




## 推理环境准备

该模型需要以下插件与驱动

| 配套                                                         | 版本                                                         | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 固件与驱动                                                   | [1.0.15](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | [5.1.RC1](https://www.hiascend.com/software/cann/commercial?version=5.1.RC1) |                                                              |
| PyTorch                                                      | [1.5.0](https://github.com/pytorch/pytorch/tree/v1.5.0)      |                                                              |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 |                                                              |                                                              |



| 依赖名称    | 版本       |
| ----------- | ---------- |
| Python      | 3.8以上    |
| ONNX        | 1.7.0      |
| Pytorch     | 1.8.0      |
| TorchVision | 0.6.0      |
| numpy       | 1.22.0以上 |
| Pillow      | 7.2.0      |
| matplotlib  | 3.5.0      |
| addict      | 2.4.0      |
| tqdm        | 4.62.3     |
| Cpython     | 0.29.24    |





## 快速上手

#### 获取源码及安装

1. 下载Pelee开源源码

   ```
   git clone https://github.com/yxlijun/Pelee.Pytorch
   cd Pelee.Pytorch
   git checkout -b 1eab4106330f275ab3c5dfb910ddd79a5bac95ef
   ```

2. 上传Pelee.patch到和源码同一级目录下，执行patch命令。

   ```
   cd ..
   find ./Pelee.Pytorch -type f -name "*" -exec dos2unix {} \;
   patch -p0 < Pelee.patch
   ```

3. 在Pelee.Pytorch目录下编译开源代码

   ```
   cd Pelee.Pytorch
   bash make.sh
   ```

4. 上传到服务器任意目（如：/home/HwHiAiUser）。

   ```shell
   .
   |-- LICENSE
   |-- Pelee.patch
   |-- ReadMe.md
   |-- acl_net.py
   |-- aipp.config                         //aipp转换配置文件
   |-- atc.sh                              //onnx模型转换om模型脚本 
   |-- modelzoo_level.txt
   |-- pth2onnx.py                         //用于转换pth模型文件到onnx模型文件
   |-- requirements.txt
   ```

5. 拷贝源码和文件到Pelee.Pytorch目录下。

6. 请用户根据依赖列表和提供的requirments.txt以及自身环境准备依赖。

   ```shell
   cd Pelee.Pytorch
   pip3 install -r requirments.txt
   ```

   

#### 准备数据集

1. 获取原始数据集。

   本模型该模型使用VOC2007的4952张验证集进行测试，请用户自行获取该数据集，上传并解压数据集到服务器指定目录。

2. 数据预处理。

   数据预处理复用Pelee开源代码，具体参考test.py代码。

   

#### 模型推理

1. 模型转换。

   本模型基于开源框架PyTorch训练的Pelee进行模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      单击[Link](https://drive.google.com/open?id=16HparGAVhxTDByi5RylYCkxLZYducK9j)在PyTorch开源预训练模型中获取Pelee_VOC.pth权重文件。

   2. 导出onnx文件。

      pth2onnx.py脚本将.pth文件转换为.onnx文件，执行如下命令。

      ```
      python3 pth2onnx.py --config ./configs/Pelee_VOC.py -m ./Pelee_VOC.pth -o ./pelee_dynamic_bs.onnx
      ```

      第一个参数为输入权重文件路径，第二个参数为输出onnx文件路径。

      运行成功后，在当前目录生成pelee_dynamic_bs.onnx模型文件。

      使用ATC工具将.onnx文件转换为.om文件，需要.onnx算子版本需为13。在pth2onnx.py脚本中torch.onnx.export方法中的输入参数opset_version的值需为13。

   3. 用onnx-simplifier简化模型

      ```
      python3 -m onnxsim pelee_dynamic_bs.onnx pelee_dynamic_bs_sim.onnx --input-shape 1,3,304,304
      ```


   4. 改图优化
        
       修改softmax节点，在softmax前插入transpose
       
       ```shell
       python3 softmax.py pelee_dynamic_bs_sim.onnx pelee_dynamic_bs_modify.onnx
       ```
    
       - softmax.py修改模型节点需要和onnx模型中Softmax节点name保持一致。如果执行脚本报错时参考onnx图中Softmax节点的name。
       
       - ONNX改图依赖om_gener工具，下载链接：https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Official/nlp/VilBert_for_Pytorch/om_gener


   5. 修改atc.sh脚本，通过ATC工具使用脚本完成转换，具体的脚本示例如下：

      ${chip_name}可通过`npu-smi info`指令查看

       ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

      ```shell
      # 配置环境变量 
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      
      # 使用二进制输入时，执行如下命令。
      atc --model=./pelee_dynamic_bs_modify.onnx --framework=5 --output=pelee_bs1 --input_format=NCHW --input_shape="image:1,3,304,304" --log=info --soc_version=Ascend${chip_name} --insert_op_conf=aipp.config --enable_small_channel=1 # Ascend310P3
      
      # 使用二进制输入时，执行如下命令。test.py推理代码脚本需要适配bs=32
      atc --model=./pelee_dynamic_bs_modify.onnx --framework=5 --output=pelee_bs32 --input_format=NCHW --input_shape="image:32,3,304,304" --log=info --soc_version=Ascend${chip_name} --insert_op_conf=aipp.config --enable_small_channel=1 # Ascend310P3
      ```

   - 参数说明：
   - --model：为ONNX模型文件。
   - --framework：5代表ONNX模型。
   - --output：输出的OM模型。
   - --input_format：输入数据的格式。
   - --input_shape：输入数据的shape。
   - --log：日志等级。
   - --soc_version：部署芯片类型。
   - --insert_op_conf=aipp.config：插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。


2. 开始推理精度验证。

   ```shell
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python3 test.py --dataset VOC  --config ./configs/Pelee_VOC.py --device_id 0 --model pelee_bs1.om
   ```

## FAQ

1. 编译报错
pycocotools/_mask.c:10546:40: error: ‘PyTypeObject {aka struct _typeobject}’ has no member named ‘tp_print’; did you mean ‘tp_dict’?
   

问题原因：开源代码和第三方库兼容性问题
解决方案：修改源码报错行，把tp_print修改为tp_dict


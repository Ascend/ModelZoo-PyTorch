# CycleGAN模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

******




# 概述

CycleGAN是基于对抗生成的图像风格转换卷积神经网络，该网络具有两个生成器，这两个生成器可以互相转换图像风格。该网络的训练是一种无监督的，少样本也可以取得很好效果的网络。


- 参考实现：

  ```
  url=https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
  branch=master
  commit_id=9bcef69d5b39385d18afad3d5a839a02ae0b43e7
  model_name=CycleGAN
  ```



  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

## 输入输出数据

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 256 x 256 | NCHW         |


- 输出数据

  | 输出数据 | 大小                      | 数据类型 | 数据排布格式 |
  | -------- | ------------------------- | -------- | ------------ |
  | output1  | batchsize x 3 x 256 x 256 | RGB_FP32 | NCHW         |


# 推理环境准备

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.6.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手

1. 安装依赖。

   **进入CycleGAN项目文件夹下。**
   文件夹下为：
   ```
   ├── CycleGAN.patch
   ├── CycleGAN_postprocess.py
   ├── CycleGAN_preprocess.py
   ├── CycleGAN_pth2onnx.py
   ├── LICENSE
   ├── README.md
   ├── modelzoo_level.txt
   ├── parse.py
   └── requirements.txt
   ```
   安装所需要的python依赖包。
   ```
   pip3 install -r requirment.txt
   ```


## 准备数据集

1. 获取原始数据集。

   该模型使用[maps数据集](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/maps.zip)的testA和testB各1098张验证集进行测试，将maps放至于**新建的datasets目录**下。

   ```
   maps
   ├── test
   ├── testA
   ├── testB
   ├── train
   ├── trainA
   ├── trainB
   ├── val
   ├── valA
   ├── valB
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行CycleGAN_preprocess.py脚本，完成预处理。

   ```
   python3 CycleGAN_preprocess.py --src_path_testA=./datasets/maps/testA/   --save_pathTestA_dst=datasetsDst/maps/testA/  --src_path_testB=./datasets/maps/testB/ --save_pathTestB_dst=./datasetsDst/maps/testB/
   ```

   参数说明：

   ```
    --src_path_testA ：指向航拍数据转卫星地图的测试集。
    --src_path_testB：指向卫星地图转航拍的测试集目录。
    --save_pathTestA_dst：指向保存航拍数据转卫星地图的测试集改变图像尺寸后的二进制数据存储目录。
    --save_pathTestB_dst：指向保存卫星地图转航拍的测试集改变图像尺寸后的二进制数据存储目录。
   ```

## 模型推理

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       - [官方CycleGAN pth权重文件](http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/)
       - [获取A800-9000训练的pth文件,该链接为百度网盘链接，提取码为：1234](https://pan.baidu.com/s/1YqHkce2wUw-W8_VY9dYD_w))

       将模型放至指定当前目录下

       ```
       mkdir checkpoints/maps_cycle_gan/ -p
       mv latest_net_G_A.pth  ./checkpoints/maps_cycle_gan/
       mv latest_net_G_B.pth  ./checkpoints/maps_cycle_gan/
       ```

   2. 使用开源仓代码完成模型代码的生成

      ```
      git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
      cd pytorch-CycleGAN-and-pix2pix
      git reset 9bcef69d5b39385d18afad3d5a839a02ae0b43e7 --hard

      patch -p1 < ../CycleGAN.patch
      cp ./models/networks.py ../
      cd ..
      ```

      开源仓中的生成器采用的padding类型为`ReflectionPad2d`，由于在转om格式模型的时候，会出现算子不兼容问题导致om模型转换失败，这里我们将改padding类型替换为`ZeroPad2d`。

   3. 导出onnx文件。

      1. 使用CycleGAN_pth2onnx.py导出onnx文件。

         运行CycleGAN_pth2onnx.py脚本。

         ```
         python3 CycleGAN_pth2onnx.py --model_ga_path=./checkpoints/maps_cycle_gan/latest_net_G_A.pth --model_gb_path=./checkpoints/maps_cycle_gan/latest_net_G_B.pth --onnx_path=./   --model_ga_onnx_name=model_Ga.onnx    --model_gb_onnx_name=model_Gb.onnx
         ```

         获得model_Ga.onnx和model_Gb.onnx文件。

         参数说明

         ```
         --model_ga_path ：航拍转卫星地图的生成器权重存储路径。
         --model_gb_path：卫星地图转航拍的生成器权重存储路径。
         --onnx_path：pth格式转onnx格式的存储路径，后面一定要带斜杠‘/’。
         --model_ga_onnx_name ：onnx格式的航拍转卫星地图的生成器名称。
         --model_gb_onnx_name ：onnx格式的卫星地图转航拍的生成器名称。
         ```

   4. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：**
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
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
         #该设备芯片名为Ascend310P3
         #（自行替换）
         export chip_name=310P3
         ```

      3. 执行ATC命令。

         * batch size为1

         ```
         atc --framework=5 --model=./model_Ga.onnx --output=CycleGAN_Ga_bs1_${chip_name} --input_format=NCHW --input_shape="img_sat_maps:1,3,256,256" --out_nodes="Tanh_156:0" --log=null --soc_version=Ascend${chip_name}

         atc --framework=5 --model=./model_Gb.onnx --output=CycleGAN_Gb_bs1_${chip_name} --input_format=NCHW --input_shape="img_maps_sat:1,3,256,256" --out_nodes="Tanh_156:0" --log=null --soc_version=Ascend${chip_name}
         ```

         * batch size为16

         ```
         atc --framework=5 --model=./model_Ga.onnx --output=CycleGAN_Ga_bs16_${chip_name} --input_format=NCHW --input_shape="img_sat_maps:16,3,256,256" --out_nodes="Tanh_156:0" --log=null --soc_version=Ascend${chip_name}

         atc --framework=5 --model=./model_Gb.onnx --output=CycleGAN_Gb_bs16_${chip_name} --input_format=NCHW --input_shape="img_maps_sat:16,3,256,256" --out_nodes="Tanh_156:0" --log=null --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --insert\_op\_conf=aipp\_resnet34.config:  AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。

           运行成功后分别生成`“CycleGAN_Ga_bs1_${chip_name}.om”，“CycleGAN_Ga_bs1_${chip_name}.om”`模型文件。其他batchsize可通过更改`output`和`input_shape`相应字段来修改。

2. 开始推理验证。

a.  使用ais-infer工具进行推理。

按照此工具的编译、安装步骤完成安装

 https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer

b.  执行推理。

找到并使用正确的`ais-bench_workload/tool/ais_infer/ais_infer.py`路径。

   ```
mkdir ./result

python3 ais_infer.py --model=./CycleGAN_Ga_bs1_${chip_name}.om --input=./datasetsDst/maps/testA/ --output=./result/ --outfmt=BIN --batchsize=1

python3 ais_infer.py --model=./CycleGAN_Gb_bs1_${chip_name}.om --input=./datasetsDst/maps/testB/ --output=./result/ --outfmt=BIN --batchsize=1

# 在INFO信息中找到output path目录，重命名为容易辨识的名称
[INFO] output path:./result/2022_09_03-03_48_48
# 示例如下
mv ./result/2022_09_03-03_48_48 ./result/Ga_bs1
mv ./result/2022_09_03-06_28_18 ./result/Gb_bs1
   ```
-   参数说明：

```
--model：模型地址
--input：预处理完的数据集文件夹
--output：推理结果保存地址
--outfmt：输出格式
--batchsize：batchsize大小
```

>**说明：**
>执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

c.  精度验证。
    调用脚本与onnx模型的结果进行对比，采用平均余弦相似度进行评价。

   ```
python3 CycleGAN_postprocess.py --dataroot=./datasets/maps/testA/ --npu_bin_file=./result/Ga_bs1/ --onnx_path=./ --om_save --onnx_save

python3 CycleGAN_postprocess.py --dataroot=./datasets/maps/testB/ --npu_bin_file=./result/Gb_bs1/ --onnx_path=./ --om_save --onnx_save
   ```

参数说明：

```
--onnx_path ：onnx格式模型的存储路径。
--model_ga_onnx_name： 航拍地图转卫星地图生成器的名称。
--model_gb_onnx_name： 卫星地图转航拍地图生成器的名称。
--dataroot：原始数据集maps的路径。
--npu_bin_file： 上一步aipp推理后保存的结果路径，路径后面一定要带斜杠‘/’。
--om_save: 是否保存om模型后处理的结果，如果使用该参数，保存在./result/Ga_bs1/om/目录下
--onnx_save: 是否保存onnx模型后处理的结果，如果使用该参数，保存在./result/Ga_bs1/onnx/目录下
```

# 模型推理性能&精度

调用ACL接口推理计算，性能参考下列数据。

**sat2maps模型**

| 芯片型号 | Batch Size | 数据集 | 精度      | 性能        |
| -------- | ---------- | ------ | --------- | ----------- |
| 310p     | 1          | maps   | 1.0       | 245.9075868 |
| 310p     | 16         | maps   | 1.0       | 215.8088165 |
| 310      | 1          | maps   | 0.9999999 | 43.83397453 |
| 310      | 16         | maps   | 0.9999999 | 42.42774673 |

**maps2sat模型**

| 芯片型号 | Batch Size | 数据集 | 精度      | 性能        |
| -------- | ---------- | ------ | --------- | ----------- |
| 310p     | 1          | maps   | 0.9990766 | 247.0199008 |
| 310p     | 16         | maps   | 0.9990766 | 213.8217157 |
| 310      | 1          | maps   | 0.9990409 | 43.84165193 |
| 310      | 16         | maps   | 0.9990409 | 42.39408798 |

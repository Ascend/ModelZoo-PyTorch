# InceptionResNetV2模型-推理指导

- [概述](#概述)

- [输入输出数据](#输入输出数据)

- [推理环境准备](#推理环境准备)

- [快速上手](#快速上手)

   - [安装依赖包](#安装依赖包)
   - [准备数据集](#准备数据集)
   - [模型推理](#模型推理)

- [模型推理性能和精度](#模型推理性能和精度)

  ******



## 概述

InceptionResNetV2结合了ResNet与Inception网络的特点，在Inception网络的基础上加入了残差连接（Residual Connections），加快了网络的训练速度，同时增大了网络的容量和复杂度。InceptionResNetV2在ImageNet数据集上取得了相比于原始的ResNet和Inception网络更高的的分类准确率。



- 参考实现：

  ```
  url=https://github.com/Cadene/pretrained-models.pytorch.git
  branch=master
  ommit_id=3c92fbda001b6369968e7cb1a5706ee6bf6c9fd7
  model_name=inceptionresnetv2
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
  | input    | RGB_FP32 | batchsize x 3 x 299 x 299 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | batchsize x 1000 | FLOAT32  | ND   |



## 推理环境准备

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

<table>
<thead>
  <tr>
    <th>配套</th>
    <th>版本</th>
    <th>环境准备指导</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>固件与驱动</td>
    <td>1.0.15</td>
    <td><a href="https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies" target="_blank" rel="noopener noreferrer">Pytorch框架推理环境准备</a></td>
  </tr>
  <tr>
    <td>CANN</td>
    <td>5.1.RC1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Python</td>
    <td>3.7.5</td>
    <td>-</td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>1.5.0及以上</td>
    <td>-</td>
  </tr>
  <tr>
    <td>onnx</td>
    <td>1.7.0及以上</td>
    <td>-</td>
  </tr>
  <tr>
    <td>torchvision</td>
    <td>0.6.0及以上</td>
    <td>-</td>
  </tr>
  <tr>
    <td>numpy</td>
    <td>1.20.3及以上</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Pillow</td>
    <td>8.2.0及以上</td>
    <td>-</td>
  </tr>
  <tr>
    <td>opencv-python</td>
    <td>4.5.2.54及以上</td>
    <td>-</td>
  </tr>
  <tr>
    <td colspan="3">说明：请用户根据自己的运行环境自行安装所需依赖。 x86架构：PyTorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install *包名* 安装。Arm架构：PyTorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install *包名* 安装。</td>
  </tr>
</tbody>
</table>


## 快速上手

### 安装依赖包

1. 安装依赖包。

   ```
   pip3 install -r requirment.txt
   ```


### 准备数据集

1. 获取原始数据集。

   该模型使用ImageNet官网的5万张验证集进行测试，图片与标签分别存放在/home/DATASETS/imagenet/val与/home/DATASETS/imagenet/val_label.txt

2. 数据预处理。

   执行预处理脚本，将原始数据集转换为模型输入的bin文件，存放在当前目录下的prep_dataset文件夹中
   ```
   python3.7 imagenet_torch_preprocess.py inceptionresnetv2 /root/datasets/imagenet/val ./prep_dataset
   ```

### 模型推理

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      下载权重文件，[链接](https://gitee.com/link?target=http%3A%2F%2Fdata.lip6.fr%2Fcadene%2Fpretrainedmodels%2Finceptionresnetv2-520b38e4.pth)
       ```
       wget http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth
       ```

   2. 导出onnx文件。

      1. 使用inceptionresnetv2_pth2onnx.py导出onnx文件。

         执行inceptionresnetv2_pth2onnx.py脚本，生成onnx模型文件

         ```
         python3.7 inceptionresnetv2_pth2onnx.py inceptionresnetv2-520b38e4.pth inceptionresnetv2_dynamic_bs.onnx
         ```

         运行成功后生成inceptionresnetv2_dynamic_bs.onnx文件。


   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
         ```

         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名称为Ascend310P3 （自行替换）
         回显如下：
         +--------------------------------------------------------------------------------------------+
         | npu-smi 22.0.0                       Version: 22.0.2                                       |
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 17.1         55                0    / 0              |
         | 0       0         | 0000:86:00.0    | 0            932  / 21534                            |
         +===================+=================+======================================================+

         ```

      3. 执行ATC命令。

         使用atc将onnx模型转换为om模型文件，工具使用方法可以参考《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。生成转换batch size为1的om模型的命令如下，对于其他的batch size，可作相应的修改。
         ```
         atc --framework=5 --model=inceptionresnetv2_dynamic_bs.onnx --output=inceptionresnetv2_bs1 --input_format=NCHW --input_shape="image:1,3,299,299" --log=debug --soc_version=Ascend${chip_name}
         ```

         参数说明：

            - --framework：5代表ONNX模型
            - --model：为ONNX模型文件路径
            - --output：输出的OM模型
            - --input\_format：输入数据的格式
            - --input\_shape：输入数据的shape
            - --log：日志级别
            - --soc\_version：处理器型号

         运行成功后生成inceptionresnetv2_bs1.om模型文件。

2. 开始推理验证。

   a.  使用ais-infer工具进行推理。

      执行命令增加工具可执行权限，并根据OS架构选择工具

      ```
      chmod u+x ais_infer.py
      ```

   b.  执行推理。

      使用batch size为1的om模型文件进行推理，其他batch size可作相应的修改
      ```
      python3.7 ais_infer.py --model ./inceptionresnetv2_bs1.om --batchsize 1 --input ./prep_dataset --output ./result --outfmt TXT --device 0
      ```

      参数说明：

         --model：需要进行推理的om模型路径
         --batchsize：om模型文件的batch size大小，用于结果吞吐率计算
         --input：模型需要的输入，支持bin文件和目录，若不加该参数，会自动生成都为0的数据
         --output：推理数据输出路径
         --outfmt：输出数据的格式，默认为“BIN”，可取值“NPY”、“BIN”、“TXT”
         --device：NPU设备编号

      推理后的输出在当前目录的result文件夹下。

      >**说明：** 
      >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见《[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)》。

   c.  精度验证。

      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在当前目录下perf文件夹中的perf_bs1.json文件中。

      在当前目录下创建文件夹perf
      ```
      mkdir ./perf
      ```

      调用脚本与数据集标签val\_label.txt比对，生成精度验证结果文件，注意需要首先删除存放推理结果的文件夹中的summary.json文件，否则会出现错误。
      ```
      rm ./result/2022_08_26-20_40_22/sumary.json

      python3.7 imagenet_acc_eval.py ./result/2022_08_26-20_40_22/  /home/DATASET/imagenet/val_label.txt ./perf perf_bs1.json
      ```

      参数说明：

         ./result/2022_08_26-20_40_22/：为生成推理结果所在路径，2022_08_26-20_40_22为ais-infer工具自动生成的目录名
    
         val_label.txt：为标签数据

         ./perf：为生成结果文件所在目录
    
         perf_bs1.json：为生成结果文件名

## 模型推理性能和精度

调用ACL接口推理计算，精度和性能参考下列数据。

### 精度对比
|           | Top1 Accuracy (%) | Top5 Accuracy (%) |
|:---------:|:-----------------:|:-----------------:|
|  310精度  |       80.15       |       95.24       |
| 310P3精度 |       80.15       |       95.24       |

将得到的om离线模型推理在310P3上的TopN精度与310上的TopN精度对比(此处为最优batch的精度，其他batch的精度与最优batch的精度无差别)，310P3上的TopN精度与310上的TopN精度无差别，精度达标。

### 性能对比

|           |   310   |   310P3  |    T4   | 310P3/310 | 310P3/T4 |
|:---------:|:-------:|:--------:|:-------:|:---------:|:--------:|
|    bs1    | 429.888 |  452.827 | 319.489 |  1.05336  | 1.417348 |
|    bs4    | 571.456 |  1187.48 | 550.808 |  2.07799  | 2.155887 |
|    bs8    | 676.092 | 1286.369 |  618.31 |  1.902654 |  2.08046 |
|    bs16   | 696.308 | 1065.154 | 667.735 |  1.529717 | 1.595175 |
|    bs32   | 693.756 |  851.128 | 677.696 |  1.226841 | 1.255914 |
|    bs64   |  692.02 |  756.475 | 672.078 |  1.09314  | 1.125576 |
| 最优batch | 696.308 | 1286.369 | 677.696 |  1.847414 |  1.89815 |

310最优batch为：bs16

310P3最优batch为：bs8

T4最优batch为：bs32


最优性能比(310P3 / 310)为1286.369 / 696.308 = 1.847倍

最优性能比(310P3 / T4)为1286.369 / 677.696 = 1.898倍

最优batch：310P3大于310的1.2倍，310P3大于T4的1.6倍，性能达标

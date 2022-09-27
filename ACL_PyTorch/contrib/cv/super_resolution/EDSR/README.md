# EDSR模型-推理指导

-   [概述](#概述)
-   [推理环境准备](#推理环境准备)
-   [快速上手](#快速上手)
	-   [准备数据集](#准备数据集)
	-   [模型推理](#模型推理)
-   [模型推理性能&精度](#模型推理性能&精度)


******

# 概述

论文通过提出EDSR模型移除卷积网络中不重要的模块并且扩大模型的规模，使网络的性能得到提升。


- 参考实现：

  ```
  url=https://github.com/sanghyun-son/EDSR-PyTorch.git
  branch=master
  commit_id=9d3bb0ec620ea2ac1b5e5e7a32b0133fbba66fd2
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

  | 输入数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- | -------- | --------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 1020 x 1020 | NCHW         |


- 输出数据

  | 输出数据 | 大小                        | 数据类型 | 数据排布格式 |
  | -------- | --------------------------- | -------- | ------------ |
  | output   | batchsize x 3 x 1020 x 1020 | RGB_FP32 | NCHW         |



## 文件结构

```
EDSR
├── edsr_postprocess.py    //验证推理结果脚本，给出Accuracy 
├── edsr_pth2onnx.py       //用于转换模型文件到onnx文件 
├── edsr_preprocess.py.py  //数据集预处理脚本，通过均值方差处理归一化图片
├── edsr.diff   		   //模型补丁 
├── get_max_size.py        //读取数据大小
├── requirements.txt       //模型安装需求
├── README.md              //模型推理指导      
```



# 推理环境准备

- 该模型需要以下插件与驱动


| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.7.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手

1.获取源代码

```
git clone https://github.com/sanghyun-son/EDSR-PyTorch.git
```

2.安装依赖。

```
pip install -r requirements.txt
```



## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，该模型使用[DIV2K官网](https://data.vision.ee.ethz.ch/cvl/DIV2K/)的100张验证集进行测试
   其中，低分辨率图像(LR)采用bicubic x2处理(Validation Data Track 1 bicubic downscaling x2 (LR images))，高分辨率图像(HR)采用原图验证集(Validation Data (HR images))。

   DIV2K验证集目录结构参考如下所示。

   ```
   ├── DIV2K              
         ├──HR  
              │──图片1
              │──图片2
              │   ...       
   ```

2. 数据预处理。
   执行预处理脚本，生成数据集预处理后的bin文件
   ```
   python3.7 edsr_preprocess.py -s /root/datasets/div2k/LR -d ./prep_data --save_img
   ```

   **说明**
   
   >第一个参数为数据集文件位置，第二个为输出bin文件位置



## 模型推理

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1.获取权重文件。

   下载地址[pth权重文件](https://cv.snu.ac.kr/research/EDSR/model_pytorch.tar) 
   
   

   2.修改模型。
   
   ​	在不影响原仓库功能的前提下实现对官方转换api的支持
   
       cd..
       patch -p1 < ./edsr.diff
       在File to patch一行输入路径：
       ./EDSR-PyTorch/src/model/__init__.py
   
   
   
   3.确定onnx输入输出的尺寸。
   
   ​	为了增加精度，本指导采用对于不满足尺寸大小要求的图像的右侧和下方填充0的方式	来使其输入图像达到尺寸大小要求。因此首先要获得需要的尺寸大小，通过命令行中	运行如下脚本
   
   ```
   python3.7 get_max_size.py --dir /root/datasets/div2k/LR
   ```
   
   ​	对于div2k数据集中scale为2的缩放，尺寸大小应为1020。
   
   
   
   4.导出onnx文件。
   
   ​	使用pth2onnx.py导出onnx文件。
   
   ​	运行pth2onnx.py。
   
   ```
   python3.7 edsr_pth2onnx.py --pth edsr_x2.pt --onnx edsr_x2.onnx --size 1020
   ```
   
   ​	 **说明**  
   
    >第一个参数为pth文件权重位置，第二个参数为输出onnx文件位置及命名，第三个为数据大小
   
   
   
   
   
   5.使用ATC工具将ONNX模型转OM模型。
   
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
      atc --model=edsr_x2.onnx --framework=5 --output=edsr_x2 --input_format=NCHW --input_shape="input.1:1,3,1020,1020" --log=debug --soc_version=Ascend${chip_name} --fusion_switch_file=switch.cfg
      ```
   
      - 参数说明：
   
        - --model：为ONNX模型文件。
        
        - --framework：5代表ONNX模型。
        
        - --output：输出的OM模型。
        
        - --input_format：输入数据的格式。
   
        - --input_shape：输入数据的shape。
        
        - --log：日志级别。
        
        - --soc_version：处理器型号。
        
          
        

​			运行成功后生成模型文件。



2. 开始推理验证。

​	a. 使用ais-infer工具进行推理。

ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]



​	b.  执行推理。

```
python3.7 ais_infer.py --model ./edsr_x2.om --input ./prep_data/bin --output ./out --batchsize 1
```

​	推理后的输出默认在当前目录下。

**说明**

>执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见使用文档。



​	c.  精度验证。

    python3.7 edsr_postprocess.py --res ./out --HR /root/datasets/div2k/HR

**说明**

>第一个参数为ais-infer输出目录，第二个为数据集配套标签



​	d. 性能验证。

由于T4服务器上的显卡显存有限，性能比较时选择的是size为256的onnx与om模型。

d-1 onnx模型转换

size：256的onnx模型生成命令

```
python3.7 edsr_pth2onnx.py --pth edsr_x2.pt --onnx edsr_x2_256.onnx --size 256
```



d-2 om模型转换

size：256的om模型生成命令示例如下

```
atc --model=edsr_x2_256.onnx --framework=5 --output=edsr_x2_bs1 --input_format=NCHW --input_shape="input.1:1,3,256,256" --log=debug --soc_version=Ascend${chip_name} --fusion_switch_file=switch.cfg
```



d-3 性能测试

npu测试命令示例如下：

```
python3.7 ais_infer.py --model /edsr_x2_256_bs1.om --output ./out --batchsize 1 --loop 20
```



# 模型推理性能&精度

1.精度对比。

|      | Acc   |
| ---- | ----- |
| 310  | 34.6% |
| 310P | 34.6% |

精度达标



2.性能参考下列数据。

|      | 310    | 310P    | T4     | 310P/310 | 310P/T4 |
| ---- | ------ | ------- | ------ | -------- | ------- |
| bs1  | 87.543 | 121.239 | 90.272 | 1.384    | 1.343   |
| bs4  | 83.437 | 109.006 | 91.070 | 1.306    | 1.196   |
| bs8  | 83.219 | 113.254 | 92.413 | 1.360    | 1.225   |
| bs16 | 83.504 | 111.150 | 94.146 | 1.331    | 1.180   |
| bs32 | 83.930 | 108.931 | 92.935 | 1.297    | 1.172   |
| bs64 | 83.503 | 107.451 | 91.255 | 1.286    | 1.177   |


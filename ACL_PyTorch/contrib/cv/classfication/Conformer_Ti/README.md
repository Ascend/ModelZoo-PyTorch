# Conformer_Ti模型-推理指导

-   [概述](#概述)
-   [推理环境准备](#推理环境准备)
-   [快速上手](#快速上手)
	-   [准备数据集](#准备数据集)
	-   [模型推理](#模型推理)
-   [模型推理性能精度](#模型推理性能和精度)


******

# 概述

Conformer_Ti是一种新型的图像分类网络，由卷积神经网络（CNN）和注意力网络（Transformer）两个分类网络组成。另一个主要特征是FCU模块，该模块允许特征信息在两个学习网络之间交互。这些特征允许Conformer_Ti实现更好的分类性能。


- 参考实现：

  ```
  url=https://github.com/pengzhiliang/Conformer
  commit_id=815aaad3ef5dbdfcf1e11368891416c2d7478cb1
  model_name=Conformer_Ti
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
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 1 x 1000 | FLOAT32  | ND           |



## 文件结构

```
Conformer_Ti 
├── conformer_postprocess.py    //验证推理结果脚本，给出Accuracy 
├── conformer_pth2onnx.py       //用于转换模型文件到onnx文件 
├── conformer_preprocess.py.py  //数据集预处理脚本，通过均值方差处理归一化图片
├── conformer_ti_change.patch   //模型补丁 
├── onnx_optimize.py            //onnx模型调优文件
├── op_precision.ini            //om高性能模式
├── requirements.txt            //模型安装需求
├── README.md                   //模型推理指导
├── modelzoo_level.txt          
```





# 推理环境准备

- 该模型需要以下插件与驱动


| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.7.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手



1. 获取源代码

   ```
   git clone https://github.com/pengzhiliang/Conformer
   ```
   
1. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```



## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集为ImageNet2012的验证集，将数据集上传到服务器任意路径下并解压。

   ImageNet2012验证集目录结构参考如下所示。

   ```
   ├── ImageNet2012              
         ├──val  
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...
         ├──val_lable.txt  
   ```

2. 数据预处理。
   执行预处理脚本，生成数据集预处理后的bin文件
   ```
   python3 conformer_preprocess.py resnet /location/imagenet/val ./val_bin
   ```

   **说明**
   
   >第一个参数为数据集类型，该模型为'resnet'，第二个参数为数据集文件位置，第三个为输出bin文件位置及命名



## 模型推理

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1.获取权重文件。

   下载地址[pth权重文件](https://pan.baidu.com/share/init?surl=2AblBmhUu5gnYsPjnDE_Jg) 提取码：hzhm
   
   

   2.修改模型。
   
   ​	为规范模型输出，先对模型进行修改
   
       patch -p0 ./Conformer/conformer.py conformer_ti_change.patch
   
   
   
   3.导出onnx文件。
   
   1. 使用pth2onnx.py导出onnx文件。
   
      移动pth2onnx.py文件到Conformer源代码文件夹下
   
      ```
      mv conformer_pth2onnx.py ./Conformer/
      ```
   
      运行pth2onnx.py。

      ```
      python3 ./Conformer/conformer_pth2onnx.py ./Conformer_tiny_patch16.pth ./conformer_ti.onnx
      ```

       **说明**  
   
       >第一个参数为pth文件权重位置，第二个参数为输出onnx文件位置及命名
   
      
   
   2. 优化ONNX文件。
   
      安装Magiconnx库
      
      ```
      git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
      cd MagicONNX
      pip3 install .
      ```
      
      优化onnx模型
      
      ```
      python3 onnx_optimize.py ./conformer_ti.onnx 8 ./conformer_ti_bs8.onnx
      ```
      
      **说明**
      
      >第一个参数为原onnx文件名，第二个为batch size大小，第三个为输出onnx文件位置及命名
   
   
   
   4.使用ATC工具将ONNX模型转OM模型。
   
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
      atc --framework=5 --model=conformer_ti_bs8.onnx --output=conformer_ti_bs8 --input_format=NCHW --input_shape="image:8,3,224,224" --log=debug --soc_version={chip_name} --op_precision_mode=./op_precision.ini
      ```
   
      - 参数说明：
   
        - --model：为ONNX模型文件。
        
        - --framework：5代表ONNX模型。
        
        - --output：输出的OM模型。
        
        - --input_format：输入数据的格式。
   
        - --input_shape：输入数据的shape。
        
        - --log：日志级别。
        
        - --soc_version：处理器型号。
        
        - --op_precision_mode: 高性能模式
        
          
        

​			运行成功后生成模型文件。



2. 开始推理验证。

​	1.  使用ais-infer工具进行推理。

​		AisBench推理工具，该工具包含前端和后端两部分。 后端基于c+开发，实现通用推理功能； 前端基于python开发，实现用户界面功能。获取工具及使用方法		可以参考https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer



​	2.  执行推理。

```
python3.7 ais_infer.py --model /location/conformer_ti_bs1.om --input /location/val_bin --output /location/out_bs1/ --batchsize 1 --outfmt TXT 
```



​	3.  精度验证。

​		调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

    python3.7 conformer_postprocess.py ./out_bs1/out_data/ ./val_label.txt ./ result.json

​		**说明**

>第一个参数为ais-infer输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名



# 模型推理性能和精度

1.精度对比。

开源Top1精度

```
CMC Scores  DukeMTMC-reID
  top-1          81.3%
```
推理Top1精度

```
CMC Scores  DukeMTMC-reID
  top-1          80.8%
```

精度达标



2.性能参考下列数据。

|      | 310P    | 310P_gelu高性能+onnx修改 | T4         | 310P_gelu高性能+onnx修改/T4 |
| ------ | --------- | -------------------------- | ------------ | ----------------------------- |
| bs1  | 232.558 | 336.564               | 123.253FPS | 2.7                         |
| bs4  | 390.244 | 676.544            | 162.004FPS | 4.2                      |
| bs8  | 529.801 | 867.716           | 188.522FPS | 4.6                        |
| bs16 | 493.903 | 822.330           | 445.794FPS | 1.8                        |
| bs32 | 451.340 | 676.534             | 228.578FPS | 2.9                         |

npu在bs为8时性能最佳，gpu在bs为16时性能最佳，两者对比：
``` 
Ascend 310P/gpu=867.716/445.794=1.95倍
```
性能达标 
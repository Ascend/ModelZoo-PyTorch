# DeeplabV3模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)
  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)
  
- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)



# 概述

DeeplabV3是一个经典的图像语义分割网络，在v1和v2版本基础上进行改进，多尺度(multiple scales)分割物体，设计了串行和并行的带孔卷积模块，采用多种不同的atrous rates来获取多尺度的内容信息，提出 Atrous Spatial Pyramid Pooling(ASPP)模块, 挖掘不同尺度的卷积特征，以及编码了全局内容信息的图像层特征，提升图像分割效果。

- 参考论文：[Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation." arXiv preprint arXiv:1706.05587 (2017).](https://arxiv.org/pdf/1706.05587.pdf)


- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmsegmentation.git
  branch=master
  commit_id=fa1554f1aaea9a2c58249b06e1ea48420091464d
  model_name=DeeplabV3
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

  | 输入数据 | 数据类型                    | 大小     | 数据排布格式 |
  | -------- | --------------------------- | -------- | ------------ |
  | input    | batchsize x 3 x 1024 x 2048 | RGB_FP32 | NCHW         |


- 输出数据

  | 输出数据 | 大小                        | 数据类型 | 数据排布格式 |
  | -------- | --------------------------- | -------- | ------------ |
  | output1  | batchsize x 1 x 1024 x 2048 | FLOAT64  | ND           |



# 推理环境准备\[所有版本]

- 硬件环境、开发环境和运行环境准备请参见《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/504/envdeployment/instg)》。

- 该模型需要以下依赖。

  | 配套                                                         | 版本    | 环境准备指导 |
  | ------------------------------------------------------------ | ------- | ------------ |
  | 固件与驱动                                                   | 22.0.2  | -            |
  | CANN                                                         | 5.1.RC2 | -            |
  | Python                                                       | 3.7.5   | -            |
  | PyTorch                                                      | 1.5.0   | -            |
  | 说明：Atias 300I Duo推理卡请以CANN版本选择实际固件与驱动版本 |         |              |



# 快速上手

## 获取源码

1. 获取源码。

```
git clone https://github.com/open-mmlab/mmsegmentation.git
pip3 install mmcv-full==1.3.7
cd mmsegmentation
git checkout fa1554f1aaea9a2c58249b06e1ea48420091464d
pip install -e . 
cd ..
```

2. 安装依赖

```
pip3 install -r requirements.txt
```

## 准备数据集

- 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   本模型将使用到Cityscapes验证集，请用户需自行获取数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。

   数据目录结构请参考：

   ```
   cityscapes
   ├── gtFine
      ├── test
      ├── train
      ├── val
   ├── leftImg8bit
      ├── tesy
      ├── train
      ├── val
   ```
   
- 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   执行“deeplabv3_torch_preprocess.py”脚本，完成预处理。

   ```
   python3 ./deeplabv3_torch_preprocess.py  ${DATASET_PATH} ./prep_dataset
   ```

   - ${DATASET_PATH}：原始数据验证集（.jpeg）所在路径。

   - “./prep_dataset”：输出的二进制文件（.bin）所在路径。


   每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成prep_dataset二进制文件。

   


## 模型推理

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      从源码包中获取权重文件“deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth”。
   
   2. 导出onnx文件。
   
      1. 使用
   
         “deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth”

         导出onnx文件。
   
         运行“pytorch2onnx.py”脚本。

         ```
         python3 mmsegmentation/tools/pytorch2onnx.py mmsegmentation/configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py --checkpoint ./deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth --output-file deeplabv3.onnx --shape 1024 2048
         ```

         - 参数说明：
           - --checkpoint:    pth所在路径。
           - --output-file:   导出的onnx所在路径
           - --shape：   输入数据的shape。
   
         获得“deeplabv3.onnx”文件。
   
      2. 使用onnxsimplifier简化模型。
   
         ```
         python3 -m onnxsim deeplabv3.onnx deeplabv3_sim_bs1.onnx --input-shape="1,3,1024,2048" --dynamic-input-shape
         ```
   
         - 参数说明：
           - --input-shape：   输入数据的shape。
           - --dynamic-input-shape: 表示动态batch。
         
         获得“deeplabv3_sim_bs1.onnx”文件。
         
         使用ATC工具将.onnx文件转换为.om文件，导出.onnx模型文件时需设置算子版本为11。
   
   3. 使用ATC工具将ONNX模型转OM模型。
   
      1. 配置环境变量。
   
         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```
      
         > **说明：** 
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
      
      2. 执行命令查看芯片名称（$\{chip\_name\}）。 
      
         ```
         npu-smi info
         
         +--------------------------------------------------------------------------------------------+
         | npu-smi 22.0.0                       Version: 22.0.2                                       |
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 17.6         65                0    / 0              |
         | 0       0         | 0000:5E:00.0    | 0            938  / 21534                            |
         +===================+=================+======================================================+
         
         ```
      
   3. 执行ATC命令。
      
      ```
         atc --framework=5 --model=deeplabv3_sim_bs1.onnx --output=deeplabv3_bs1 --input_format=NCHW --input_shape="input:1,3,1024,2048" --log=debug --soc_version=Ascend${chip_name}  
      ```
      
      - 参数说明：
      
        - --model：为ONNX模型文件。
        - --framework：5代表ONNX模型。
        - --output：输出的OM模型。
        - --input_format：输入数据的格式。
        - --input_shape：输入数据的shape。
        - --log：日志级别。
        - --soc_version：处理器型号。
        
      运行成功后生成模型“deeplabv3_bs1.om”文件。



2. 开始推理验证。

   使用ais-infer工具进行推理。

   a.  使用ais-infer工具进行推理。
   
   ​	ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]
   
   b.  推理模型
   
   ```
   python3 ais_infer.py  --model ${om_path} --output ./ --outfmt BIN --input ${Bin_data_path}
   ```
   
   - ${om_path}: 之前生成的OM模型（deeplabv3_bs1.om）的位置
   - ${Bin_data_path}: 数据预处理后，二进制文件所在目录（prep_dataset）
   
   
   - --model:    需要进行推理的om模型
   
   - --output:   推理结果输出路径。默认会建立日期+时间的子文件夹保存输出结果 如果指定output_dirname 将保存到output_dirname的子文件夹下。
   
   - --outfmt:   输出数据的格式，默认”BIN“，可取值“NPY”、“BIN”、“TXT”
   
   - --input:      模型需要的输入，支持bin文件和目录，若不加该参数，会自动生成都为0的数据
   
   说明： 执行ais-infer工具请选择与运行环境架构相同的命令。
   
   c.  精度验证。
   
   删除生成结果文件中的sumary.json，返回主目录，调用“deeplabv3_torch_postprocess.py”脚本与数据集groundtrue位于“cityscapes/gtFine/val”比对，可以获得mIoU数据，结果保存在“result.json”中。
   
   ```
   python3 ./deeplabv3_torch_postprocess.py --output_path=.tools/ais-bench_workload/tool/ais_infer/2022*/ --gt_path=${DATASET_PATH}/val --result_path=./result
   ```
   
   - --output_path：ais_infer生成推理结果所在路径。
   
   - --gt_path：标签数据。
   
   - --result_path：生成结果文件。

 


# 模型推理性能&精度

调用ACL接口推理计算，性能参考下列数据。

精度：

| Precesion | mAP         |
| --------- | ----------- |
| 310精度   | mIoU  79.06 |
| 310p精度  | mIoU  79.12 |

此处精度为bs1精度，bs1和最优bs精度无差别。

性能:

|      | 310   | 310p   | T4    | 310p/310 | 310p/T4 |
| ---- | ----- | ------ | ----- | -------- | ------- |
| bs1  | 2.789 | 5.3444 | 5.804 | 1.916    | 0.920   |

注：此模型不支持多batch。

bs1: 310p大于310的1.2倍，性能达标。
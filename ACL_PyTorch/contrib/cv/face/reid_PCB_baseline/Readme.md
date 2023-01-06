

# PCB模型-推理指导

## 概述

 在行人检索中使用part-level features会给description提供细粒度（fine-grained）的信息。而part-level features能发挥作用的前提是每一个part都需要有正确的定位。人体姿势估计（human pose estimation）是定位part的其中一种方法，但该论文利用的是每个part内部的上下文信息的一致性（content consistency）。

 该论文主要有两点contributions，一是设计了PCB（Part-based Convolutional Baseline）网络，其输入是一张图像，输出是包含数个均分的part-level的descriptor。二是提出了RPP（Refined part pooling）方法，由于PCB采用的是均分特征图的策略，这样就不可避免地在各个part中引入了outliers（outliers指的是其应该属于其他的part，因为在信息的一致性上更接近于其他的part），于是作者提出了RPP来重新分配这些outliers. 

-   参考论文：[Beyond Part Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/pdf/1711.09349.pdf)
-   参考实现：

```shell
url=https://github.com/syfafterzy/PCB_RPP_for_reID
branch=master
commit_id=e29cf54486427d1423277d4c793e39ac0eeff87c 
```



### 输入输出数据

- #### 输入输出数据

  - 输入数据

    | 输入数据 | 大小                      | 数据类型 | 数据排布格式 |
    | -------- | ------------------------- | -------- | ------------ |
    | input_1  | batchsize x 3 x 384 x 128 | RGB_FP32 | NCHW         |

  - 输出数据

    | 输出数据  | 大小                      | 数据类型   | 数据排布格式   |
    | -------- | -------------------------| -------- | ------------ |
    | output_1 | batchsize x 2048 x 6 x 1 | FLOAT32  | ND           |


### 推理环境准备

- 该模型需要以下插件与驱动

  | 配套                                                         | 版本                                                         | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 固件与驱动                                                   | [1.0.15](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | [5.1.RC1](https://www.hiascend.com/software/cann/commercial?version=5.1.RC1) |                                                              |
  | PyTorch                                                      | [1.5.1](https://github.com/pytorch/pytorch/tree/v1.5.1)      |                                                              |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 |                                                              |                                                              |

- 该模型需要以下依赖。

  | 依赖名称      | 版本     |
  | ------------- | -------- |
  | onnx          | >=1.9.0  |
  | torch         | >=1.13.0 |
  | torchVision   | >=0.14.0 |
  | numpy         | >=1.19.2 |
  | Pillow        | >=8.2.0  |
  | opencv-python | >=4.5.2  |
  | skl2onnx      | >= 1.8.0 |
  | h5py          | >=3.3.0  |
  | scikit-learn  | >=0.24.1 |

## 快速上手

### 获取源码

1. 源码上传到服务器任意目录（如：/home/HwHiAiUser）。

   ```
   .
   |-- README.md
   |-- PCB_pth_postprocess.py           //后处理及验证推理结果脚本，比对模型输出的分类结果和标签，给出Accuracy
   |-- PCB_pth_preprocess.py            //数据集预处理脚本
   |-- requirements.txt
   |-- pth2onnx.py                      //用于转换pth模型文件到onnx模型文件
   ```

   

2. 请用户根据依赖列表和提供的requirments.txt以及自身环境准备依赖。

   ```
   pip3 install  -r requirments.txt
   ```

   

### 准备数据集

1. 获取原始数据集。

   该模型使用[Market数据集](https://pan.baidu.com/s/1ntIi2Op?_at_=1622802619466)的19732张验证集进行测试，请用户自行获取该数据集，上传数据集到服务器任意目录（如：*/home/HwHiAiUser/dataset*）。

   

2. 数据预处理。
   设置PYTHONPATH环境变量，增加PCB开源代码仓路径。
   执行预处理脚本，生成数据集预处理后的bin文件。

   ```
   python3 PCB_pth_preprocess.py -d market -b 1 --height 384 --width 128 --data-dir /home/HwHiAiUser/dataset/Market-1501 -j 4
   ```


### 模型推理

1. 模型转换。

   本模型基于开源框架PyTorch训练的PCB进行模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。
   
   1. 获取权重文件。
       点击[Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/ATC%20PCB%28FP16%29%20from%20Pytorch%20-%20Ascend310/zh/1.1/ATC%20PCB%28FP16%29%20from%20Pytorch%20-%20Ascend310.zip)，解压后在zip包中获取PCB权重文件PCB_3_7.pt。
   
   
   2. 导出onnx文件。
   
      pth2onnx.py脚本将.pt文件转换为.onnx文件，执行如下命令在当前目录生成PCB.onnx模型文件。
      
      ```shell
      python3 pth2onnx.py -p PCB_3_7.pt -o PCB_bs1.onnx -b 1
      ```
      参数说明：
      - -p：pth权重文件。
      - -o：onnx文件。
      - -b：batch_size。
   
      使用ATC工具将.onnx文件转换为.om文件，导出.onnx模型文件时需设置算子版本为11。
   
   3. 使用ATC工具将ONNX模型转OM模型。
      
      1. 执行命令查看芯片名称（${chip_name}）。

      ${chip_name}可通过`npu-smi info`指令查看

      ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

      2. 使用atc将onnx模型转换为om模型文件。
      
      ```shell
      source /usr/local/Ascend/ascend-toolkit/set_env.sh

      atc --framework=5 --model=./PCB_bs1.onnx --output=pcb_bs1 --input_format=NCHW --input_shape="input_1:1,3,384,128" --log=debug --soc_version=Ascend${chip_name}
      ```

      参数说明：
      - --model：为ONNX模型文件。
      - --framework：5代表ONNX模型。
      - --output：输出的OM模型。
      - --input_format：输入数据的格式。
      - --input_shape：输入数据的shape。
      - --log：日志级别。
      - --soc_version：处理器型号，支持Ascend310系列。
      - --enable_small_channel：是否使能small channel的优化，使能后在channel<=4的卷积层会有性能收益。
      - --insert_op_conf=aipp.config: AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。
      注：不同的batch_size需要用不同的onnx文件
   
2. 开始推理验证。

   a.  使用ais_bench工具进行推理。
   
   参考[ais_bench工具源码地址](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)安装将工具编译后的压缩包放置在当前目录；解压工具包，安装工具压缩包中的whl文件；
   
   ```
   pip3 install aclruntime-0.01-cp37-cp37m-linux_xxx.whl
   ```
   
    b.  执行推理。
   
    ```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    python3 -m ais_bench --model ./pcb_bs1.om --input ./gallery_preproc_data_Ascend310/ --output ./result/ --outfmt BIN
    
    python3 -m ais_bench --model ./pcb_bs1.om --input ./query_preproc_data_Ascend310/ --output ./result/ --outfmt BIN
    ```
   
    -   参数说明：   
        --model：模型地址
        --input：预处理完的数据集文件夹
        --output：推理结果保存地址
        --outfmt：推理结果保存格式
   
    运行成功后会在result/xxxx_xx_xx-xx-xx-xx（时间戳）下生成推理输出的txt文件。
   
    **说明：** 
    执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见 --help命令。
   
    **因工具限制，需要把result/xxxx_xx_xx-xx-xx-xx/summary.json从结果目录中删除，或者迁移到其他目录；**
   
   c.  精度验证。
   
   调用imagenet_acc_eval.py脚本与数据集标签val_label.txt比对，可以获得Accuracy Top5数据，结果保存在result.json中。
   
   ```shell
   python3 PCB_pth_postprocess.py -q ./result/xxxx_xx_xx-xx-xx-xx（query推理时间戳） -g ./result/xxxx_xx_xx-xx-xx-xx（gallery推理时间戳） -d market --data-dir /home/HwHiAiUser/dataset/Market-1501
   ```
   
   第一个参数为生成推理结果所在路径，第二个参数为标签数据，第三个参数为生成结果文件路径，第四个参数为生成结果文件名称。



## 模型推理性能和精度



| 芯片型号    | accuracy     | accuracy     |
| ----------- | ------------ | ------------ |
| Ascend310P3 | Top1 = 92.1% | Top5 = 98.1% |

| 芯片型号    | Throughput | 310P   |
| ----------- | ---------- | ------ |
| Ascend310P3 | bs1        | 975.7  |
| Ascend310P3 | bs16       | 2031.3 |
| Ascend310P3 | bs32       | 1888.9 |
| Ascend310P3 | 最优batch  | 2031.3 |
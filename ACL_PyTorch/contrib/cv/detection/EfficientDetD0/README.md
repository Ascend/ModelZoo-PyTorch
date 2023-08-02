# EfficientDetD0模型-推理指导


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

EfficientDet是在EfficientNet基础上提出来的目标检测模型，它将EfficientNet主干网络、级联的双向特征金字塔网络(bi-directional feature pyramid network,BiFPN)和联合缩放方法结合，可以快速高效完成目标检测，且检测准确率较高，同时网络参数量较之主流检测模型大幅减少，检测速度也得到了很大提升，是目前最先进的目标检测算法之一。


- 参考实现：

  ```
  url=https://github.com/rwightman/efficientdet-pytorch.git
  commit_id=c5b694aa34900fdee6653210d856ca8320bf7d4e
  model_name=EfficientDetD0
  ```
  





## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 512 x 512 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output0  | FLOAT32  | batchsize x 810 x 64 x 64 | NCHW           |
  | output1  | FLOAT32  | batchsize x 810 x 32 x 32 | NCHW           |
  | output2  | FLOAT32  | batchsize x 810 x 16 x 16 | NCHW           |
  | output3  | FLOAT32  | batchsize x 810 x 8 x 8 | NCHW           |
  | output4  | FLOAT32  | batchsize x 810 x 4 x 4 | NCHW           |
  | output5  | FLOAT32  | batchsize x 36 x 64 x 64 | NCHW           |
  | output6  | FLOAT32  | batchsize x 36 x 32 x 32 | NCHW           |
  | output7  | FLOAT32  | batchsize x 36 x 16 x 16 | NCHW           |
  | output8  | FLOAT32  | batchsize x 36 x 8 x 8 | NCHW           |
  | output9  | FLOAT32  | batchsize x 36 x 4 x 4 | NCHW           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |





# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/rwightman/efficientdet-pytorch.git  
   cd efficientdet-pytorch  
   git reset --hard c5b694aa34900fdee6653210d856ca8320bf7d4e
   patch -p1 < ../effdet.patch
   cd ..
   ```

2. 安装依赖。

   ```
   pip install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）


   本模型支持[coco2017验证集](http://images.cocodataset.org/zips/val2017.zip)和对应的[标注文件](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)。解压并上传数据集到源码包路径下。目录结构如下：


   ```
   coco_data
   ├──val2017
   ├── annotations
      ├── instances_val2017.json
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行EfficientDetD0_preprocess.py脚本，完成预处理。

   ```
   python EfficientDetD0_preprocess.py --root=coco_data --bin-save=bin_save 
   ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       请在本项目下下载pth文件

   2. 导出onnx文件。

      1. 使用EfficientDetD0_pth2onnx.py导出onnx文件。

         运行EfficientDetD0_pth2onnx.py脚本。

         ```
         python EfficientDetD0_pth2onnx.py --checkpoint=d0.pth --out=d0.onnx 
         ```

         获得d0.onnx文件。

      2. 优化ONNX文件，安装[auto_optimizer](https://gitee.com/ascend/tools.git)模块。

         ```
	       python -m onnxsim d0.onnx d0_sim.onnx
         python modify.py --model=d0_sim.onnx --out=d0_m.onnx
         ```

         获得d0_m.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

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
			atc --framework=5 \
				--model=d0_m.onnx \
				--output=d0_bs${bs} \
				--input_format=NCHW \
				--input_shape="x.1:${bs},3,512,512" \
				--log=debug \
				--soc_version=Ascend${chip_name} \
				--precision_mode=allow_mix_precision \
				--modify_mixlist=ops_info.json  
			```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***d0_bs${bs}***</u>模型文件。

2. 开始推理验证。
   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```
		python -m ais_bench --model=d0_bs${bs}.om --input=./bin_save --batchsize=${batch_size} --output=./ --output_dirname=./result  
        ```
         - 参数说明：

           -   --model：om模型文件。
           -   --input：预处理数据
           -   --output：推理数据保存路径
           -   --output_dirname：推理数据保存目录
           -   --batchsize：batchsize大小


   3. 精度验证。

      调用脚本postprocess.py预测精度

      ```
       python postprocess.py --root=./coco_data --omfile=./result
      ```
      - 参数说明：
        - root：数据集路径
        - omfile：推理结果保存路径		


   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
        python -m ais_bench --model=d0_bs${bs}.om --loop=100 --batchsize=${batch_size}
        ```

      - 参数说明：
        - model：om模型路径
        - batchsize：batchsize大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | 数据集 | 精度 |
| --------- | ---------- | ---------- |
|    Ascend310P3       |    coco        |     33.4%       |
|    Ascend310B1      |    coco        |     33.4%       |

| Batch Size | 310P3 | 310B1 |
| ---------- | ----- | ----- |
| 1          | 124   | 71.21 |
| 4          | 260   | 64.3  |
| 8          | 256   | 64.42 |
| 16         | 250   | 64.75 |
| 32         | 241   | 64.64 |
| 64         | 238   | 54.31 |
| 最优性能   | 260   | 71.21 |
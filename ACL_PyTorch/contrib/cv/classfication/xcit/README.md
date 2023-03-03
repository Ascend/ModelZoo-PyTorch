# xcit模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  ******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Xcit是针对于图片处理设计的基于Transformer架构的神经网络。该网络基于在风格迁移领域的Gram矩阵的思想，提出使用协方差计算代替传统Transformer中的自适应模块。模型除了使用了改进的自适应模块 (attention block) 外，也使用到了卷积和全连接操作，并采用堆叠block的方式构成了最后的图形处理网络。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/xcit
  branch=master
  commit_id=82f5291f412604970c39a912586e008ec009cdca
  ```

  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | batchsize x 1000 | FLOAT32  | ND           |




# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.8.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>


## 获取源码<a name="section4622531142816"></a>

1.获取开源代码仓
 ```
    git clone https://github.com/facebookresearch/xcit.git
    cd xcit
    git checkout 82f5291f412604970c39a912586e008ec009cdca
    patch -p1 < ../xcit.patch
    cd ..
```

2. 安装依赖。
```
    pip3 install -r requirment.txt
```
注：如遇报错优先查看依赖是否安装正确的版本。

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到预处理后的“target.json”文件作为最后的比对标签。

    数据目录结构请参考：

    
        ├──ImageNet
            ├──val2017
            ├──result
                  ├──target.json
    
2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   执行xcit_preprocess.py脚本，完成预处理。

   ```
   mkdir prep_dataset
   python3.7 xcit_preprocess.py --data-path=${dataset_path} --resume=./prep_dataset
    
   ```
     - 参数说明：

        -   --data-path：原始数据验证集（.jpeg）所在路径。

        -   --resume：输出的二进制文件（.bin）所在路径。

        每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成“prep_dataset”二进制文件夹。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       从源码包中获取权重文件：[xcit_small_12_p16.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/XCIT/PTH/xcit_small_12_p16_224.pth)。

   2. 导出onnx文件。

      1. 使用“xcit_small_12_p16.pth”导出onnx文件。   
        运行“xcit_pth2onnx.py”脚本。


        ```
        mkdir onnx_models
        python3.7 xcit_pth2onnx.py --pretrained=./xcit_small_12_p16_224.pth --batch-size=16
        ```

         获得“xcit_b16.onnx”文件。

   2. 使用ATC工具将ONNX模型转OM模型。

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
         atc --framework=5 --model=onnx_models/xcit_b16.onnx  --output=xcit_b16 --input_format=NCHW --input_shape="image:16,3,224,224" --log=debug --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

              运行成功后生成“xcit_b16.om”模型文件。



2. 开始推理验证。

    a.  安装ais_bench推理工具。

   请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  


    b.  执行推理。   
   ```
        python3 -m ais_bench --model onnx_models/xcit_b16.om --input ./prep_dataset --output ./ --outfmt TXT
   ```

    -   参数说明：

        -   --model：om文件路径。
        -   --input：预处理后bin文件路径。
        -   --outfmt：以TXT格式输出。
		...

        推理后的输出默认在当前目录下。


    c.  精度验证。

    调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

    ```
    rm -rf {your_result_path}/summary.json
    python3.7 xcit_postprocess.py    --result_path {your_result_path}  --target_file ./target.json   --save_file ./result.json
    ```
      -   参数说明：
    
             -   --{your_result_path}：为生成推理结果所在路径 
    
             -   --target.json：为标签数据
    
             -   --result.json：为生成结果文件

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|     310P3      |        8          |     imagenet 2012       |     81.86%       |        443.885fps         |
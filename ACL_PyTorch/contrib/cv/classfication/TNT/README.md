# TNT模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

TNT是针对图像分类的模型，该模型将图像的patch进一步划分为sub-patch，通过visual sentences和visual words在获得全局信息的同时更好的捕捉细节信息，提升分类效果。



- 参考实现：

  ```
  url=https://github.com/huawei-noah/CV-backbones/tree/master/tnt_pytorch
  branch=master 
  commit_id=03e8cdfe92494a55ddfb11cc875ff2e1c33f91da
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 196 x 16 x 24 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 1 x 1000 | FLOAT32  | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.5.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```shell
   git clone https://github.com/huawei-noah/CV-Backbones.git
   cd CV-Backbones
   git checkout 7a0760f0b77c2e9ae585dcadfd34ff7575839ace
   patch tnt_pytorch/tnt.py ../TNT.patch
   cd ..
   cp CV-Backbones/tnt_pytorch/tnt.py ./
   ```

2. 安装依赖。

   ```shell
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到ILSVRC2012_img_val.tar验证集，请自行下载验证需要的标签文件“imagenet_labels_tnt.json”。   
   数据目录结构请参考：   
    ```
    ├──ImageNet
        ├──ILSVRC2012_img_val
        ├──imagenet_labels_tnt.json
    ```
2. 数据预处理

   数据预处理将原始数据集转换为模型输入的数据。

    使用“TNT_preprocess.py”对JPEG图片文件进行预处理并将其转换为bin文件。


      ```shell
      python3.7 TNT_preprocess.py --src-path /home/HwHiAiUser/dataset/imagenet/val --save-path ./prep_dataset
      ```
    
      --src-path：原始数据验证集（.jpeg）所在路径。
    
      --save-path：输出的二进制文件（.bin）所在路径。
    
      每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成“prep_dataset”二进制文件夹。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
      ```shell
      wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/TNT/PTH/tnt_s_81.5.pth.tar
      ```
   2. 导出onnx文件。

      1. 使用“tnt_s_81.5.pth.tar”导出onnx文件。
            运行“TNT_pth2onnx.py”脚本。

         ```
         python3.7 TNT_pth2onnx.py --pretrain_path tnt_s_81.5.pth.tar --batch_size 16
         ```
    
         获得`tnt_s_patch16_224_bs16_cast.onnx`文件。
    
      2. 优化ONNX文件。
    
         ```
         python3.7 -m onnxsim tnt_s_patch16_224_bs16_cast.onnx tnt_s_patch16_224_bs16_cast_sim.onnx --input-shape "16,196,16,24"
         ```
         获得“tnt_s_patch16_224_bs16_cast_sim.onnx”文件。
         > 注意 bs1不需要进行onnxsim优化，否则会存在精度问题  


   3. 使用ATC工具将ONNX模型转OM模型。

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
          atc --framework=5 --model=tnt_s_patch16_224_bs16_cast_sim.onnx --output=TNT_bs16 --input_format=NCHW --input_shape="inner_tokens:16,196,16,24" --log=debug --soc_version=Ascend{chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号

            运行成功后生成<u>***TNT_bs16.om***</u>模型文件。



2. 开始推理验证。

   a.  使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]


   b.  执行推理。

      ```
        python3 ais_infer.py --input ./prep_dataset --output ./  --model TNT_bs16.om --outfmt TXT --batchsize 16 --output_dirname result
      ```
    
      -   参数说明：
    
           -   model：模型路径。
           -   input：预处理文件路径。
           -   output：输出路径。
           -   outfmt：输出文件格式。
    	...
    
      推理后的输出默认在当前目录result下。
    
      >**说明：** 
      >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

   c.  精度验证。

      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。
    
      ```
      python3 TNT_postprocess.py --label_file=/home/HwHiAiUser/dataset/imagenet/imagenet_labels_tnt.json --pred_dir=./${your_result_path} > result.json
      ```
    - 参数说明
      - ${your_result_path}：为生成推理结果所在路径 
    
      - val_label.txt：为标签数据
    
      生成结果文件 result.json


   d.  性能验证。

      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
    
      ```
       python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size} 
      ```
      - 参数说明
        - --model：om模型
        - --loop：循环次数
        - --batchsize：模型batch size


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据
| 芯片型号 | Batch Size |  数据集  | 精度 |  性能  |
| :------: | :--------: | :------: | :--: | :----: |
|  310P3   |     1      | ImageNet | 81.5 | 186 |
|  310P3   |     4      | ImageNet | 81.5 | 259 |
|  310P3   |     8      | ImageNet | 81.5 | 274 |
|  310P3   |     16     | ImageNet | 81.5 | 264 |
|  310P3   |     32     | ImageNet | 81.5 | 231 |
|  310P3   |     64     | ImageNet | 81.5 | 195 |
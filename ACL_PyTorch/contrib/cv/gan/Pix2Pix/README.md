# Pix2Pix模型-推理指导


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

pix2pix是一个图像合成网络，是将GAN应用于有监督的图像到图像翻译的经典论文。其是将CGAN的思想运用在了图像翻译的领域上，学习从输入图像到输出图像之间的映射，从而得到指定的输出图像。



- 参考实现：

  ```
  url=//github.com/junyanz/pytorch-CycleGAN-and-pix2pix
  commit_id=master
  commit_id=aac572a869b6cfc7486d1d8e2846e5e34e3f0e05
  model_name=pix2pix
  ```
  



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 256 x 256 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output  | RGB_FP32  | batchsize x 3 x 256 x 256 | NCHW          |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.5.0   | [AscendPyTorch环境准备](https://gitee.com/ascend/pytorch)                                                           |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   目录结构如下：(注：可不用下载源码仓代码)
   ```
   ├─options
   ├─models
   ├─datasets
   ├─data
   ├─scripts
   ├─util
   ├─checkpoints
   ├─pix2pix_postprocess.py
   ├─pix2pix_preprocess.py
   ├─pix2pix_postprocess.py
   ├─modelzoo_level.txt
   ├─requirements.txt
   ├─LICENSE
   ├─README.md
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持[facades 验证集](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz)。用户可自行获取facades数据集上传到服务器，可放置于任意路径下，以"./datasets"目录为例。下：

   ```
   ├─datasets
      ├──facades
            ├──train
            ├──test      //验证集
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行pix2pix_preprocess.py脚本，完成预处理。

   ```
   python3 pix2pix_preprocess.py --dataroot ./datasets/facades --results_dir ./pre_bin
   ```

   - 参数说明：
      - --dataroot：数据集路径。
      - --results_dir：输出结果路径。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      下载权重文件[latest_net_G.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Pix2pix/PTH/Pix2Pix_pth.pth)，放到./checkpoints/facades_label2photo_pretrained目录下。

   2. 导出onnx文件。

      1. 使用pix2pix_pth2onnx.py导出onnx文件。

         运行pix2pix_pth2onnx.py脚本。

         ```
         python3 pix2pix_pth2onnx.py --direction BtoA --model pix2pix --checkpoints_dir ./checkpoints --name facades_label2photo_pretrained
         ```
         在./checkpoints/facades_label2photo_pretrained/路径下生成netG_onnx.onnx文件。


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
         atc --framework=5 --model=./checkpoints/facades_label2photo_pretrained/netG_onnx.onnx --output=./netG_om_bs1 --input_format=NCHW --input_shape="inputs:1,3,256,256" --log=debug --soc_version=Ascend${chip_name} 
         ```

         - 参数说明：
            - --model：为ONNX模型文件。
            - --framework：5代表ONNX模型。
            - --output：输出的OM模型。
            - --input\_format：输入数据的格式。
            - --input\_shape：输入数据的shape。
            - --log：日志级别。
            - --soc\_version：处理器型号。

         运行成功后生成netG_om_bs1.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```
        mkdir results
        python3 -m ais_bench --model ./netG_om_bs1.om --input ./pre_bin --output ./results --output_dirname bs1 --outfmt BIN --batchsize 1  --device 0
        ```

      - 参数说明：
         - --model：om文件路径。
         - --input：输入的bin文件路径。
         - --output：推理结果文件路径。
         - --outfmt：输出结果格式。
         - --batchsize：批大小。
         - --device：NPU设备编号。

        推理后的输出在推理结果文件路径的子文件路径下(./results/bs1/)。


   3. 精度验证。

      调用脚本生成om结果复原图片图片。

      ```
      python3 pix2pix_postprocess.py --bin2img_file=./result/bin2img_bs1/  --npu_bin_file=./result/bs1/
      ```

      - 参数说明：
         - --bin2img_file：推理om模型的结果复原图路径。
         - --npu_bin_file：推理om模型的结果路径。

      调用脚本生成onnx的推理结果复原图片，对onnx和om的结果进行观察对比。
      ```
      python3 test.py --dataroot ./datasets/facades/ --direction BtoA --model pix2pix --name facades_label2photo_pretrained --num_test 106
      ```
      - 参数说明：
         - --dataroot：数据集的路径。
         - --num_test：验证集的数目。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
      ```

      - 参数说明：
         - --model：om模型的路径。
         - --loop：推理的循环次数。
         - --batch_size：批大小。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| :---------: | :----------------: | :----------: | :----------: | :---------------: |
|  Ascend310P   |   1   |   facades   |  通过观察图片精度达标   |   640.766   |
|  Ascend310P   |   4   |   facades   |  通过观察图片精度达标   |   774.654   |
|  Ascend310P   |   8   |   facades   |  通过观察图片精度达标   |   931.718   |
|  Ascend310P   |   16   |   facades   |  通过观察图片精度达标   |   945.187   |
|  Ascend310P   |   32   |   facades   |  通过观察图片精度达标   |   963.043   |
|  Ascend310P   |   64   |   facades   |  通过观察图片精度达标   |   956.723   |
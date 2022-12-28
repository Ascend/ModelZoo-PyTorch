# SRGAN模型-推理指导


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

基于PyTorch实现SRGAN生成对抗网络的照片级真实感单图像超分辨率。

- 论文参考： [Ledig C, Theis L, Huszár F, et al. Photo-realistic single image super-resolution using a generative adversarial network[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 4681-4690.](https://arxiv.org/pdf/1609.04802.pdf) 


- 参考实现：

  ```
  url=https://github.com/leftthomas/SRGAN
  branch=master 
  commit_id=961e557de8eaeec636fbe1f4276f5f623b5985d4 
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 140 x 140| NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output  | FLOAT32  | batchsize x 3 x 280 x 280 | NCHW          |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码(可以不下载开源仓代码)。

   ```
   git clone https://github.com/leftthomas/SRGAN
   cd SRGAN
   git checkout master
   git reset --hard 961e557de8eaeec636fbe1f4276f5f623b5985d4
   cd ..
   ```
   目录结构如下：
   ```
   ├──pytorch_ssim
   ├──srgan_prepreprocess.py
   ├──pix2pixhd_preprocess.py
   ├──pix2pixhd_postprocess.py
   ├──pix2pixhd_pth2onnx.py
   ├──eidt_onnx.py
   ├──model.py
   ├──LICENCE
   ├──requirements.txt
   ├──README.md
   ├──modelzoo_level.txt
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持Set5验证集。用户需自行获取数据集，可将数据集放置于任意路径下以"./datasets"将包路径下。目录结构如下：

   ```
   ./datasets
      ├──Set5    //原始的只有5张数据集图片的路径。
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   依次执行脚本，完成预处理。
   1. 对图片进行缩放处理(如果下载的数据集中有缩小2倍的，可以不需要此步骤)。

      ```
      python3 srgan_prepreprocess.py --src_path=./datasets/Set5/  --result_path=./datasets/Set5_X2/
      ```
      - 参数说明：
         - --src_path：原始的只有5张数据集图片的路径。
         - --result_path：为了验证数据集效果，缩放2倍的图片路径。
   
   2. 对图片进行预处理

      ```
      python3 srgan_preprocess.py  --src_path=./datasets/Set5_X2/ ve_path=./preprocess_data
      ```
      - 参数说明：
         - --src_path：缩放2倍的图片路径。
         - --result_path：预处理的完成后的路径 。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      下载[权重文件 netG_best.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/22.1.30/ATC%20SRGAN%28FP16%29%20from%20Pytorch%20-%20Ascend310.zip)(该权重为放大2倍的权重),可放置于任意位置，以"./"为例。

   2. 导出onnx文件。

      1. 使用srgan_pth2onnx.py导出onnx文件。

         运行srgan_pth2onnx.py脚本。

         ```
         python3 srgan_pth2onnx.py --src_path=./netG_best.pth --result_path=./srgan.onnx
         ```
         - 参数说明：
            - --src_path：为ONNX模型文件。
            - --result_path：5代表ONNX模型。

         获得srgan.onnx文件。

      2. 优化ONNX文件(使用[auto-optimizer](https://gitee.com/ascend/tools/tree/master/auto-optimizer)工具对onnx模型修改)。

         ```
         python3 eidt_onnx.py --src_path=./srgan.onnx --result_path=./srgan_fix.onnx
         ```
         - 参数说明：
            - --src_path：为ONNX模型文件。
            - --result_path：5代表ONNX模型。

         获得srgan_fix.onnx文件。

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
         atc --model=./srgan_fix.onnx --framework=5 --output=./srgan_bs1 --input_format=NCHW --input_shape="lrImage:1,3,-1,-1" --dynamic_image_size="140,140;256,256;172,114;128,128;144,144" --log=info --soc_version=Ascend{chip_name} 
         ```

         - 参数说明：
            - --model：为ONNX模型文件。
            - --framework：5代表ONNX模型。
            - --output：输出的OM模型。
            - --input\_format：输入数据的格式。
            - --input\_shape：输入数据的shape。
            - --log：日志级别。
            - --soc\_version：处理器型号。
            - --dynamic\_image_size：数据集图片的高和宽

            运行成功后生成srgan_bs1.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      ```
      python3 estimate_per.py   --interpreter='python3 ${tool_path}/ais_infer.py' --om_path=./srgan_bs1.om --src_path=./preprocess_data/ --save_path=./result/bs1  --batchsize=1 --device=0  
      ```

      - 参数说明：
         - --interpreter：NPU推理工具。
         - --om_path：om模型的路径。
         - --src_path：预处理后的数据集路径。
         - --save_path：保存后的数据集路径。
         - --batchsize：批大小。
         - --device：NPU设备ID。

        推理后的输出默认在当前目录./result/bs1下。


   3. 精度验证。

      调用脚本与真实数据集图片比对，可以获得Accuracy数据，结果显示到终端界面。

      ```
      python3 srgan_om_infer.py --hr_path=./datasets/Set5/ --result_path=./result/bs1/ --save_path=./result/bs1_fer_om_res
      ```

      - 参数说明：
         - --hr_path：数据集原图片路径。 
         - --result_path：om模型的推理结果路径。
         - --save_path：om模型推理结果复原图片路径。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | :----------: | ---------- | ---------- | --------------- |
|   Ascend310P  |  1  |   Set5    |   PSNR: 33.4391;SSIM: 0.9308    |  379.036   |
|   Ascend310P  |  4  |   Set5    |   PSNR: 33.4391;SSIM: 0.9308      |  374.082   |
|   Ascend310P  |  8  |   Set5    |   PSNR: 33.4391;SSIM: 0.9308      |  380.647   |
|   Ascend310P  |  16  |   Set5    |   PSNR: 33.4391;SSIM: 0.9308      |  377.447   |
|   Ascend310P  |  32  |   Set5    |   PSNR: 33.4391;SSIM: 0.9308      |  379.026   |
|   Ascend310P  |  64  |   Set5    |   PSNR: 33.4391;SSIM: 0.9308      |  370.578   |
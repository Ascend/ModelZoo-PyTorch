# stable-diffusion模型-推理指导  


- [概述](#ZH-CN_TOPIC_0000001172161501)
   
   - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

   stable-diffusion是一种文本到图像的扩散模型，能够在给定任何文本输入的情况下生成照片逼真的图像。有关稳定扩散函数的更多信息，请查看[Stable Diffusion blog](https://huggingface.co/blog/stable_diffusion)。

- 参考实现：
  ```
  url=https://huggingface.co/runwayml/stable-diffusion-v1-5
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据  | 大小      | 数据类型                | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    |  1 x 77 | FLOAT32|  ND|


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 1 x 512 x 512 x 3 | FLOAT32  | NHWD           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表
  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 23.0.rc1（NPU驱动固件版本为6.3.T3.1.B221）  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.3.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |                                                           |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 安装依赖。
   ```
   pip3 install -r requirements.txt
   ```

2. 安装ais_bench推理工具。
   请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据README文档进行工具安装。

3. 代码修改

   执行命令：
   
   ```
   python3 stable_diffusion_clip_patch.py
   ```
   
## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型输入文本信息生成图片，无需数据集。
   
## 模型推理<a name="section741711594517"></a>

1. 模型转换。
   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。
   1. 导出ONNX模型

      执行命令：

      ```
      python3 stable_diffusion_2_onnx.py
      ```
      
      执行成功后生成onnx模型列表：  
   
         - models/clip/clip.onnx  
         - models/unet/unet.onnx
         - models/vae/vae.onnx  
   
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

      3. 执行ATC命令。此模型当前仅支持batch_size=1。

         ```
         # clip
         atc --framework=5 \
             --model=./models/clip/clip.onnx \
             --output=./models/clip/clip \
             --input_format=ND \
             --input_shape="prompt:1,77" \
             --log=error \
             --soc_version=Ascend${chip_name}
         
         # unet
         cd ./models/unet/
         atc --framework=5 \
             --model=./unet.onnx \
             --output=./unet \
             --input_format=NCHW \
             --input_shape="latent_model_input:2,4,64,64;t:1;encoder_hidden_states:2,77,768" \
             --log=error \
             --soc_version=Ascend${chip_name}
         cd ../../

         # vae
         atc --framework=5 \
             --model=./models/vae/vae.onnx \
             --output=./models/vae/vae \
             --input_format=NCHW \
             --input_shape="latents:1,4,64,64" \
             --log=error \
             --soc_version=Ascend${chip_name}
         ```
      
      参数说明：使用`atc -h`命令查看参数说明

      执行成功后生成om模型列表：  

         - models/clip/clip.om  
         - models/unet/unet.om  
         - models/vae/vae.om  
   
2. 开始推理验证。
   1. 执行推理脚本。

      ```
      python3 stable_diffusion_ascend_infer.py
      ```
      
      执行完成后在`./results`目录下生成推理图片。并在终端显示推理时间，参考如下：

      ```
      [info] infer number: 16; use time: 292.648s; average time: 18.290s
      ```
   
   2. 测试推理图片展示在`./test_results`目录下，注：每次生成的图像不同。部分测试结果如下：

      ![](./test_results/illustration_0.png)  
      Prompt: "Beautiful illustration of The ocean. in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper"

      ![](./test_results/illustration_1.png)  
      Prompt: "Beautiful illustration of Islands in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper"

      ![](./test_results/illustration_2.png)  
      Prompt: "Beautiful illustration of Seaports in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper"

   3. 性能验证。
       可使用ais_bench推理工具的纯推理模式验证om模型的性能，参考命令如下：
       ```
       python3 -m ais_bench --model=${om_model} --loop=20 --batchsize=1
       ```
       - 参数说明：使用`python3 -m ais_bench -h`命令查看。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | 模型 | Batch Size   | 数据集 | 精度 | 性能| 耗时 |
| --------- | ------ | ---------- | ---------- | ---------- | ------- | -------- |
| 310P3     | clip | 1 | -  | - | 380.604 fps | 2.627 ms |
| 310P3     | unet | 1 | -  | - | 3.191 fps | 313.285 ms |
| 310P3     | vae  | 1 | -  | - | 7.682 fps | 130.160ms |

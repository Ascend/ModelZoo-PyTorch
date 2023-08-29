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
  ```bash
   # StableDiffusion v1.5
   https://huggingface.co/runwayml/stable-diffusion-v1-5

   # StableDiffusion v2.1
   https://huggingface.co/stabilityai/stable-diffusion-2-1-base
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
   ```bash
   pip3 install -r requirements.txt
   ```

2. 代码修改

   执行命令：
   
   ```bash
   python3 stable_diffusion_clip_patch.py
   ```

3. 安装昇腾统一推理工具（AIT）

   请访问[AIT代码仓](https://gitee.com/ascend/ait/tree/master/ait#ait)，根据readme文档进行工具安装。

   安装AIT时，可只安装需要的组件：benchmark和debug，其他组件为可选安装。
   
## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型输入文本信息生成图片，无需数据集。
   
## 模型推理<a name="section741711594517"></a>

1. 模型转换。
   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   0. 获取权重（可选）

      可提前下载权重，以避免执行后面步骤时可能会出现下载失败。

      ```bash
      # 需要使用 git-lfs (https://git-lfs.com)
      git lfs install

      # v1.5
      git clone https://huggingface.co/runwayml/stable-diffusion-v1-5

      # v2.1
      git clone https://huggingface.co/stabilityai/stable-diffusion-2-1-base
      ```

   1. 导出ONNX模型

      设置模型名称或路径
      ```bash
      # v1.5 (执行时下载权重)
      model_base="runwayml/stable-diffusion-v1-5"

      # v1.5 (使用上一步下载的权重)
      model_base="./stable-diffusion-v1-5"

      # v2.1 (执行时下载权重)
      model_base="stabilityai/stable-diffusion-2-1-base"

      # v2.1 (使用上一步下载的权重)
      model_base="./stable-diffusion-2-1-base"
      ```

      注意：若条件允许，该模型可以双芯片并行的方式进行推理，从而获得更短的端到端耗时。具体指令的差异之处会在后面的步骤中单独说明，请留意。

      执行命令：

      ```bash
      python3 stable_diffusion_2_onnx.py --model ${model_base} --output_dir ./models

      # 使用并行方案
      python3 stable_diffusion_2_onnx.py --model ${model_base} --output_dir ./models --parallel
      ```

      参数说明：
      - --model：模型名称或本地模型目录的路径
      - --output_dir: ONNX模型输出目录
      
      执行成功后生成onnx模型：  
         - models/clip/clip.onnx  
         - models/unet/unet.onnx
         - models/vae/vae.onnx

   2. 优化onnx模型
      
      FlashAttention算子需区分硬件形态，支持Atlas 300I Duo和Atlas 300I A2，请根据使用的硬件形态执行命令：
      ```bash
      # 硬件形态为Duo或PRO
      python3 modify_onnx.py models/unet/unet.onnx models/unet/unet_fa.onnx Duo

      # 硬件形态为A2
      python3 modify_onnx.py models/unet/unet.onnx models/unet/unet_fa.onnx A2
      ```
   
   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```bash
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

      3. 执行ATC命令。此模型当前仅支持单张图片推理。

         ```bash
         # v1.5
         encoder_hidden_size=768

         # v2.1
         encoder_hidden_size=1024

         ```
         ```bash
         # clip
         atc --framework=5 \
             --model=./models/clip/clip.onnx \
             --output=./models/clip/clip \
             --input_format=ND \
             --log=error \
             --soc_version=Ascend${chip_name}
         
         # unet
         cd ./models/unet/

         atc --framework=5 \
             --model=./unet_fa.onnx \
             --output=./unet \
             --input_format=NCHW \
             --log=error \
             --optypelist_for_implmode="Gelu,Sigmoid" \
             --op_select_implmode=high_performance \
             --soc_version=Ascend${chip_name}

         cd ../../

         # vae

         # 硬件形态为A2
         atc --framework=5 \
             --model=./models/vae/vae.onnx \
             --output=./models/vae/vae \
             --input_format=NCHW \
             --log=error \
             --soc_version=Ascend${chip_name} \
             --customize_dtypes custom_precision.cfg
             
         # 硬件形态为Duo
         atc --framework=5 \
             --model=./models/vae/vae.onnx \
             --output=./models/vae/vae \
             --input_format=NCHW \
             --log=error \
             --soc_version=Ascend${chip_name}
         ```
      
      参数说明：
      - --model：为ONNX模型文件。
      - --output：输出的OM模型。
      - --framework：5代表ONNX模型。
      - --log：日志级别。
      - --soc_version：处理器型号。
      - --input_shape: 模型的输入shape信息。


      执行成功后生成om模型列表：  

         - models/clip/clip.om  
         - models/unet/unet.om
         - models/vae/vae.om  
   
2. 开始推理验证。

   1. 执行推理脚本。
      ```bash
      # 普通方式
      python3 stable_diffusion_ascend_infer.py \
              --model ${model_base} \
              --model_dir ./models \
              --prompt_file ./prompts.txt \
              --device 0 \
              --save_dir ./results \
              --steps 50

      # 并行方式
      python3 stable_diffusion_ascend_infer.py \
              --model ${model_base} \
              --model_dir ./models \
              --prompt_file ./prompts.txt \
              --device 0,1 \
              --save_dir ./results \
              --steps 50
      ```

      参数说明：
      - --model：模型名称或本地模型目录的路径。
      - --model_dir：存放导出模型的目录。
      - --prompt_file：输入文本文件，按行分割。
      - --save_dir：生成图片的存放目录。
      - --steps：生成图片迭代次数。
      - --device：推理设备ID；可用逗号分割传入两个设备ID，此时会使用并行方式进行推理。
      
      执行完成后在`./results`目录下生成推理图片。并在终端显示推理时间，参考如下：

      ```
      [info] infer number: 16; use time: 292.648s; average time: 18.290s
      ```
   
   3. 测试推理图片展示在`./test_results`目录下，注：每次生成的图像不同。部分测试结果如下：

      ![](./test_results/illustration_0.png)  
      Prompt: "Beautiful illustration of The ocean. in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper"

      ![](./test_results/illustration_1.png)  
      Prompt: "Beautiful illustration of Islands in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper"

      ![](./test_results/illustration_2.png)  
      Prompt: "Beautiful illustration of Seaports in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper"

## 精度验证<a name="section741711594518"></a>

   由于生成的图片存在随机性，所以精度验证将使用CLIP-score来评估图片和输入文本的相关性，分数的取值范围为[-1, 1]，越高越好。

   注意，由于要生成的图片数量较多，进行完整的精度验证需要耗费很长的时间。

   1. 下载Parti数据集

      ```bash
      wget https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv --no-check-certificate
      ```

   2. 下载Clip模型权重

      ```bash
      GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
      cd ./CLIP-ViT-H-14-laion2B-s32B-b79K

      # 用 git-lfs 下载
      git lfs pull

      # 或者访问https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/open_clip_pytorch_model.bin，将权重下载并放到这个目录下
      ```

   2. 使用推理脚本读取Parti数据集，生成图片
      ```bash
      # 普通方式
      python3 stable_diffusion_ascend_infer.py \
              --model ${model_base} \
              --model_dir ./models \
              --prompt_file ./PartiPrompts.tsv \
              --prompt_file_type parti \
              --num_images_per_prompt 4 \
              --max_num_prompts 0 \
              --device 0 \
              --save_dir ./results \
              --steps 50

      # 并行方式
      python3 stable_diffusion_ascend_infer.py \
              --model ${model_base} \
              --model_dir ./models \
              --prompt_file ./PartiPrompts.tsv \
              --prompt_file_type parti \
              --num_images_per_prompt 4 \
              --max_num_prompts 0 \
              --device 0,1 \
              --save_dir ./results \
              --steps 50
      ```

      参数说明：
      - --model：模型名称或本地模型目录的路径。
      - --model_dir：存放导出模型的目录。
      - --prompt_file：输入文本文件，按行分割。
      - --prompt_file_type: prompt文件类型，用于指定读取方式。
      - --num_images_per_prompt: 每个prompt生成的图片数量。
      - --max_num_prompts：限制prompt数量为前X个，0表示不限制。
      - --save_dir：生成图片的存放目录。
      - --steps：生成图片迭代次数。
      - --device：推理设备ID；可用逗号分割传入两个设备ID，此时会使用并行方式进行推理。

      执行完成后会在`./results`目录下生成推理图片，并且会在当前目录生成一个`image_info.json`文件，记录着图片和prompt的对应关系。

   4. 计算CLIP-score

      ```bash
      python clip_score.py \
             --device=cpu \
             --image_info="image_info.json" \
             --model_name="ViT-H-14" \
             --model_weights_path="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
      ```

      参数说明：
      - --device: 推理设备。
      - --image_info: 上一步生成的`image_info.json`文件。
      - --model_name: Clip模型名称。
      - --model_weights_path: Clip模型权重文件路径。

      执行完成后会在屏幕打印出精度计算结果。

   
# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

### StableDiffusion v2.1

| 硬件形态 | 迭代次数 | 平均耗时    |
| :------: | :--: | :--------: |
| Duo并行  |  20  |  1.322s   |
| A2     |  20  |  0.811s   | 

迭代20次的参考精度结果如下：

   ```
   average score: 0.379
   category average scores:
   [Abstract], average score: 0.294
   [Vehicles], average score: 0.379
   [Illustrations], average score: 0.378
   [Arts], average score: 0.417
   [World Knowledge], average score: 0.387
   [People], average score: 0.385
   [Animals], average score: 0.386
   [Artifacts], average score: 0.372
   [Food & Beverage], average score: 0.369
   [Produce & Plants], average score: 0.374
   [Outdoor Scenes], average score: 0.370
   [Indoor Scenes], average score: 0.387
   ```
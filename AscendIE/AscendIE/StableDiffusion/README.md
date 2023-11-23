# AIE-StableDiffusion推理指导
AIE-StableDiffusion利用AscendIE的onnx解析、build编译优化和Runtime执行等功能，适配StableDiffusion在NPU上推理，并通过PASS改图、Batch并行等优化手段，提升了推理性能。


# 概述

   stable-diffusion是一种文本到图像的扩散模型，能够在给定任何文本输入的情况下生成照片逼真的图像。有关稳定扩散函数的更多信息，请查看[Stable Diffusion blog](https://huggingface.co/blog/stable_diffusion)。

- 参考实现：
  ```bash
   # StableDiffusion v1.5
   https://huggingface.co/runwayml/stable-diffusion-v1-5

   # StableDiffusion v2.1
   https://huggingface.co/stabilityai/stable-diffusion-2-1-base
  ```

## 输入输出数据

- 输入数据

  | 输入数据  | 大小      | 数据类型                | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    |  1 x 77 | INT64|  ND|


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 1 x 3 x 512 x 512 | FLOAT32  | NCHW           |

# 推理环境准备

该模型需要使用Ascend-inference接口，安装好aie的包，并配置好环境变量，以/xxx/aie/7.0.T10为例
- source /usr/local/Ascend/ascend-toolkit/set_env.sh
- source /xxx/aie/set_env.sh
- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/xxx/aie/7.0.T10/lib
- export HOME=${model_path}
- export ASCENDIE_HOME=/xxx/aie/7.0.T10
- export TUNE_BANK_PATH=/xxx/aie/7.0.T10/aoe

# 快速上手
## 获取源码
1. 提前拉取StableDiffusion代码，并cd进目录中

2. 安装依赖。
Ascend-inference的python接口需要在python3.9环境使用
   ```bash
   pip3.9 install -r requirements.txt
   ```

   
## 准备数据集

1. 获取原始数据集。

   本模型输入文本信息生成图片，根据输入文本prompts.txt路径修改输入参数中的prompt_file
   
## 模型推理

1. 获取pipeline权重和config信息，根据路径修改输入参数的model

    1）可提前下载权重，以避免执行后面步骤时可能会出现下载失败。

      ```bash
      # 需要使用 git-lfs (https://git-lfs.com)
      git lfs install

      # v1.5
      git clone https://huggingface.co/runwayml/stable-diffusion-v1-5

      # v2.1
      git clone https://huggingface.co/stabilityai/stable-diffusion-2-1-base
      ```

    2）后续直接根据下载好的base权重路径，修改输入参数model

2. 修改模型路径

 - om模型：
  根据模型路径修改输入参数中的om_model_dir
- onnx模型：
  根据模型路径修改输入参数中的onnx_model_dir

3. 开始推理验证
  - 加载om模型，执行推理脚本
      ```bash
      # 普通方式：需要2batch的unet模型
      python3.9 aie_stable_diffusion_pipeline.py
              --model ${model_base} \
              --om_model_dir ${om_model_path} \
              --prompt_file ./prompts.txt \
              --device 0 \
              --save_dir ./results \
              --steps 50

      # 并行方式：需要1batch的unet模型
      python3.9 aie_stable_diffusion_pipeline.py
              --model ${model_base} \
              --om_model_dir ${om_model_path} \
              --prompt_file ./prompts.txt \
              --device 0,1 \
              --save_dir ./results \
              --steps 50
      ```
  - 加载onnx模型，执行推理脚本
      ```bash
      # 普通方式：需要2batch的unet模型
      python3.9 aie_stable_diffusion_pipeline.py
              --model ${model_base} \
              --onnx_model_dir ${onnx_model_path} \
              --prompt_file ./prompts.txt \
              --device 0 \
              --save_dir ./results \
              --steps 50
              --use_onnx_parser

      # 并行方式：需要1batch的unet模型
      python3.9 aie_stable_diffusion_pipeline.py
              --model ${model_base} \
              --onnx_model_dir ${onnx_model_path} \
              --prompt_file ./prompts.txt \
              --device 0,1 \
              --save_dir ./results \
              --steps 50
              --use_onnx_parser
      ```
      参数说明：
      - --model：模型名称或本地模型目录的路径。
      - --om_model_dir：存放om模型的目录。
      - --onnx_model_dir：存放onnx模型的目录。
      - --prompt_file：输入文本文件，按行分割。
      - --save_dir：生成图片的存放目录。
      - --batch_size：模型batch size。
      - --steps：生成图片迭代次数。
      - --device：推理设备ID；可用逗号分割传入两个设备ID，此时会使用**并行方式**进行推理。
      - --use_onnx_parser：是否使用onnx parser解析模型。不设置时，加载om模型。设置时，使用onnx parser加载onnx模型
      
      执行完成后在`./results`目录下生成推理图片。并在终端显示推理时间，参考如下：

      ```
      [info] infer number: 16; use time: 86.592s; average time: 5.412s
      ```

## 精度验证

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

   3. 使用推理脚本读取Parti数据集，生成图片
      ```bash
      # 普通方式
      python3 aie_stable_diffusion_pipeline.py \
              --model ${model_base} \
              --onnx_model_dir ${onnx_model_path} \
              --prompt_file ./PartiPrompts.tsv \
              --prompt_file_type parti \
              --num_images_per_prompt 4 \
              --device 0 \
              --save_dir ./results \
              --batch_size ${bs} \
              --steps 50

      # 并行方式
      python3 aie_stable_diffusion_pipeline.py \
              --model ${model_base} \
              --onnx_model_dir ${onnx_model_path} \
              --prompt_file ./PartiPrompts.tsv \
              --prompt_file_type parti \
              --num_images_per_prompt 4 \
              --device 0,1 \
              --save_dir ./results \
              --batch_size ${bs} \
              --steps 50
      ```

      参数说明：
      - --model：模型名称或本地模型目录的路径。
      - --onnx_model_dir：存放onnx模型的目录。
      - --prompt_file：输入文本文件，按行分割。
      - --prompt_file_type: prompt文件类型，用于指定读取方式。
      - --num_images_per_prompt: 每个prompt生成的图片数量。
      - --save_dir：生成图片的存放目录。
      - --batch_size：模型batch size。
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

## 动态分档

   支持生成不同分辨率的图片，目前支持三种分辨率，高和宽分别是 512,512/576,768/768,576。

   pipeline权重和config信息可参考模型推理章节获取。

   1. 准备动态onnx模型。动态分辨率只与unet和vae模型有关，需要导出H、W维是动态的unet与vae模型。

   2. 修改模型路径
   - om模型：
   根据模型路径修改输入参数中的om_model_dir
   - onnx模型：
   根据模型路径修改输入参数中的onnx_model_dir

   3. 开始推理验证
  - 加载om模型，执行推理脚本
      ```bash
      # 普通方式: 需要2batch的unet动态模型，1batch的vae动态模型
      python3.9 pipeline_dynamic.py
              --model ${model_base} \
              --om_model_dir ${om_model_path} \
              --prompt_file ./prompts.txt \
              --device 0 \
              --save_dir ./results \
              --steps 50 \
              --use_dynamic_dims \
              --resolution 576,768

      # 并行方式: 需要1batch的unet动态模型，1batch的vae动态模型
      python3.9 pipeline_dynamic.py
              --model ${model_base} \
              --om_model_dir ${om_model_path} \
              --prompt_file ./prompts.txt \
              --device 0,1 \
              --save_dir ./results \
              --steps 50 \
              --use_dynamic_dims \
              --resolution 576,768
      ```
  - 加载onnx模型，执行推理脚本
      ```bash
      # 普通方式:需要2batch的unet动态模型，1batch的vae动态模型
      python3.9 pipeline_dynamic.py
              --model ${model_base} \
              --onnx_model_dir ${onnx_model_path} \
              --prompt_file ./prompts.txt \
              --device 0 \
              --save_dir ./results \
              --steps 50 \
              --use_onnx_parser \
              --use_dynamic_dims \
              --resolution 576,768

      # 并行方式:需要1batch的unet动态模型，1batch的vae动态模型
      python3.9 pipeline_dynamic.py
              --model ${model_base} \
              --onnx_model_dir ${onnx_model_path} \
              --prompt_file ./prompts.txt \
              --device 0,1 \
              --save_dir ./results \
              --steps 50 \
              --use_onnx_parser \
              --use_dynamic_dims \
              --resolution 576,768
      ```
      参数说明：
      - --model：模型名称或本地模型目录的路径。
      - --om_model_dir：存放om模型的目录。
      - --onnx_model_dir：存放onnx模型的目录。
      - --prompt_file：输入文本文件，按行分割。
      - --save_dir：生成图片的存放目录。
      - --batch_size：模型batch size。
      - --steps：生成图片迭代次数。
      - --device：推理设备ID；可用逗号分割传入两个设备ID，此时会使用**并行方式**进行推理。
      - --use_onnx_parser：是否使用onnx parser解析模型。不设置时，加载om模型。设置时，使用onnx parser加载onnx模型
      - --use_dynamic_dims：是否使用动态挡位。不设置时，需要将onnx路径设置为静态模型的路径。设置时，需要将onnx路径设置为动态模型的路径，使用动态挡位，加载动态模型。
      - --resolution：图片分辨率；可用逗号分割图片的高和宽，目前仅支持三种分辨率**512,512/576,768/768,576**。
      
      执行完成后在`./results`目录下生成推理图片。
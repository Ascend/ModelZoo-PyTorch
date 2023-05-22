# StableDiffusion-2.1 for PyTorch

-   [概述]
-   [准备环境]
-   [图像生成任务推理]
-   [版本说明]

# 概述

## 简述
StableDiffusion 是 StabilityAI公司于2022年提出的图片生成的预训练模型，论文和代码均已开源，下游任务包括文生图、图生图、图片压缩等等

- 参考实现：

  ```
  url=https://github.com/Stability-AI/stablediffusion 
  commit_id=535d370de05c0e55a6bcbdd32933e549e125b946
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/diffusion/stablediffusion-2.1
  ```

# 准备环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.11 | torchvision==0.12.0; transformers==4.19.2  |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  
  pip install -r requirements.txt
  ```


# 图像生成任务推理
  
  先source环境变量：
  ```
  source ./test/env_npu.sh
  ```


  执行图像生成脚本：
  ```
  python scripts/txt2img.py \
        --prompt "a professional photograph of an astronaut riding a horse" \
        --ckpt /xxx/xxx.ckpt \
        --config configs/stable-diffusion/v2-inference-v.yaml \
        --H 768 \
        --W 768 \
        --device_id 4 \
        --precision full \
        --n_samples 1 
        --n_iter 1 
        --dpm 
        --steps 15 

  # 注：生成的图像在outputs目录下 
  
  ```

  或者使用shell脚本启动：

  ```
  bash ./test/run_infer_full_1p.sh --ckpt_path=/data/xxx/ --n_samples=1 --device_id=0 # 单卡推理
  ```

  
  参数说明如下：
  
  ```
  
   --ckpt_path 模型文件目录，需要指定到具体的ckpt文件

   --n_samples 每次生成图片的batch数

   --n_iter 每个prompt生成的图片数，每个prompt生成的图片总数为：（n_samples * n_iter）

   --devive 除`cpu`外，默认为`cuda`

   --device_id 运行的设备ID
  
  ```

  

# 版本说明

## 变更


## FAQ

1. 该模型当前只支持在线推理，不支持训练。

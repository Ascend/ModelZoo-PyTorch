# StableDiffusion-2.1 for PyTorch

-   [概述](#概述)
-   [准备环境](#准备环境)
-   [图像生成任务推理](#图像生成任务推理)
-   [版本说明](#版本说明)

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

  | Torch_Version |              三方库依赖版本               |
  | :-----------: | :---------------------------------------: |
  | PyTorch 1.11  | torchvision==0.12.0; transformers==4.19.2 |
  | PyTorch 2.1   | torchvision==0.16.0; transformers==4.19.2 |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  
  pip install -r requirements.txt
  ```


# 图像生成任务推理
- 下载预训练的[ckpt文件](https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main)

- 先source环境变量：

  ```shell
  source ./test/env_npu.sh
  ```

-   执行图像生成脚本：

  ```shell
   python scripts/txt2img.py \
          --prompt "a professional photograph of an astronaut riding a horse" \
          --ckpt /xxx/xxx.ckpt \
          --config configs/stable-diffusion/v2-inference-v.yaml \
          --H 768 \
          --W 768 \
          --device_id 4 \
          --precision full \
          --bf16 \
          --n_samples 1 \
          --n_iter 1 \
          --dpm \
          --steps 15 
  
  # 或者使用shell脚本启动：
  bash ./test/run_infer_full_1p.sh  #单卡推理，请将脚本里的ckpt权重路径更换成实际的路径
  
  # 注：生成的图像在outputs目录下 
  ```

-  参数说明如下：  

  ```
  --ckpt_path #模型文件目录，需要指定到具体的ckpt文件
  
  --n_samples #每次生成图片的batch数
  
  --n_iter #每个prompt生成的图片数，每个prompt生成的图片总数为：（n_samples * n_iter）
  
  --devive #除`cpu`外，默认为`cuda`
  
  --device_id #运行的设备ID
  ```

  
# 公网地址说明

代码涉及公网地址参考 public_address_statement.md 
# 版本说明

## 变更


## FAQ

1. 该模型当前只支持在线推理，不支持训练。

2. `requests.exceptions.SSLError:HttpSConnectionPool(host='huggingface.co', port=443)`错误

   可以修改当前环境的`requests`包下的sessions.py文件的684行，增加`kwargs["verify"] = False`，一般路径在`conda_path/envs/conda_name/lib/python*/site-packages/requests/sessions.py`，查看`requests`包的所在路径可以通过`pip show requests`查看



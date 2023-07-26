# InstructBLIP for PyTorch

-   [概述](#概述)
-   [准备环境](#准备环境)
-   [视觉问答(VQA)任务推理](#视觉问答(VQA)任务推理)
-   [版本说明](#版本说明)
-   [公网地址说明](#公网地址说明)



# 概述

## 简述

InstructBLIP是从BLIP2模型微调而来的模型。InstructBLIP模型更擅长「看」、「推理」和「说」，即能够对复杂图像进行理解、推理、描述，还支持多轮对话等。最重要的是，InstructBLIP在多个任务上实现了最先进的性能，甚至在图片解释和推理上表现优于GPT4。

- 参考实现：

  ```
  url=https://github.com/salesforce/LAVIS
  commit_id=f3212e7d57bf3bb6635f8ae0461f167103cee2b4
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/mlm
  ```


# 准备环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.11 | torchvision==0.12.0 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  ```
  pip install -r requirements.txt
  python3 setup.py install
  ```

# 视觉问答(VQA)任务推理

1. 获取Vicuna-7B权重：

   通过以下链接获取Hugging Face格式的原始LLaMA权重(https://huggingface.co/docs/transformers/main/model_doc/llama)
   
   通过以下链接获取应用了delta的vicuna权重。(https://huggingface.co/lmsys/vicuna-7b-delta-v1.1/tree/main)
   
   在模型根目录下使用以下转换脚本将delta权重应用于原始Llama权重，需要30GB cpu内存，如果cpu内存小于32GB，大于16GB，可在命令后加--low-cpu-mem，转换后的权重放在模型根目录下
   ```
   python3 -m fastchat.model.apply_delta \
    --base-model-path /path/to/llama-7b \  
    --target-model-path vicuna-7b \
    --delta-path lmsys/vicuna-7b-delta-v1.1
   ```
   参数说明如下：
   ```
   base-model-path：原始Llama权重路径
   target-model-path：转换后的vicuna权重存放路径
   delta-path：应用了delta的vicuna权重路径
   ```

2. 获取instruct_blip_vicuna7b_trimmed权重放在模型根目录下(https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth)

3. 获取bert-base-uncased放在模型根目录下(https://huggingface.co/bert-base-uncased/tree/main)

4. source环境变量：

   ```shell
   source ./test/env_npu.sh
   ```

5. 在模型根目录下运行推理脚本。
  
   ```
   python run_scripts/instructblip/infer/infer.py --img_path xxx --prompt xxx
   
   # 或者使用shell脚本启动：
   bash ./test/run_infer_full_1p.sh  #单卡推理，请将脚本里的img_path更换成实际的路径,prompt更换为需要推理的prompt
   ```

   模型推理脚本参数说明如下。
   
   ```
   公共参数：
   --img_path                             //待推理图片路径
   --prompt                               //推理prompt
   ```



# 版本说明

## 变更

2023.07.17：首次发布。

## FAQ

1. 目前只支持InstructBLIP的推理功能。

2. `requests.exceptions.SSLError:HttpSConnectionPool(host='huggingface.co', port=443)`错误

   可以修改当前环境的`requests`包下的sessions.py文件的684行，增加`kwargs["verify"] = False`，一般路径在`conda_path/envs/conda_name/lib/python*/site-packages/requests/sessions.py`，查看`requests`包的所在路径可以通过`pip show requests`查看

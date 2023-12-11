# Diffusers for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

扩散模型 (Diffusion Models) 是一种生成模型，可生成各种各样的高分辨率图像。Diffusers 是Huggingface发布的模型套件，包含基于扩散模型的多种下游任务训练与推理，可用于生成图像、音频，甚至分子的 3D 结构。

- 参考实现：

  ```
  url=https://github.com/huggingface/diffusers
  commit_id=29f15673ed5c14e4843d7c837890910207f72129
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/diffusion/
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 2.1 | diffusers==0.21.0 accelerate==0.24.0|
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》搭建torch环境。
  
- 安装依赖。

  在模型根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```shell
  pip install -e .                    # 安装diffusers
  cd examples/text_to_image/           # 根据下游任务安装对应依赖
  pip install -r requirements_sdxl.txt  
  ```
  





  


## 准备数据集

1. 联网情况下，数据集会自动下载。

2. 无网络情况下，用户需自行获取pokemon数据集，并在shell启动脚本中将`dataset_name`参数，设置为本地数据集的绝对路径，填写一级目录。

   数据结构如下：

   ```
   $dataset
   ├── README.MD
   ├── data
   	├── dataset_infos.json
   	└── train-0001.parquet
   └── dataset_infos.json
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

   

## 获取预训练模型

1. 联网情况下，预训练模型会自动下载。

2. 无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

   ```
   stabilityai/stable-diffusion-xl-base-1.0
   madebyollin/sdxl-vae-fp16-fix
   ```

3. 获取对应的预训练模型后，在shell启动脚本中将`model_name`参数，设置为本地预训练模型路径，填写一级目录。

# 开始训练

## 训练模型

本节以文生图下游任务为例，展示模型训练方法。

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     ```shell
     bash test/train_1p_text_to_image_sdxl_lora.sh # 单卡精度，SDXL_lora，fp32
     bash test/train_1p_text_to_image_sdxl_lora.sh --max_train_steps=200 # 单卡性能，SDXL_lora，fp32
     ```
     
   - 单机8卡训练
   
     ```shell
     bash test/train_8p_text_to_image_sdxl_lora.sh  # 8卡精度训练 ，SDXL_lora，fp32
     bash test/train_8p_text_to_image_sdxl_lora.sh --max_train_steps=200 # 8卡性能训练 ，SDXL_lora，fp32
     ```
     
   
   模型训练python训练脚本参数说明如下。
   
   ```shell
   train_text_to_image_lora_sdxl.py：
   --max_train_steps                   //训练步数
   --pretrained_model_name_or_path     //预训练模型名称或者地址
   --dataset_name                      //加载数据集的方式，从官网或者本地cache中读取数据
   --vae_name                          //预训练vae模型名称或者地址
   --dataset_config_name               //数据集配置     
   --train_data_dir                    //符合huggingface结构的训练数据集
   --train_batch_size                  //设置batch_size
   --image_column                      //图片所在列
   --caption_column                    //图片caption所在列
   --max_train_samples                 //最大训练样本数
   --validation_prompts                //验证提示词
   --output_dir                        //输出路径
   --resolution                        //分辨率
   --num_train_epochs                  //训练epoch数
   --gradient_accumulation_steps       //梯度累计步数
   --mixed_precision                   //精度模式
   --num_train_epochs                  //训练回合数

   ```
   
   训练完成后，权重文件保存在`test/output`路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | sd版本 | FPS  | batch_size | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :------: | :-----------: | :-----------: |
| 8p-竞品A | xl | 7.41 | 1 | fp32 |      2.1      |
|  8p-NPU-910  | xl | 6.48 | 1 | fp32 |      2.1      |



# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 版本说明

## 变更

2023.11.02：首次发布。

## FAQ


   

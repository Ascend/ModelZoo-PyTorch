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
  commit_id=1c0f6bb2cfbeacb2b6ac902db7d2f5fce4d342f1
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
  | PyTorch 1.11 | diffusers==0.18.1 accelerate==0.20.3 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》搭建torch环境。
  
- 安装依赖。

  在模型根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```shell
  pip install -e .                    # 安装diffusers
  cd examples/text_to_image/           # 根据下游任务安装对应依赖
  pip install -r requirements.txt  
  ```
  
- 修改三方库代码：

  ```shell
  # 1. 在${python路径}/python3.7/site-packages/accelerate/accelerator.py中找到：
  self.scaler = torch.cuda.amp.GradScaler(**kwargs)
  修改为：
  import torch_npu
  self.scaler = torch.npu.amp.GradScaler(**kwargs)
  
  # 2. 修改${python路径}/python3.7/site-packages/accelerate/utils/dataclasses.py，给类GradScalerKwargs添加属性：
  dynamic: bool = True
  ```

- 卸载safetensors（如有）：

  ```shell
  pip uninstall safetensors
  ```

- 安装Megatron-LM，[参考链接](http://gitee.com/ascend/Megatron-LM)。

  


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
   CompVis/stable-diffusion-v1-4
   runwayml/stable-diffusion-v1-5
   stabilityai/stable-diffusion-2
   stabilityai/stable-diffusion-2-1
   ```

3. 获取对应的预训练模型后，在shell启动脚本中将`model_name`参数，设置为本地预训练模型路径，填写一级目录。

# 开始训练

## 训练模型

本节以文生图下游任务为例，展示模型训练方法，其余下游任务controlnet、dreambooth、textual inversion等需要替换启动脚本。

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     ```shell
     bash test/train_full_1p_text_to_image.sh  # 单卡精度
     ```
     
   - 单机8卡训练
   
     ```shell
     bash test/train_full_8p_text_to_image_sd1-5_fp16.sh  # 8卡精度，SD1.5，fp16
     bash test/train_full_8p_text_to_image_sd1-5_fp32.sh  # 8卡精度，SD1.5，fp32
     bash test/train_full_8p_text_to_image_sd2-1_fp32.sh  # 8卡精度，SD2.1，fp32
     bash test/train_full_8p_text_to_image_sd2-1_fp16.sh  # 8卡精度，SD2.1，fp16
     bash test/train_full_8p_text_to_image_sd2-1_fp16_fa.sh # 8卡精度，SD2.1，fp16+FA
     bash test/train_performance_8p_text_to_image_sd1-5_fp16.sh # 8卡性能，SD1.5，fp16
     bash test/train_performance_8p_text_to_image_sd1-5_fp32.sh # 8卡性能，SD1.5，fp32
     bash test/train_performance_8p_text_to_image_sd2-1_fp32.sh # 8卡性能，SD2.1，fp32
     ```
     
   - 跑fp16不带FA时，由于attention模块中bmm算子走fp16会有溢出，需修改以下代码，使bmm算子走fp32计算：（适配FA后不走bmm算子，因此不会溢出）
   
     将src/diffusers/models/attention_processor.py中get_attention_scores方法里的
   
     ```
     if attention_mask is None:
         attention_scores = torch.mul(self.scale, torch.bmm(query, key.transpose(-1, -2)))
     else:
         beta = 1
         attention_scores = torch.add(torch.mul(beta, attention_mask),
                                     torch.mul(self.scale, torch.bmm(query, key.transpose(-1, -2))))
     ```
   
     修改为：
   
     ```
     with torch.cuda.amp.autocast(enabled=False):
         if attention_mask is None:
             attention_scores = torch.mul(self.scale, torch.bmm(query, key.transpose(-1, -2)))
         else:
             beta = 1
             attention_scores = torch.add(torch.mul(beta, attention_mask),
                                         torch.mul(self.scale, torch.bmm(query, key.transpose(-1, -2))))
     ```
   
   
   模型训练python训练脚本参数说明如下。
   
   ```shell
   train_text_to_image.py：
   --max_train_steps                   //训练步数
   --pretrained_model_name_or_path     //预训练模型名称或者地址
   --dataset_name                      //加载数据集的方式一，从官网或者本地cache中读取数据
   --local_data_dir                    //加载数据集的方式二，从本地磁盘中读取数据（官方数据类型经过dick处理）
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
   --use_megatron_npu_adamW            //使用megatron优化器
   --use_npu_fuse_adamW                //使用NPU融合优化器
   --use_clip_grad_norm_fused          //使用融合CLIP操作（必须搭配NPU融合优化器使用）
   --enable_npu_flash_attention        //使能Flash Attention大kernel融合算子（目前仅fp16支持FA）
   --enable_pin_memory                 //使能数据加载时的pin_memory
   --enable_persistent_workers         //使能数据加载时的persistent_workers
   ```
   
   训练完成后，权重文件保存在`test/output`路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | sd版本 | FPS  | batch_size | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :------: | :-----------: | :-----------: |
| 8p-竞品A | 2.1 | 10 | 4 | fp32 |      1.13      |
|  8p-NPU-910  | 2.1 | 15 | 4 | fp32 |      1.11      |
| 8p-竞品A | 2.1 | 22 | 4 | fp16 |      1.13      |
|  8p-NPU-910  | 2.1 | 16 | 4 | fp16 |      1.11      |
| 8p-竞品A+FA | 2.1 | 65.5 | 24 | fp16 | 1.13 |
| 8p-NPU-910+FA | 2.1 | 51.3 | 24 | fp16 | 1.11 |

**表3** 训练支持场景

| SD版本/AMP_Type |                fp16                 | fp16+FA                                                      |               fp32                |
| :-------------: | :---------------------------------: | ------------------------------------------------------------ | :-------------------------------: |
|      SD1.5      | 支持，需设置--mixd_precision="fp16" | 不支持                                                       | 支持，需设置--mixd_precision="no" |
|      SD2.1      | 支持，需设置--mixd_precision="fp16" | 支持，需设置--mixd_precision="fp16"，和--enable_npu_flash_attention | 支持，需设置--mixd_precision="no" |


# 推理

## 文生图
参考实现：(https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation)

### 预训练模型准备

联网情况下，预训练模型会自动下载。无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

```shell 
CompVis/ldm-text2im-large-256
```

获得对应的预训练模型后，修改以下代码中的地址为本地地址即可

```python 
generator = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256",torch_dtype=torch.float16)
```

### 运行在线推理

```shell 
python test_infer/text-to-image.py
```

修改prompt等操作需要对代码进行修改

## 文本指导图生图

参考实现：(https://huggingface.co/docs/diffusers/using-diffusers/img2img)

### 预训练模型准备

联网情况下，预训练模型会自动下载。无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

```shell 
nitrosocke/Ghibli-Diffusion
```

获得对应的预训练模型后，修改以下代码中的地址为本地地址即可

```python 
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("nitrosocke/Ghibli-Diffusion").to(device)
```

### 运行在线推理

修改test_infer/text-guide-img-to-img.py中url为本地图片地址

```shell 
python test_infer/text-guide-img-to-img.py
```

修改prompt等操作需要对代码进行修改

## 文本指导图像修复

参考实现：(https://huggingface.co/docs/diffusers/using-diffusers/inpaint)

### 预训练模型准备

联网情况下，预训练模型会自动下载。无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

```shell 
runwayml/stable-diffusion-inpainting
```
获得对应的预训练模型后，修改以下代码中的地址为本地地址即可

```python 
pipeline = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
```
### 运行在线推理

```shell 
python test_infer/text-guide-image-inpainting.py
```

修改prompt等操作需要对代码进行修改

## 纹理反转

参考实现：(https://huggingface.co/docs/diffusers/using-diffusers/textual_inversion_inference)

### 预训练模型准备

联网情况下，预训练模型会自动下载。无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

```shell 
runwayml/stable-diffusion-v1-5
```

获得对应的预训练模型后，修改以下代码中的地址为本地地址即可

```python 
pretrained_model_name = "runwayml/stable-diffusion-v1-5"
```

### 数据集准备

联网情况下，数据会自动下载。无网络时，用户可访问huggingface官网自行下载，文件namespace如下。用户也可参考该数据集自行准备数据集：

```shell 
sd-concepts-library/cat-toy
```

获得对应的预训练模型后，修改以下代码中的地址为本地地址即可

```python 
repo_id = "sd-concepts-library/cat-toy"
```

### 运行在线推理


```shell 
python  test_infer/textual-inversion.py
```

修改prompt等操作需要对代码进行修改

## 文本指导图像深度生成

参考实现：(https://huggingface.co/docs/diffusers/using-diffusers/depth2img)

### 预训练模型准备

联网情况下，预训练模型会自动下载。无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

```shell 
stabilityai/stable-diffusion-2-depth
```

获得对应的预训练模型后，修改以下代码中的地址为本地地址即可

```python 
pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth").to("npu")
```

### 运行在线推理

```shell 
python test_infer/text-guide-depth-to-image.py
```

修改prompt等操作需要对代码进行修改

## 无条件图像生成

参考实现：(https://huggingface.co/docs/diffusers/using-diffusers/unconditional_image_generation)

### 预训练模型准备

联网情况下，预训练模型会自动下载。无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

```shell 
anton-l/ddpm-butterflies-128
```

获得对应的预训练模型后，修改以下代码中的地址为本地地址即可

```python 
generator = DiffusionPipeline.from_pretrained("anton-l/ddpm-butterflies-128")
```

### 运行在线推理

```shell 
python test_infer/unconditional-image-generation.py
```

修改prompt等操作需要对代码进行修改

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 版本说明

## 变更

2023.06.20：首次发布。

## FAQ

1. 使用训练后的权重推理，如果出现NSFW检测，需要在推理前，关闭模型中的NSFW检测，具体做法：

   ```
   1) 找到模型文件model_index.json，将其中的requires_safety_checker参数设置为false
   2) 删除safetychecker参数及其对应的参数值
   ```

   

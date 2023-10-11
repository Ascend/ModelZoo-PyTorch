# VisualGLM for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

VisualGLM-6B 是一个开源的，支持图像、中文和英文的多模态对话语言模型，语言模型基于 ChatGLM-6B，具有 62 亿参数；图像部分通过训练 BLIP2-Qformer 构建起视觉模型与语言模型的桥梁，整体模型共78亿参数。

VisualGLM-6B 依靠来自于 CogView 数据集的30M高质量中文图文对，与300M经过筛选的英文图文对进行预训练，中英文权重相同。该训练方式较好地将视觉信息对齐到ChatGLM的语义空间；之后的微调阶段，模型在长视觉问答数据上训练，以生成符合人类偏好的答案。

- 参考实现：

  ```
  url=https://github.com/THUDM/VisualGLM-6B
  commit_id=7fd95c2075efa60867c0ea16d061f55878d3d282
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/mlm
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.11 | deepspeed 0.9.2 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。
  建议使用conda或者镜像环境，使用python3.7

  1. 基本环境
  在模型源码包根目录下执行命令，安装模型需要的依赖。
  ```
  conda create -n env_name python=3.7
  pip install -r requirements.txt

  ```
  2. 安装deepspeed
  需要安装指定版本GCC，版本为GCC 7.5.0

  ```
  pip install deepspeed==0.9.2
  git clone git@gitee.com:ascend/DeepSpeed.git -b v0.9.2 deepspeed_npu
  cd deepspeed_npu
  pip install -e .

  ```

  3. 适配迁移代码
  首先通过
  ```
  pip show SwissArmyTransformer
  ```
  找到sat的路径，设为path
  在python安装路径，即path/sat下，找出chatglm_model.py、glm130B_model.py和rotary_embeddgins.py文件，并用code_for_change下同名文件进行替换，三个文件在sat的具体位置为：
  sat/model/official/chatglm_model.py
  sat/model/official/glm130B_model.py
  sat/model/position_embedding/rotary_embeddings.py

## 准备预训练模型

- 新建文件夹 “glm”，分别下载visualglm和chatglm模型，目录如下所示。
```
  ├── glm
      ├──visualglm
         ├──1
         ├──latest
         ├──model_config.json
      ├──chatglm 
```
修改model_config.json文件第二行"toeknizer_type"，将其地址设为"path/glm/chatglm"，即chatglm目录所在位置

## 微调数据集

1. 原仓数据集：
   解压主文件目录下的fewshot-data.zip，共有中文标注的图片数据20张

2. 从官网下载COCO2017数据集，放到COCO2017目录下

   数据集目录结构参考如下所示。
   
   ```
   ├── COCO
      ├──train2017   
      ├──annotations
         ├──captions_train2017.json
   ```
   其中，train2017为图片所在位置，annotations/captions_train2017.json是对应标注所在位置，COCO请大写，代码中通过关键词"COCO"来判定是否使用COCO2017，没有该关键词则使用原仓自带的数据集。
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

# 开始训练

## 微调任务

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash test/train_full_1p.sh  COCO2017数据路径  预训练模型路径       # 单卡训练
     
     bash test/train_performance_1p.sh  COCO2017数据路径  预训练模型路径       # 单卡性能
     ```
     
   - 单机8卡训练

     启动8卡训练。
     ```
     bash test/train_full_8p.sh  COCO2017数据路径  预训练模型路径       # 8卡训练
     
     bash test/train_performance_8p.sh  COCO2017数据路径  预训练模型路径       # 8卡性能
     ```
     
     
   
   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。
   
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 推理任务
```
python3 cli_demo.py \
--english \
--from_pretrained test/output/${ASCEND_DEVICE_ID}/checkpoints/ \
--chatglm_path glm/chatglm \
--prompt_en "What's in the image?"
```
--english使用英文输入，--from_pretrained是训练保存的checkpoints，chatglm_path加载预训练chatglm地址，在glm/chatlglm路径下。

# 训练结果展示

## 精度结果
通过对比Loss曲线，与竞品Loss平均相对误差为0.4%。





## 性能结果
通过单步时间对比，平均性能为0.84 * 竞品。



# 版本说明

## 变更

2023.05.16：首次发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 ```./public_address_statement.md```
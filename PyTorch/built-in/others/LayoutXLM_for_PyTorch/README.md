# LayoutXLM for PyTorch

-   [概述](#1)
-   [准备训练环境](#2)
-   [开始训练](#3)
-   [训练结果展示](#4)
-   [版本说明](#5)

# 概述

## 简述

LayoutXLM是用于多语言文档理解的多模式预训练模型，旨在消除视觉丰富的文档理解的语言障碍。实验结果表明，它在XFUND数据集上显著优于现有的SOTA跨语言预训练模型。

- 参考实现：
  
  ```bash
    url=https://github.com/microsoft/unilm/tree/master/layoutxlm
    commit_id=ec8c2624c8832aa4ca89005fd20e85a211f20a8f
  ```

- 适配昇腾 AI 处理器的实现：

  ```bash
    url=https://gitee.com/ascend/ModelZoo-PyTorch.git
    code_path=PyTorch/built-in/others
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  ****表 1**** 版本支持表

  | Torch_Version     | 三方库依赖版本 
  | --------          |:---------:
  | PyTorch 1.8       | transformers==4.5.1; detectron2==0.3; seqeval==1.2.2; datasets==2.7.1; packaging==21.0

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。


  ```bash
  # 安装detectron2
  git clone https://github.com/facebookresearch/detectron2.git -b v0.3
  python -m pip install -e detectron2
  ```

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。

  ```bash
  pip install -r requirements.txt
  pip install -e .
  ```
  
## 准备数据集
- 在有网络的情况下，模型训练需要的数据集会在训练开始之前由训练脚本自动下载，无需准备数据集。

- 在没有网络的情况下，用户也可以自行下载xfun-zh数据集，并且移动到 */root/.cache/huggingface/datasets/xfun/xfun.zh/0.0.0/affa7f771c23899f4ea7b3b522db75470abe55a08e8cf96de60597348837b9ed* 路径下，数据集目录参考结构如下所示：

   ```
   0.0.0
   |——————affa7f771c23899f4ea7b3b522db75470abe55a08e8cf96de60597348837b9ed
   |        └——————dataset_info.json
   |        └——————xfun-train.arrow
   |        └——————xfun-validation.arrow   
   ```
## 获取预训练模型
本文使用layoutxlm-base预训练模型

- 用户在有网络的情况下，预训练模型会在训练开始之前由训练脚本自动下载。

- 在没有网络的情况下，需要用户自行下载预训练模型layoutxlm-base，将获取的预训练模型上传至 */root/.cache/huggingface/transformers/8680422ada73a219d10ded26623c015f44a909e815304488fd43ed77efe03e27.89dd075ca1cb2d01705599c66b2867ecedda5e15879080b08735d2dbdf3631b7* 目录下。


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```bash
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练、单机8卡训练。

   + 单机单卡训练

     启动单卡训练：

     ```bash
     bash test/train_full_1p_re.sh   # RE任务
     bash test/train_performance_1p_re.sh   # RE性能任务
       
     bash test/train_full_1p_ser.sh     # SER任务 
     bash test/train_performance_1p_ser.sh     # SER性能任务 
     ```
   
   + 单机8卡训练
   
     启动8卡训练：
   
     ```bash
     bash test/train_full_8p_re.sh   # RE任务
     bash test/train_performance_8p_re.sh   # RE性能任务
       
     bash test/train_full_8p_ser.sh     # SER任务      
     bash test/train_performance_8p_ser.sh     # SER性能任务      
     ```
     
   --fp32开启FP32模式
     
   + 脚本中调用的python命令参数说明如下：
     
      ```bash
      --output_dir                                   // 训练结果和checkpoint保存路径
      --nproc_per_node                               // 训练使用的卡数
      --model_name_or_path                           // 预训练模型文件夹路径
      --do_train                                     // 执行训练
      --do_eval                                      // 执行评估
      --fp16                                         // 使用混合精度
      --fp16_opt_level                               // 混合精度级别
      --per_device_train_batch_size                  // 训练时使用的batch_size
      --warmup_ratio                                 // warmup率，用于调整学习率
     ```
     
     训练完成后，权重文件保存在output_dir路径下，并输出模型训练精度和性能信息。
     
     

# 训练结果展示

**表 2**  训练结果展示表

***RE:***

| NAME     | eval f1 |   FPS    | AMP_Type | Epochs | Batch Size |
| -------- |:---------:|:--------:| :------: | ------ | ---------- |
| 1p-NPU   |    -     |  3.03  |    O1    | 26.6      | 2         |
| 1p-竞品V |    -     |  3.934 |    O1    | 26.6      | 2         |
| 1p-竞品A |    -     |  6.282 |    O1    | 26.6      | 2         |
| 8p-NPU   |  0.6781   | 9.57 |    O1    | 208.42      | 16         |
| 8p-竞品V |  0.6820   | 20.30  |    O1    | 208.42      | 16         |
| 8p-竞品A |  0.6868   | 37.728  |    O1    | 208.42      | 16         |


***SER:***

| NAME     | eval f1 |   FPS    | AMP_Type | Epochs | Batch Size |
| -------- |:---------:|:--------:| :------: | ------ | ---------- |
| 1p-NPU   |    -     |  9.256  |    O1    | 41.67      | 8         | 
| 1p-竞品V |    -    | 13.184  |    O1    | 41.67      | 8         | 
| 1p-竞品A |    -    | 19.312  |    O1    | 41.67      | 8         | 
| 8p-NPU   |  0.8835   | 32.51 |    O1    | 333.33      | 64         |
| 8p-竞品V |  0.8892   | 68.352  |    O1    | 333.33      | 64         |
| 8p-竞品A |  0.8834   | 129.856  |    O1    | 333.33      | 64         |


# 版本说明

## 变更

2023.03.09：首次发布。
## FAQ
1.下载数据集时，出现报错**SSLCertVerificationError**时，可以将 _/site-packages/requests/api.py_ 下的 
```python 
return session.request(method=method, url=url, **kwargs)  
```
修改为
```python 
return session.request(method=method, url=url, verify=False, **kwargs)
```

# 公网地址说明

代码涉及公网地址参考 ```./public_address_statement.md```
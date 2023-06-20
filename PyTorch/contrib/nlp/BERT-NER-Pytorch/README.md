# BERT-NER-CRF for PyTorch
- [概述](#概述) 
- [准备训练环境](#准备训练环境)
- [准备数据集](#准备数据集)
- [开始训练](#开始训练)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)

## 概述
### 简述
BERT-CRF 是用于自然语言处理中实体识别任务的模型
* 参考实现 https://github.com/lonePatient/BERT-NER-Pytorch
* 本代码仓为适配NPU的实现

## 准备训练环境
### 准备环境
* 当前模型支持的PyTorch版本和已知三方库依赖如下表所示
表1 依赖库列表
| 依赖名 | 版本号 |
| --- | --- |
| PyTorch | 1.11.0 |
| transformers | 4.29.2 |

* 环境中也需要安装对应版本的CANN和torch_npu，可参考《[PyTorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC1alpha002/softwareinstall/instg/instg_000018.html)》
* 安装依赖：
```
pip install -r requirements.txt
```

### 准备数据集
* 在https://www.cluebenchmarks.com/introduce.html 下载Cluener数据集，放到datasets目录下

### 准备预训练权重
* 在https://huggingface.co/bert-base-chinese/tree/main/ 下载预训练权重和config文件等相关信息

## 开始训练
### 运行训练脚本
* 启动单机8卡训练
```
bash test/run_ner_crf.sh 
```

训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

## 训练结果展示
表2 

| Name | F1 |  ms/Iteration | Samples/Second | Epochs |
| --- | --- | --- | --- | --- |
| 8p-NPU | 79.16 | 171.9 | 1129.4 | 4 | 

## 版本说明
### 变更
2023.6.19 首次发布

### FAQ
无

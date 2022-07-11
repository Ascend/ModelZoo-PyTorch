##  基本信息

**发布者（Publisher）：黄宇威**

**应用领域（Application Domain）：Natural Language Processing**

**版本（Version）：1**

**修改时间（Modified） ：2022.6.24**

**大小（Size）：1.22GB**

**框架（Framework）：Pytorch 1.7**

**模型格式（Model Format）：ckpt**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于mosei数据集的情感识别训练代码**

## 概述

```
该模型完成了基于mosei数据集的情感识别任务
```

网络整体框架为Bert模型，中间插入了MAG多模态适应门，以完成多模态数据的情感识别。

- 参考论文：

  https://www.aclweb.org/anthology/2020.acl-main.214.pdf

- 参考实现：

  https://github.com/WasifurRahman/BERT_multimodal_transformer

- 适配昇腾 AI 处理器的实现：

  https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Research/nlp/LeNet_for_TensorFlow

- 通过Git获取对应commit_id的代码方法如下：

  ```
  git clone https://github.com/WasifurRahman/BERT_multimodal_transformer   # 克隆仓库的代码
  cd BERT_multimodal_transformer-master   # 切换到模型的代码仓目录
  
  ```

##  默认配置

- 训练数据集预处理：

  - 音频输入维度为74

  - 图像输入维度为47

  - 文本输入维度为768

    

- 测试数据集预处理：

  - 音频输入维度为74

  - 图像输入维度为47

  - 文本输入维度为768

    

- 训练超参

  - max_seq_length：50
  - train_batch_size：48
  - n_epochs: 10

##  支持特性

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 否       |
| 并行数据   | 否       |

##  训练环境准备

1. 硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南](https://gitee.com/link?target=https%3A%2F%2Fsupport.huawei.com%2Fenterprise%2Fzh%2Fcategory%2Fai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。

2. 宿主机上需要安装Docker并登录[Ascend Hub中心](https://gitee.com/link?target=https%3A%2F%2Fascendhub.huawei.com%2F%23%2Fdetail%3Fname%3Dascend-tensorflow-arm)获取镜像。

   当前模型支持的镜像列表如[表1](https://gitee.com/alvin_yan/modelzoo_demo/blob/master/LeNet_ID0127_for_TensorFlow/README.md#zh-cn_topic_0000001074498056_table1519011227314)所示。

   **表 1** 镜像列表

   

   | *镜像名称*                                                   | *镜像版本* | *配套CANN版本*                                               |
   | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ |
   | *ARM架构：[ascend-tensorflow-arm](https://gitee.com/link?target=https%3A%2F%2Fascend.huawei.com%2Fascendhub%2F%23%2Fdetail%3Fname%3Dascend-tensorflow-arm)**x86架构：[ascend-tensorflow-x86](https://gitee.com/link?target=https%3A%2F%2Fascend.huawei.com%2Fascendhub%2F%23%2Fdetail%3Fname%3Dascend-tensorflow-x86)* | *20.2.0*   | *[20.2](https://gitee.com/link?target=https%3A%2F%2Fsupport.huawei.com%2Fenterprise%2Fzh%2Fascend-computing%2Fcann-pid-251168373%2Fsoftware)* |

## 快速上手

- 数据集准备

1. 模型训练使用Mosei数据集，数据集可以从这个OBS链接获取：/obs-hyw/MAG_BERT/dataset/。

## 模型训练

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

  [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend 910训练平台环境变量设置?sort_id=3148819)

- 单卡训练

  1. 配置训练参数。

     首先在脚本test/train_full_1p.sh中，配置batch_size、steps、epochs、data_path等参数，请用户根据实际路径配置data_path，或者在启动训练的命令行中以参数形式下发。

     ```
     max_seq_length=50
     train_batch_size=48
     n_epochs=10
     data_path="/home/ma-user/modelarts/inputs/data_url_0"
     ```

  2. 启动训练。

     启动单卡训练 （脚本为BERT_multimodal_transformer-maste\test/train_full_1p.sh）

     ```
     bash train_full_1p.sh --data_path='/home/ma-user/modelarts/inputs/data_url_0'
     ```

     

##  训练结果

- 精度结果比对

| 精度指标项 | 论文发布 | GPU实测 | NPU实测 |
| ---------- | -------- | ------- | ------- |
| F1_SCORE   | 84.1     | 84.27   | 82.59   |

## 高级参考

## 脚本和示例代码

```
├── multimodal_driver.py                      //网络训练与测试代码
├── README.md                                 //代码说明文档
├── modeling.py                               //MAG门代码
├── global_configs.py                         //全局参数设置
├── argparse_utils.py                         //局部参数设置
├── bert.py                                   //bert模型代码
├── requirements.txt                          //训练python依赖列表
├── modelarts_entry_acc.py                    //单卡全量训练启动代码
├── modelarts_entry_perf.py                   //单卡训练验证性能启动代码
├── test
│    ├──train_performance_1p.sh               //单卡训练验证性能启动脚本
│    ├──train_full_1p.sh                      //单卡全量训练启动脚本
```

## 脚本参数

```
max_seq_length=50
train_batch_size=48
n_epochs=10
data_path="/obs-hyw/MAG_BERT/dataset/"
--data_path                  数据集路径，默认：/home/ma-user/modelarts/inputs/data_url_0
--max_seq_length             单句最大长度，默认：50
--train_batch_size           每个NPU的batch size，默认：48
--learing_rate          初始学习率，默认：0.00001
--n_epochs                训练epcoh数量，默认：10
```


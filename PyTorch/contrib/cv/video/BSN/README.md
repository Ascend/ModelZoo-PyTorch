# BSN训练

note
- please download data from origin repo:
- https://github.com/wzmsltw/BSN-boundary-sensitive-network/tree/master/data


```
This repo holds the pytorch-version codes of paper: "BSN: Boundary Sensitive Network for Temporal Action Proposal Generation", which is accepted in ECCV 2018. You can also find tensorflow-version implementation in [[BSN.tensorflow]](https://github.com/wzmsltw/BSN-boundary-sensitive-network).

[[Arxiv Preprint]](http://arxiv.org/abs/1806.02964)

Temporal action proposal generation is an important yet challenging problem, since temporal proposals with rich action content are indispensable for analysing real-world videos with long duration and high proportion irrelevant content. This problem requires methods not only generating proposals with precise temporal boundaries, but also retrieving proposals to cover truth action instances with high recall and high overlap using relatively fewer proposals. To address these difficulties, we introduce an effective proposal generation method, named Boundary-Sensitive Network (BSN), which adopts “local to global” fashion. Locally, BSN first locates temporal boundaries with high probabilities, then directly combines these boundaries as proposals. Globally, with Boundary-Sensitive Proposal feature, BSN retrieves proposals by evaluating the confidence of whether a proposal contains an action within its region. We conduct experiments on two challenging datasets: ActivityNet-1.3 and THUMOS14, where BSN outperforms other state-of-the-art temporal action proposal generation methods with high recall and high temporal precision. Finally, further experiments demonstrate that by combining existing action classifiers, our method significantly improves the state-of-the-art temporal action detection performance.
```

For more detail：http://arxiv.org/abs/1806.02964



## Requirements

use pytorch, you can use pip or conda to install the requirements

```
# for pip
cd $project
pip3.7 install -r requirements.txt
CANN版本：
CANN toolkit_5.0.3.1
FrameworkPTAdapter torch 1.5.0+ascend.post3.20211206
固件驱动 21.0.3.1
torch版本：
torch==1.5.0
torchvision==0.2.2
```



## 数据集准备

1.参考开源仓的方式获取数据集

开源仓链接：https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch

**注意：将解压好的数据集csv_mean_100文件夹放入data/activitynet_feature_cuhk内**

2.另下载数据集activity_net.v1-3.min.json，activity_net_1_3_new.json，anet_anno_action.json文件

开源仓链接：https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch

参考开源仓链接，将BSN-boundary-sensitive-network.pytorch/Evaluation/data内的activity_net.v1-3.min.json，activity_net_1_3_new.json文件放入本项目Evaluation/data文件夹内，将BSN-boundary-sensitive-network.pytorch/data/activitynet_annotations/anet_anno_action.json文件放入本项目data/activitynet_annotations内


2.文件结构如下：


```
BSN
|-- data                               /数据集文件夹
|   |-- activitynet_annotations        /注解
|       |-- anet_anno_action.json      /json文件
|   |-- activitynet_feature_cuhk       /数据集与数据处理脚本
|       |-- csv_mean_100               /数据集
|-- Evaluation                         /运行结果文件夹
|   |--data                            /结果模板数据集
|       |-- activity_net.v1-3.min.json /json文件
|       |-- activity_net_1_3_new.json  /json文件
|-- test                               /脚本文件夹
|   |--env_npu.sh                      /环境配置文件
|   |--train_full_1p.sh                /单卡训练shell
|   |--train_full_8p.sh                /8卡训练shell
|   |--train_performance_1p.sh         /单卡性能shell
|   |--train_performance_8p.sh         /8卡性能shell
|-- dataset.py                         /数据集预处理脚本
|-- loss_function.py                   /损失函数脚本
|-- models.py                          /模型脚本
|-- preprocess.py                      /数据集预处理脚本
|-- eval.py                            /精度测试脚本
|-- main_1p.py                         /单卡训练推理脚本
|-- main_8p.py                         /8卡训练推理脚本
|-- opts.py                            /数据加载脚本
|-- pgm.py                             /PGM脚本
|-- post_processing.py                 /后处理脚本
```

将数据集按照以上结构放在代码目录下





## TRAIN and TEST

### 单卡训练、测试

source 环境变量

```
source ./test/env_npu.sh
```

训练、测试：

```
bash ./test/train_full_1p.sh --data_path=./data/activitynet_feature_cuhk/
```

**注：单卡训练日志保存在./test/output/0/train_0.log**

单卡性能脚本：

```
bash ./test/train_performance_1p.sh --data_path=./data/activitynet_feature_cuhk/
```



### 多卡训练、测试

source 环境变量：

```
source ./test/env_npu.sh
```

训练、测试：

```
bash ./test/train_full_8p.sh --data_path=./data/activitynet_feature_cuhk/
```

   **注：多卡TEM训练日志保存在./test/output/0/train_tem_0.log**

   **注：多卡PEM训练日志保存在./test/output/0/train_pem_0.log**

多卡性能脚本：

```
bash ./test/train_performance_8p.sh --data_path=./data/activitynet_feature_cuhk/
```

模型运行结果保存在 `./output`



## BSN training result

| Ac@100 |        FPS         | Npu_nums | Epochs | AMP_Type |
| :----: | :----------------: | :------: | :----: | :------: |
| 74.64  |  TEM:215 PEM:880   |    1     |   20   |    O2    |
| 74.47  | TEM:6500 PEM:26000 |    8     |   20   |    O2    |

## 


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
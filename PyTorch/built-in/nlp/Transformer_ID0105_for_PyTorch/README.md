# Machine Translation with Transformer

## Requirements
* NPU配套的run包安装
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)
* dllogger


## Dataset Prepare
1. 运行sh run_preprocessing.sh下载数据集，并处理

## 模型训练

单卡训练流程：

1.安装环境
2.开始训练
        bash ./test/train_full_1p.sh  --data_path=数据集路径            
多卡训练流程

1.安装环境
2.开始训练
        bash ./test/train_full_8p.sh  --data_path=数据集路径           



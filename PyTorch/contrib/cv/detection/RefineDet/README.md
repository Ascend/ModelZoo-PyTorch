#RefineDet模型PyTorch 训练指导

## 1 环境准备

1.安装必要的依赖

pip install -r requirements.txt

2.获取数据集

```
sh data/scripts/VOC2007.sh
sh data/scripts/VOC2012.sh
```
下载好的数据集位于  ./data/VOCdevkit


## 2 训练

路径要写到 VOCdevkit

```
# npu env
source test/env_npu.sh

# 1p train perf
bash test/train_performance_1p.sh --data_path=xxx

# 路径要写到 VOCdevkit
# 例如

bash test/train_performance_1p.sh --data_path=./data/VOCdevkit

# 8p train perf
bash test/train_performance_8p.sh --data_path=xxx

# 8p train full
bash test/train_full_8p.sh --data_path=xxx 

# 8p eval
bash test/train_eval_8p.sh --data_path=xxx 

# finetuning
bash test/train_finetune_1p.sh --data_path=xxx 

# online inference demo 
python3.7 demo.py

```
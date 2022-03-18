# bert-base模型(finetuning)使用说明

#### Requirements
请参考requirements.txt安装相关的依赖包

#### 数据集准备

1. 下载SQuADv1.1数据集：

```
cd data/squad
mkdir v1.1
cd v1.1
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json --no-check-certificate
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json --no-check-certificate
wget https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/ -O evaluate-v1.1.py --no-check-certificate
```

2. 确认数据集路径
    请确保数据集路径如下

```
---bert_for_pytorch
---data
   ---squad
      ---v1.1
         ---train-v1.1.json
         ---dev-v1.1.json
         ---evaluate-v1.1.py
```

​     

#### 预训练模型准备
1. 本模型的训练依赖 “bert_base_pretrain.pt”（源码包内已提供），用户可根据实际情况下载或者根据目录下的bert_base_config.json自己预训练的方式得到预训练模型。
2. 确认预训练模型路径
请确保如下路径：  

```
---bert_for_pytorch
   ---checkpoints
      ---bert_base_pretrain.pt
```



#### 启动训练

##### 单卡

bash scripts/run_squad_npu_1p.sh 

##### 8卡
bash scripts/run_squad_npu_8p.sh

# Q&A

(1) Q:第一次训练的第一个step特别慢

​      A:第一次训练会对SQuAD做预处理，该过程非常耗时，通常需要十分钟左右。预处理完之后会在数据集相同目录下生成缓存文件，下次训练时会快很多。






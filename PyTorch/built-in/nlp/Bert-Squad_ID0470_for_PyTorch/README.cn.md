# bert模型(finetuning)使用说明

#### Requirements
请参考requirements.txt安装相关的依赖包

#### 数据集准备

1. 下载SQuADv1.1数据集：

```
cd data/squad
mkdir v1.1
cd v1.1
下载数据集，参照：https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT
将数据集放到v1.1目录下
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

3. 下载词典

在工程根目录执行

```
mkdir data/uncased_L-24_H-1024_A-16
cd data/uncased_L-24_H-1024_A-16
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt -O vocab.txt
cd ../../
```

#### 预训练模型准备
1. 获取预训练模型，新建checkpoints目录，并将预训练模型置于checkpoints目录下，参照：https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT
2. 确认预训练模型路径
请确保如下路径：  
bert-large:
```
---bert_for_pytorch
   ---checkpoints
      ---bert_large_pretrained_amp.pt
```
bert-base:
```
---bert_for_pytorch
   ---checkpoints
      ---bert_base_pretrained_amp.pt
```


#### bert-large启动训练

##### 单卡

bash scripts/run_squad_npu_1p.sh 

##### 8卡
bash scripts/run_squad_npu_8p.sh


#### bert-base启动训练

##### 单卡

bash scripts/run_squad_base_npu_1p.sh 

##### 8卡
bash scripts/run_squad_base_npu_8p.sh


# Q&A

(1) Q:第一次训练的第一个step特别慢

​      A:第一次训练会对SQuAD做预处理，该过程非常耗时，通常需要十分钟左右。预处理完之后会在数据集相同目录下生成缓存文件，下次训练时会快很多。







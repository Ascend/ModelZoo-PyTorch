# bert模型(finetuning)使用说明

#### Requirements
请参考requirements.txt安装相关的依赖包

#### 数据集准备

1. 下载SQuADv1.1数据集：

```
mkdir v1.1
cd v1.1
下载数据集，参照：https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT
将数据集放到v1.1目录下
```

2. 确认数据集

```
   ---squad
      ---v1.1
         ---train-v1.1.json
         ---dev-v1.1.json
         ---evaluate-v1.1.py
```

3. 下载词典

在数据集v1.1目录执行

```
mkdir data/uncased_L-24_H-1024_A-16
cd data/uncased_L-24_H-1024_A-16
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt -O vocab.txt

```

#### 预训练模型准备
1. 获取预训练模型，参照：https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT
2. 确认预训练模型
    bert-large:
```
          ---bert_large_pretrained_amp.pt
```
    bert-base:
```
         ---bert_base_pretrained_amp.pt
```
### 模型训练
参数说明：
data_path为v1.1数据集的路径,eg:/data/squad/v1.1
ckpt_path为预训练模型存放路径,只需要传入文件所在目录即可,无需包含与训练模型文件名.
#### bert-large启动训练


##### 单卡

bash test/train_large_full_1p.sh  --data_path=/data/squad/v1.1  --ckpt_path=real_path

##### 8卡
bash test/train_large_full_8p.sh  --data_path=/data/squad/v1.1  --ckpt_path=real_path


#### bert-base启动训练

##### 单卡

bash test/train_base_full_1p.sh  --data_path=/data/squad/v1.1  --ckpt_path=real_path

##### 8卡
bash test/train_base_full_8p.sh  --data_path=/data/squad/v1.1  --ckpt_path=real_path


# Q&A

(1) Q:第一次训练的第一个step特别慢

​      A:第一次训练会对SQuAD做预处理，该过程非常耗时，通常需要十分钟左右。预处理完之后会在数据集相同目录下生成缓存文件，下次训练时会快很多。







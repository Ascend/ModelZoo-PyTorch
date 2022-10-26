# SpanBERT模型(finetuning)使用说明

#### Requirements
请参考requirements.txt安装相关的依赖包
注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision ,建议Pillow版本是9.1.0 torchvision版本是0.6.0
#### 数据集准备

1. 下载SQuADv1.1数据集以及字典：

```
mkdir data
cd data
从 https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset 下载数据集
将数据集放到data目录下
```

字典文件dict.txt的下载地址为https://github.com/facebookresearch/SpanBERT/tree/main/pretraining
需放置在目录'SpanBERT/pretraining'下


2. 确认数据集

```
      ---data
         ---train-v1.1.json
         ---dev-v1.1.json
```


### 模型训练
参数说明：
data_path为SQuAD1.1数据集的路径,eg:/data, 只需要传入文件所在目录即可,无需包含文件名.
#### 启动训练


```bash

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=data

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=data

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=data

```


# Q&A

(1) Q:第一次训练的第一个step特别慢

​      A:第一次训练会对SQuAD做预处理，该过程非常耗时，通常需要十分钟左右。预处理完之后会在数据集相同目录下生成缓存文件，下次训练时会快很多。







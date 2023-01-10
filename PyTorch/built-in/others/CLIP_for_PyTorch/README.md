## 在COCO数据集上进行CLIP模型训练

### 0.简介

以下代码展示了如何在COCO数据集上进行CLIP模型的训练，
[CLIP模型](https://openai.com/blog/clip/)可用于自然语言图像检索和zero-shot图像分类。

### 1、安装依赖

```
pip3 install -r requirements.txt
```

### 2.安装transformers

```
cd transformers
pip3 install -e ./
cd ..
```

### 3、准备数据集

3.1 下载COCO数据集(2017)

```
mkdir /opt/npu/dataset/coco
cd /opt/npu/dataset/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
cd ..
```

3.2 生成clip-roberta模型数据

```bash
python3.7 save_clip_roberta.py
```
执行以上代码，将会生成clip-roberta文件夹。

### 4.训练

单卡训练

```
bash test/train_clip_full_1p.sh --data_path=/opt/npu/dataset/coco --model_path=/opt/npu/dataset/clip-roberta  --train_epochs=3    # 单卡精度训练
bash test/train_clip_performance_1p.sh --data_path=/opt/npu/dataset/coco --model_path=/opt/npu/dataset/clip-roberta  --train_epochs=1    # 单卡性能训练
```

单机8卡训练

```
bash test/train_clip_full_8p.sh --data_path=/opt/npu/dataset/coco --model_path=/opt/npu/dataset/clip-roberta  --train_epochs=3    # 8卡精度训练
bash test/train_clip_performance_8p.sh --data_path=/opt/npu/dataset/coco --model_path=/opt/npu/dataset/clip-roberta  --train_epochs=1    # 8卡性能训练
```

训练脚本参数说明:
```
--data_path:  coco数据集路径,和3.1中的coco文件夹路径保持一致
--model_path:  clip-roberta模型文件夹路径，和3.2中生成的clip-roberta文件夹路径保持一致
--train_epochs: 训练的epoch数
```


### 5、训练结果展示

**表 1**  训练结果展示表

| NAME   | eval loss |                 FPS |  AMP_Type |
|--------|----------|--------------------:| ---------:|
| 1p-NPU |    2.1984      |                25.486   |        O2 |
| 1p-竞品A |          |               |        O2 |
| 8p-NPU | 1.5591   |             193.268 |      O2 |
| 8p-竞品A | 1.5565   |             177.471 |       O2 |

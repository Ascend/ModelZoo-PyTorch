# 1.版本说明
使用DeepCTR-Torch的Tags=v0.2.6, python版本为3.7.5

# 2.准备数据集

## 2.1下载criteo数据集，并解压

下载

```
wget https://criteostorage.blob.core.windows.net/criteo-research-datasets/kaggle-display-advertising-challenge-dataset.tar.gz
```

解压

```
tar -xvf kaggle-display-advertising-challenge-dataset.tar.gz
```

解压后文件如下所示：

```
├── train.txt # train数据集，含有标注
├── test.txt # test数据集，没有标注，由于没有标注，这个数据集不使用
├── readme.txt # 说明
```

## 2.2 预处理

将模型根目录下的criteo_preprocess.py拷贝到数据集目录，然后进行预处理

```
python3 criteo_preprocess.py train.txt
```

运行上述脚本后，将在train.txt的同级目录下生成train_after_preprocess_trainval_0.93.txt和train_after_preprocess_test_0.07.txt两个文件

# 3.1p训练

```
bash ./test/train_full_1p.sh  --data_path=数据集路径           # 精度训练
bash ./test/train_performance_1p.sh  --data_path=数据集路径    # 性能训练
```

# 4.8p训练

```
bash ./test/train_full_8p.sh  --data_path=数据集路径           # 精度训练
bash ./test/train_performance_8p.sh  --data_path=数据集路径    # 性能训练
```

# 5.训练结果

```
./test/output/0
```

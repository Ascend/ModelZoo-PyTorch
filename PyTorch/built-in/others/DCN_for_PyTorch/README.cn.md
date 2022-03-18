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

修改run_dcn_1p_local.sh的数据集路径参数和npu id，按照如下注释中所示

```
python3.7 -u run_classification_criteo_dcn.py \
--npu_id=0 \ # 修改为需要运行的npu设备
--trainval_path='path/to/criteo_trainval.txt' \ # 修改为train_after_preprocess_trainval_0.93.txt的绝对路径
--test_path='path/to/criteo_test.txt' \ # 修改为train_after_preprocess_test_0.07.txt的绝对路径
--lr=0.0001 \
--use_fp16
```

启动训练

```
bash run_dcn_1p_local.sh > 1p.log &
```

训练日志会被重定向到1p.log中

# 4.8p训练

修改run_dcn_8p_local.sh的数据集路径参数，按照如下注释中所示

```
if [ $(uname -m) = "aarch64" ]
then
	for i in $(seq 0 7)
	do 
	let p_start=0+24*i
	let p_end=23+24*i
	taskset -c $p_start-$p_end python3.7 -u run_classification_criteo_dcn.py \
	--npu_id $i \
	--device_num 8 \
	--trainval_path='path/to/criteo_trainval.txt' \ # 修改为train_after_preprocess_trainval_0.93.txt的绝对路径
	--test_path='path/to/criteo_test.txt' \ # 修改为train_after_preprocess_test_0.07.txt的绝对路径
	--dist \
	--lr=0.0006 \
	--use_fp16 &
	done
else
   for i in $(seq 0 7)
   do
   python3.7 -u run_classification_criteo_dcn.py \
   --npu_id $i \
   --device_num 8 \
   --trainval_path='path/to/criteo_trainval.txt' \ # 修改为train_after_preprocess_trainval_0.93.txt的绝对路径
   --test_path='path/to/criteo_test.txt' \ # 修改为train_after_preprocess_test_0.07.txt的绝对路径
   --dist \
   --lr=0.0006 \
   --use_fp16 &
   done
fi
```

启动训练

```
bash run_dcn_8p_local.sh > 8p.log &
```

训练日志会被重定向到8p.log中
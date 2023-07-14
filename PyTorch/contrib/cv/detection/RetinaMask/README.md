## Before running

- install numactl：

```
apt-get install numactl # for Ubuntu
yum install numactl # for CentOS
```

- get R-50.pkl：

```
mkdir -p /root/.torch/models/
wget https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl
mv R-50.pkl /root/.torch/models/
```

- ln -s dataset：

```
mkdir ./dataset
ln -snf path_to_coco ./dataset/coco
```

- other requirements：

```
pip3 install torchvision==0.2.1

# other recommended requirements
apex==0.1+ascend.20220315
torch==1.5.0+ascend.post5.20220315
```

- source env and build：

```
source test/env_npu.sh
```



## Running

- To train：

```
# 1p train full
bash test/train_full_1p.sh --data_path=./dataset/

# 1p train perf
bash test/train_performance_1p.sh --data_path=./dataset/

# 8p train full
bash test/train_full_8p.sh --data_path=./dataset/

# 8p train perf
bash test/train_performance_8p.sh --data_path=./dataset/
```

- To evaluate:

```
bash test/train_eval_1p.sh --data_path=./dataset/ --weight_path=./model_0044999.pth  # for example
```



## Result

1p batch_size == 8，8p batch_size == 64

|  NAME  | Steps  | BBOX-MAP | SEGM-MAP | FPS  |
| :----: | :----: | :------: | :------: | :--: |
| GPU-1p | 360000 |    -     |    -     | 8.7  |
| GPU-8p | 20000  |   29.0   |   25.7   | 55.1 |
| NPU-1p |  400   |    -     |    -     | 4.6  |
| NPU-8p | 20000  |   28.8   |   25.7   | 34.8 |


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
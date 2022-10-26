### Centroids-reid
在数据集DukeMTMC-reID实现对Centroids-reid的训练。
- 数据下载地址：
https://ascend-pytorch-one-datasets.obs.cn-north-4.myhuaweicloud.com:443/train/zip/DukeMTMC-reID.zip
### Centroids-reid的实现细节
### 环境准备
- 安装PyTorch(pytorch.org)
- pip install -r requirements.txt
- 下载数据集DukeMTMC-reID，请在下载和解压时确保硬盘空间充足。
- 请在data文件夹遵循以下的目录结构。
```
|-- data
|   |-- DukeMTMC-reID
|   |   |-- bounding_box_test/
|   |   |-- bounding_box_train/
        ......
```
- 下载权重文件，并放在models文件夹下，models文件夹遵循以下的目录结构。
```
权重文件下载链接：
|-- models
|   |-- resnet50-19c8e357.pth
```
### 模型训练
- 注意，在Centroids-reid目录下会自动保存代码运行的日志。
- 运行脚本文件进行模型训练：
```
# 1p train perf
bash test/train_performance_1p.sh --data_path=xxx

# 8p train perf
bash test/train_performance_8p.sh --data_path=xxx

# 1p train full
bash test/train_full_1p.sh --data_path=xxx

# 8p train full
bash test/train_full_8p.sh --data_path=xxx
```
### 训练结果
Centroids-reid pytorch-lightning rusult 
| 服务器类型 | 性能       | 是否收敛 | MAP   |
|-------    |----------  |------ |-------- |
| GPU1卡    | 1.91it/s   | 是    | 0.95844 |
| GPU8卡    | 1.20it/s   | 是    | 0.94051 |
| NPU1卡    | 2.26it/s   | 是    | 0.96056 |
| NPU8卡    | 1.30it/s   | 是    | 0.95472 |
### 其他说明
- 在centroids-reid-main/configs目录下找到256_resnet50.yml，将文件中的PRETRAIN_PATH修改为权重文件resnet50-19c8e357.pth的当前路径


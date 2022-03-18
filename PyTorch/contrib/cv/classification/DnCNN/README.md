# DnCNN
原代码的gpu版本的github地址 https://github.com/SaoYan/DnCNN-PyTorch 

## 运行须知
- python3.7安装
- 安装华为的torch版本的包
- 根据requirements文件安装相应的python依赖 pip install -r requirements.txt
- 需要拷贝该目录下的数据集，在 https://github.com/SaoYan/DnCNN-PyTorch 下面的 data 目录整个拷贝到 DnCNN 目录下

## 训练
使用下面的命令执行，执行完后会对应的生成log。
```
# 1p train 
bash ./test/train_full_1p.sh  --data_path=数据集路径    # 精度训练
bash ./test/train_performance_1p.sh  --data_path=数据集路径  # 性能训练

# 1p train 
bash ./test/train_full_8p.sh  --data_path=数据集路径         # 精度训练
bash ./test/train_performance_8p.sh  --data_path=数据集路径  # 性能训练

# online inference demo
bash test/demo.sh

# To ONNX
bash test/pth2onnx.sh
```
训练好的模型存放在与 test 同级的目录下，命名为 net_1p.pth或net8p.pth
## 训练日志路径
日志存放在./test/output文件夹下，以 8卡为例
```
test/output/devie_id/train_${device_id}.log           # 训练脚本原生日志
test/output/devie_id/DncNN_bs512_8p_perf.log          # 8p性能训练结果日志
test/output/devie_id/ShuffleNetV1_bs8192_8p_acc.log   # 8p精度训练结果日志
```

## DnCNN training result 

|  类型     | 精度      | FPS     | NPU/GPU 卡数 | BatchSize | Epochs | AMP_Type |
| :------:  | :----:     | :-----: | :------: | :----:    | :----: | :------: |
|  NPU-1P   |   31.85    | 10520   |    1     |  512      |  50   |    O2    |
|  NPU-8P   |   31.12    | 32100    |    8     |  512      |  50   |    O2    |
|  GPU-1P   |   32.06    | 2200    |    1     |  512      |  50   |    O2    |
|  GPU-8P   |   31.35    |  --     |    8     |  512      |  50   |    O2    |




# ctpn.pytorch
Pytorch implementation of CTPN (Detecting Text in Natural Image with Connectionist Text Proposal Network)

# Paper
https://arxiv.org/pdf/1609.03605.pdf

# 环境准备

请使用apt-get install zip或yum install zip安装压缩工具zip

# 数据准备

请下载 icdar13 dataset并解压到`${ROOT}/data/icdar13/`，并将其中的`gt.zip`标签文件移动到`${ROOT}`/下
# Training

```bash
# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=real_data_path

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path

```

# Test
测试的权重采用最后一个epoch的权重文件，即当epoch=200时，权重路径为output_models/checkpoint-200.pth.tar
```
# test 8p accuracy
bash test/train_eval.sh --data_path=real_data_path --pth_path=output_models/checkpoint-200.pth.tar
```
测试精度包含三个部分，hmean为用于比对的精度
```
Calculated!{"precision": 0.7331386861313869, "recall": 0.7094063926940639, "hmean": 0.7210773213359865}
```
## CTPN training result

|  名称  | 精度 |  性能 |
| :----: | :--: |  :------: |
| NPU-8p |  72.1 |   17.66fps   |
| GPU-8p | 72.4  |    13.25fps    |
| NPU-1p |       |     1.695fps   |
| GPU-1p |       |    6.295fps|
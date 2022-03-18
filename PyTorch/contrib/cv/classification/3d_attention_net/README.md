# 3d_attention_net

* 该项目在CIFAR-10数据集上实现了3d_attention_net
* 参考实现：https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch.git

## Requirement

1. 运行程序所需安装的环境依赖在requirements.txt中

2. 获取CIFAR-10数据
   * 在当前目录下创建data目录。
   * 数据集获取方式请参考模型参考仓readme，参考仓：https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch.git
   * 将下载好的CIFAR-10数据集上传至data目录，而后解压。

## Training

```bash
# training 1p accuracy
bash ./test/train_full_1p.bash

# training 1p performance
bash ./test/train_performance_1p.bash

# training 8p accuracy
bash ./test/train_full_8p.bash

# training 8p performance
bash ./test/train_performance_8p.bash

# finetune 1p
bash ./test/train_finetune_1p.sh

# run demo
python demo.py
```

## Result

| Top1 acc | FPS  | Epochs | AMP_Type | Device |  Bs  |
| :------: | :--: | :----: | :------: | :----: | :--: |
|    -     | 1432 |   1    |    O2    | npu_1p | 512  |
|  85.48   | 9587 |  300   |    O2    | npu_8p | 512  |
|    -     | 1396 |   1    |    O2    | gpu_1p | 512  |
|  85.37   | 8342 |  300   |    O2    | gpu_8p | 512  |

| Top1 acc | FPS  | Epochs | AMP_Type | Device |  bs  |
| :------: | :--: | :----: | :------: | :----: | :--: |
|  94.13   | 723  |  300   |    O2    | npu_8p |  32  |
|  94.32   | 2461 |  300   |    O2    | gpu_8p |  32  |


# SlowFast

SlowFast 模型在 Kinetics400 数据集上的实现，主要修改自 [open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2) 源码

## 环境准备

- 安装 Pytorch 和混合精度训练工具 Apex
- 安装依赖 `pip install -r requirements.txt`
- 下载 kinetics400 数据集

注：有关依赖中的 decord，若在 x86 平台上，可以直接通过 `pip install decord` 进行安装，若在 ARM Linux 平台上，则需通过源码编译安装，具体参见 [dmlc/decord](https://github.com/dmlc/decord/blob/master/README.md)

## 安装 MMCV

下载 mmcv 源码

```shell
git clone -b v1.3.9 https://github.com/open-mmlab/mmcv.git
mv mmcv/ mmcv-master/
mv mmcv-master/mmcv ./
rm -rf mmcv-master/
```

替换 mmcv 中的文件

```shell
/bin/cp -f mmcv_need/base_runner.py mmcv/runner/base_runner.py
/bin/cp -f mmcv_need/builder.py mmcv/runner/optimizer/builder.py
/bin/cp -f mmcv_need/checkpoint.py mmcv/runner/hooks/checkpoint.py
/bin/cp -f mmcv_need/data_parallel.py mmcv/parallel/data_parallel.py
/bin/cp -f mmcv_need/dist_utils.py mmcv/runner/dist_utils.py
/bin/cp -f mmcv_need/distributed.py mmcv/parallel/distributed.py
/bin/cp -f mmcv_need/epoch_based_runner.py mmcv/runner/epoch_based_runner.py
/bin/cp -f mmcv_need/iter_timer.py mmcv/runner/hooks/iter_timer.py
/bin/cp -f mmcv_need/optimizer.py mmcv/runner/hooks/optimizer.py
/bin/cp -f mmcv_need/test.py mmcv/engine/test.py
/bin/cp -f mmcv_need/transformer.py mmcv/cnn/bricks/transformer.py
```

## 训练

训练阶段，脚本调用 `train.py` 进行训练

```bash
# 1p train perf
bash test/train_performance_1p.sh --data_path=/opt/npu

# 8p train perf
bash test/train_performance_8p.sh --data_path=/opt/npu

# 8p train full
bash test/train_full_8p.sh --data_path=/opt/npu

# 8p eval 
bash test/train_eval_8p.sh --data_path=/opt/npu

# finetuning
bash test/train_finetune_1p.sh --data_path=/opt/npu
```

注: 可以通过修改 `--data_path` 来指定数据集文件夹的位置，例如，你的数据集地址为：`/opt/npu/kinetics400`， 可设置 `--data_path=/opt/npu`

Log Path:
- slowfast_performance_1p.log    # 1p 训练下性能测试日志
- slowfast_performance_8p.log    # 8p 训练下性能测试日志
- slowfast_full_1p.log       # 1p 完整训练下性能和精度测试日志
- slowfast_full_8p.log       # 8p 完整训练下性能和精度测试日志
- slowfast_eval_8p.log       # 8p 测试模型验证集精度日志
- slowfast_finetune_1p.log   # 1p 下 fine-tuning 日志

## TSM-NonLocal 训练结果

| top1 acc (单view) |   FPS   |  Epochs | AMP_Type |  Device  |
|  :---:   | :-----: |  :---:  | :------: | :------: |
|    -     |  11  |    1    |    O1    |  1p Npu  |
|  33.43 (30 epochs)   | 81  |    256   |    O1    |  8p Npu  |
|    -     |  14  |    1    |    O1    |  1p Gpu  |
|  30.24 (30 epochs)   | 86  |    256   |    O1    |  8p Gpu  |

注：源仓库模型单 view 测试为 70.19 (256 epochs)，30 views 测试为 77.0（256 epochs）


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
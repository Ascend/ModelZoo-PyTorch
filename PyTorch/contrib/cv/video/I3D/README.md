# I3D

这是I3D模型在kinetics400模型上的训练部分，修改于来自[open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2)的源码




## Requirements 

- 安装提供NPU支持的PyTorch和混合精度Apex模块
- 安装必备的依赖包，命令：`pip install -r requirements.txt`
- 下载kinetics400数据集，并将其放在**/opt/npu**文件夹下

## 安装 decord

如果是ARM平台的处理器，需要源码编译decord，详见：https://github.com/dmlc/decord/blob/master/README.md

如果是X86平台的处理器，直接通过`pip install decord `来进行安装

## 安装 MMCV
从源码进行mmcv的安装：
```
git clone -b v1.3.9 https://github.com/open-mmlab/mmcv.git
mv mmcv/ mmcv-master/
mv mmcv-master/mmcv ./
rm -rf mmcv-master/
```

替换掉mmcv库里的一些文件：
```
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
/bin/cp -f mmcv_need/registry.py mmcv/utils/registry.py
```

## 安装 mmaction2

关于mmaction2的安装，详见：https://github.com/open-mmlab/mmaction2/blob/master/docs_zh_CN/install.md

注意安装不含cuda算子的版本

将此库里的mmaction完全平替掉mmaction2/mmaction

## 数据集获取和处理

关于kinetics400数据集的获取和处理，详见：https://github.com/open-mmlab/mmaction2/blob/master/tools/data/kinetics/README_zh-CN.md

## 训练 

在训练阶段，接口脚本为： `train.py` 

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

# online inference demo
source test/env_npu.sh
python3.7 demo.py
```
注意:
- 如果你把数据集放在了其他位置而不是/opt/npu，可以通过修改`--data_path`来进行数据集文件夹的指定
  例如，你的数据集地址为：/home/dataset/kinetics400， 那么`--data_path=/home/dataset`

日志地址:
- tsm_performance_1p.log    # 1p下测试性能的结果日志
- tsm_performance_8p.log    # 8p下测试性能的结果日志
- tsm_full_1p.log       # 1p下完整训练的性能和精度的结果日志
- tsm_full_8p.log       # 8p 下完整训练的性能和精度的结果日志
- tsm_eval_8p.log       # 8p 下验证精度的结果日志
- tsm_finetune_1p.log   # 1p下fine-tuning的结果日志

## I3D 训练结果 

| Top1 acc |  FPS  | Epochs | AMP_Type | Device |
| :------: | :---: | :----: | :------: | :----: |
|    -     | 18.41 |   1    |    O1    | 1p Npu |
|  56.26   | 58.92 |   40   |    O1    | 8p Npu |
|    -     | 20.68 |   1    |    O1    | 1p Gpu |
|  53.78   | 74.84 |   40   |    O1    | 8p Gpu |



## FAQ

1.在安装decord库的时候apt_pak报错
```
使用如下命令查看python版本
ls /usr/lib/python3/dist-packages/apt_pkg*
使用下面命令：
vim /usr/bin/apt-add-repository
把首行的
#! /usr/bin/python3
改为对应版本（我这里是python3.6）
#! /usr/bin/python3.6
```
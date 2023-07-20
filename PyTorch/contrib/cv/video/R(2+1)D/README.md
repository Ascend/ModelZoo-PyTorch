# R(2+1)d模型使用说明

## Dataset Prepare

将处理好的数据集放到模型目录的 /data 文件夹下

## Requirements

安装依赖
```shell
pip3.7 install ./requirements/requirements.txt
注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision ,建议Pillow版本是9.1.0 torchvision版本是0.6.0
```

安装mmcv

```shell
git clone -b 1.3.9 https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # 安装full版本
```

## Before Train

更改底层文件,分别修改mmcv和apex的底层源码

```shell
# mmcv_need
/bin/cp -f additional_need/mmcv/distributed.py ../mmcv/mmcv/parallel/
/bin/cp -f additional_need/mmcv/dist_utils.py ../mmcv/mmcv/runner/
/bin/cp -f additional_need/mmcv/optimizer.py ../mmcv/mmcv/runner/hooks/
/bin/cp -f additional_need/mmcv/text.py ../mmcv/mmcv/runner/hooks/logger/

# apex_need
/bin/cp -f additional_need/amp/scaler.py ../apex/amp/
```

## Train MODEL

## 单卡

```shell
# 精度
sh ./test/run_1p.sh

# 性能
sh ./test/run_1p_perf.sh
```

## 8卡

```shell
chmod a+x ./tools/dist_train.sh

# 精度
sh ./test/run_8p.sh

# 性能
sh ./test/run_8p_perf.sh
```

## log文件位置

运行产生的log文件和模型的参数文件在 项目目录的 work_dirs 文件夹下

## 公网地址说明

代码涉及公网地址参考 public_address_statement.md
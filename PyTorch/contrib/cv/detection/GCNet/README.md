# GCNet

This implements training of GCNet on the Coco dataset, mainly modified from [pytorch/examples](https://github.com/open-mmlab/mmdetection).

## GCNet Detail

GCNet is initially described in [arxiv](https://arxiv.org/abs/1904.11492). Via absorbing advantages of Non-Local Networks (NLNet) and Squeeze-Excitation Networks (SENet), GCNet provides a simple, fast and effective approach for global context modeling, which generally outperforms both NLNet and SENet on major benchmarks for various recognition tasks.

## Requirements

- NPU配套的run包安装
- Python 3.7.5
- PyTorch(NPU版本)
- apex(NPU版本)

### Document and data preparation

1. 下载压缩GCNet文件夹
2. 于npu服务器解压GCNet压缩包
3. 准备coco数据集并放置在指定位置

### Download and modify mmcv

1. 下载mmcv-full，使用的版本为1.3.8; 下载mmdetection，使用的版本为1.2.7

```
git clone -b v1.3.8 https://github.com/open-mmlab/mmcv.git
git clone -b v2.10.0 https://github.com/open-mmlab/mmdetection.git
```

2. 用GCNet/dependency/mmcv目录替换clone文件夹里mmcv的mmcv（mmcv/mmcv）

```
cp -f dependency/mmcv/_functions.py ./mmcv/mmcv/parallel/
cp -f dependency/mmcv/base_runner.py ./mmcv/mmcv/runner/
cp -f dependency/mmcv/builder.py ./mmcv/mmcv/runner/optimizer/
cp -f dependency/mmcv/checkpoint.py ./mmcv/mmcv/runner/
cp -f dependency/mmcv/context_block.py ./mmcv/mmcv/cnn/bricks
cp -f dependency/mmcv/data_parallel.py ./mmcv/mmcv/parallel/
cp -f dependency/mmcv/dist_utils.py ./mmcv/mmcv/runner/
cp -f dependency/mmcv/distributed.py ./mmcv/mmcv/parallel/
cp -f dependency/mmcv/epoch_based_runner.py ./mmcv/mmcv/runner/
cp -f dependency/mmcv/iter_based_runner.py ./mmcv/mmcv/runner/
cp -f dependency/mmcv/iter_timer.py ./mmcv/mmcv/runner/hooks/
cp -f dependency/mmcv/optimizer.py ./mmcv/mmcv/runner/hooks/
cp -f dependency/mmcv/roi_align.py ./mmcv/mmcv/ops/
```

或是pip安装mmcv-full后手动替换库文件

3. 用GCNet/dependency/mmdet目录替换clone文件夹里mmdetection的mmdet（mmdetection/mmdet）

```
rm -r mmdetection/mmdet
cp -r dependency/mmdet mmdetection/
```

或是pip安装mmdet后手动替换库文件



### Configure the environment

1. 推荐使用conda管理

```
conda create -n gcnet --clone env  # 复制一个已经包含依赖包的环境 
conda activate gcnet
```

2. 配置安装mmcv

```
source ./test/env_npu.sh
cd mmcv
export MMCV_WITH_OPS=1
export MAX_JOBS=8
python3 setup.py build_ext
python3 setup.py develop
pip3 list | grep mmcv  # 查看版本和路径
```

3. 配置安装mmdet

```
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

注意：

```
如果复制的conda环境安装过pycocotools包，需要卸载后重新安装mmpycocotools
pip uninstall pycocotools
pip install mmpycocotools
```



## Train MODEL

### 进入GCNet文件夹下

```
cd GCNet
```

```
# 1p train_1p
bash ./test/train_full_1p.sh  --data_path=数据集路径

#  8p train_8p
bash ./test/train_full_8p.sh  --data_path=数据集路径

#  8p perf_8p
bash ./test/train_performance_8p.sh    --data_path=数据集路径

#  1p perf_1p
bash ./test/train_performance_1p.sh    --data_path=数据集路径

# 1p eval
bash ./test/eval.sh  --weight_path=数据集路径
```



### 参考精度/性能

| 名称   | 精度(mAP) | 性能(fps) |
| ------ | --------- | --------- |
| GPU-1p | -         | 8.47      |
| GPU-8p | 39.9      | 44.62     |
| NPU-1p | -         | 0.52      |
| NPU-8p | 39.1      | 2.35      |


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
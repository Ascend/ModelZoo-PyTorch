# NonLocal 模型 PyTorch 离线推理指导

## 1 环境准备

1. 安装必要依赖

```shell
pip3.7 install -r requirements.txt
```

2. 获取，修改，安装开源模型代码

```shell
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
git checkout 92e5517f1b3cbf937078d66c0dc5c4ba7abf7a08
git am --signoff < ../nonlocal.patch
pip3.7 install -r requirements/build.txt
pip3.7 install -v -e .
cd ..
```

3. 获取权重文件

tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb_20200724-d8ad84d2.pth

4. 数据集

使用 kinetics400 数据集，受磁盘空间限制，tsm_nonlocal 的离线推理使用 video 格式的数据集（没有抽帧成 rawframes），故需要安装 decord 用于数据前处理时的在线解帧。在 x86 架构下，可以直接使用指令 `pip3.7 install decord` 安装，而在 arm 架构下，需源码编译安装 decord。下载 video 格式的数据集，可按照 MMAction2 [准备 Kinetics 数据集](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/kinetics/README_zh-CN.md) 中的引导进行。下载数据集和标注文件并生成文件列表的指令如下所示：

```shell
cd ./mmaction2/tools/data/kinetics

# 若只下载验证集用于推理，则可以删除以下 3 个 shell 脚本中，与训练集有关的指令
bash download_annotations.sh kinetics400
bash download_videos.sh kinetics400
bash generate_videos_filelist.sh kinetics400

cd ../../../..
```

注：若已经预先准备好数据集和文件列表，则需在 mmaction2 文件夹中的相应位置处，软链接到已有文件，如下所示：

```shell
mkdir -p ./mmaction2/data/kinetics400
ln -s /your/validation/dataset/path ./mmaction2/data/kinetics400/videos_val
ln -s /your/validation/dataset/filelist/path ./mmaction2/data/kinetics400/kinetics400_val_list_videos.txt

按如上操作整理数据集和文件列表后，文件树应当如下

```
NonLocal
├── test
├── mmaction2
│   ├── mmaction
│   ├── tools
│   ├── configs
│   ├── data
│   │   ├── kinetics400
│   │   │   ├── kinetics400_val_list_videos.txt
│   │   │   ├── videos_val
│   │   │   ├── ...
│   ├── ...
├── ...
```

5. [获取 benchmark 工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)

将 benchmark.x86_64 或 benchmark.aarch64 放在当前目录

6. [获取 msame 工具](https://gitee.com/ascend/tools.git)

获取并安装 msame 工具，其可执行二进制文件应位于 `tools/msame/out/msame` 处

## 2 离线推理

在 Ascend 310 上执行，推理前使用 `npu-smi info` 查看设备状态，确保设备空闲

```shell
# OM model generation
bash test/pth2om.sh

# OM model inference
bash test/eval_acc_perf.sh --datasets_path=mmaction2/data/kinetics400

# ONNX model inference
bash test/perf_gpu.sh
```

## 评测结果

| 模型 | pth 精度 | 310 精度 | 性能基准 | 310 性能 |
| :------: | :------: | :------: | :------:  | :------:  |
| tsm_nl bs 1 | top1 acc: 71.60% | top1 acc: 71.63% | 59.307 fps | 41.498 fps |
| tsm_nl bs 4 | top1 acc: 71.60% | top1 acc: 71.63% | 65.407 fps | 51.284 fps |
| tsm_nl bs 8 | top1 acc: 71.60% | top1 acc: 71.63% | 66.042 fps | 45.436 fps |
| tsm_nl bs 16 | top1 acc: 71.60% | top1 acc: 71.63% | 64.957 fps | 44.134 fps |

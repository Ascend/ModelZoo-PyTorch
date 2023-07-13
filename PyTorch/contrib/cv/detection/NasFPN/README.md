# NasFPN

本项目实现了NasFPN 在 NPU 上的训练, 迁移自 mmdetection-2.11.0.
[mmdetection github链接](https://github.com/open-mmlab/mmdetection)

## NasFPN Detail

本项目对 mmdetection-2.11.0 做了如下更改：
1. 迁移到 NPU 上
2. 使用混合精度训练
3. 对于一些操作使用 NPU 算子优化性能


## Requirements

- NPU 配套的 run 包安装(20211018)
- Python 3.7.5
- PyTorch(NPU20210930 版本)
- apex(NPU20210930 版本)
- torchvision 0.6.0
- decorator
- sympy
- opencv-python
- 准备coco2017数据集

## Install

```bash
# 安装mmcv
cd NasFPN
cd ../
git clone -b v1.2.6 --depth=1 https://github.com/open-mmlab/mmcv.git
export MMCV_WITH_OPS=1
export MAX_JOBS=8
source NasFPN/test/env_npu.sh
cd mmcv
python3 setup.py build_ext
python3 setup.py develop
pip3 list | grep mmcv

# 安装mmdetection
cd ../NasFPN
pip3 install -r requirements/build.txt
pip3 install -v -e .
pip3 list | grep mm

# 替换mmcv中的部分文件
cp -f mmcv_need/_functions.py ../mmcv/mmcv/parallel/
cp -f mmcv_need/builder.py ../mmcv/mmcv/runner/optimizer/
cp -f mmcv_need/data_parallel.py ../mmcv/mmcv/parallel/
cp -f mmcv_need/dist_utils.py ../mmcv/mmcv/runner/
cp -f mmcv_need/distributed.py ../mmcv/mmcv/parallel/
cp -f mmcv_need/optimizer.py ../mmcv/mmcv/runner/hooks/
cp -f mmcv_need/text.py ../mmcv/mmcv/runner/hooks/logger/
cp -f mmcv_need/epoch_based_runner.py ../mmcv/mmcv/runner/
```
## Training

```bash
cd NasFPN

# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=/opt/npu/coco

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=/opt/npu/coco

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=/opt/npu/coco

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=/opt/npu/c0c0

#test 8p accuracy
bash test/train_eval_8p.sh --data_path=/opt/npu/coco --pth_path=work_dirs/retinanet_r50_nasfpn_crop640_50e_coco/latest.pth

# finetuning 1p
bash test/train_finetune_1p.sh --data_path=/opt/npu/coco --pth_path=work_dirs/retinanet_r50_nasfpn_crop640_50e_coco/latest.pth
```

Log path:
    test/output/${device_id}/train_${device_id}.log           # training detail log
    test/output/${device_id}/NasFPN_bs192_8p_perf.log  # 8p training performance result log
    test/output/${device_id}/NasFPN_bs192_8p_acc.log   # 8p training accuracy result log



## CascadedMaskRCNN training result

| Acc@1 |  FPS  | Npu_nums | Epochs | AMP_Type |
| :---: | :---: | :------: | :----: | :------: |
|   -   | 18.7  |    1     |   50   |    O1    |
| 40.4  | 109.5 |    8     |   50   |    O1    |

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
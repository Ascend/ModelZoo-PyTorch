# SPACH

This implements training of SPACH on the ImageNet dataset.

Modified from [microsoft/SPACH](https://github.com/microsoft/SPACH).


## SPACH Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.

Therefore, SPACH is re-implemented using semantics such as custom OP.


## Requirements
+ install numactl：

```
apt-get install numactl # for Ubuntu
yum install numactl # for CentOS
```
+ install requirement
```
pip3 install torchvision
pip3 install einops==0.4.1
pip3 install --no-deps timm==0.4.5
Note:Install the torchvision that corresponds to the torch version
```
- source env and build：

```
source test/env_npu.sh
```
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

推荐将数据集该挂载在内存中或者nvme硬盘，模型训练较依赖IO性能。否则性能可能不达标。

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=real_data_path

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path

#test 8p accuracy
bash test/train_eval_8p.sh --data_path=real_data_path --resume=real_pre_train_model_path
```

Log path:

    test/output/devie_id/train_${device_id}.log           # training detail log

    test/output/devie_id/SPACH_2_bs8192_8p_perf.log  # 8p training performance result log

    test/output/devie_id/SPACH_2_bs8192_8p_acc.log   # 8p training accuracy result log


## SPACH training result

batch_size=128

| Acc@1    | FPS       | Name     | Epochs   | AMP_Type |
| :----:   | :----:    | :------: | :------: | :--:     |
| :------: | 298.9424  | GPU-1p   | 5        | O2       |
| 81.6%    | 2604.2726 | GPU-8p   | 300      | O2       |
| :------: | 297.3475  | NPU-1p   | 5        | O2       |
| 82.1%    | 2626.9882 | NPU-8p   | 300      | O2       |

# Swin-Transformer

This implements training of Swin-Transformer on the ImageNet dataset, mainly modified
from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).

## Swin-Transformer Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset
    - Then, and move validation images to labeled subfolders,
      using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

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
bash test/train_eval_8p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path

```

## 多机多卡性能数据获取流程

```shell
	1. 安装环境
	2. 开始训练，每个机器所请按下面提示进行配置
        bash ./test/train_performance_multinodes.sh  --data_path=数据集路径 --batch_size=单卡batch_size*所有卡数 --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
```

Log path:
test/output/devie_id/train_${device_id}.log # training detail log
test/output/devie_id/WideReesnet50_2_bs8192_8p_perf.log # 8p training performance result log
test/output/devie_id/WideReesnet50_2_bs8192_8p_acc.log # 8p training accuracy result log

## Swin-Transformer training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 142       | 1        | 1        | O2       |
| 81.32   | 1222      | 8        | 200      | O2      |

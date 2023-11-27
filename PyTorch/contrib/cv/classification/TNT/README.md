# TNT

This implements training of TNT on the ImageNet dataset, mainly modified from [CV-Backbones/tnt_pytorch](https://github.com/huawei-noah/CV-backbones/tree/master/tnt_pytorch).

## TNT Detail

TNT divides the image patches into sub-patches. With the structure of visual sentences and visual words, TNT can better perform the image classification task.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
  Note:Install the torchvision that corresponds to the torch version
- Download the ImageNet dataset
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
- Please add the following shapes into /usr/local/Ascend/ascend-toolkit/5.0.3/arm64-linux/opp/op_impl/built-in/ai_core/tbe/impl/dynamic/transpose.py for better performance.
  - [2, 25088, 4, 16, 6], [25088, 16, 2, 4, 6]

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

# finetuning 1p 
bash test/train_finetune_1p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path
```

Log path:
    test/output/devie_id/train_${device_id}.log           # training detail log
    test/output/devie_id/WideReesnet50_2_bs8192_8p_perf.log  # 8p training performance result log
    test/output/devie_id/WideReesnet50_2_bs8192_8p_acc.log   # 8p training accuracy result log



## TNT training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 57        | 1        | 1        | O1       |
| 71.4%    | 410       | 8        | 79       | O1       |

自测结果说明：
- 项目交付精度要求：Acc@1 81.5%，目前在gpu中Acc@1为81.2%，已达目标精度的99%以上。
  
> 由于跑完310个epoch训练时间过长，本项目中测试npu上79个epoch后Acc@1为71.4%，此时gpu的Acc@1为71.7%，910A（NPU）精度大于V100（GPU）精度的99%（即71.7% * 99% = 71.0%），说明精度已经对齐，符合验收标准。


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md
# Deit

This implements training of Deit on the ImageNet dataset, mainly modified from [facebookresearch/deit](https://github.com/facebookresearch/deit).

## Deit Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.Therefore, Deit is re-implemented using semantics such as custom OP.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path

# 8p eval
bash test/train_eval_8p.sh --data_path=real_data_path --pth_path=./checkpoints/model_best.pth

# finetuning
bash test/train_finetune_1p.sh --data_path=real_data_path --data_set=data_set_name --pth_path=./checkpoints/checkpoint.pth

# online inference demo 
python demo.py
```

Log path:
    test/output/devie_id/train_${device_id}.log           # training detail log
    test/output/devie_id/Deit_Small_bs512_8p_perf.log  # 8p training performance result log
    test/output/devie_id/Deit_Small_bs512_8p_acc.log   # 8p training accuracy result log



## Deit training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 270       | 1        | 1        | O1       |
| 79.996   | 2086      | 8        | 300      | O1       |

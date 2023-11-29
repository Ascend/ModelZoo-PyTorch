# XCIT

This implements training of XCIT on the ImageNet dataset, mainly modified
from [XCIT](https://github.com/facebookresearch/xcit).

## XCIT Detail

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

Log path:
test/output/devie_id/train_${device_id}.log # training detail log
test/output/devie_id/WideReesnet50_2_bs8192_8p_perf.log # 8p training performance result log
test/output/devie_id/WideReesnet50_2_bs8192_8p_acc.log # 8p training accuracy result log

## XCIT training result
| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        |   173    | 1        | 1        | O1       |
| 81.89   |  1401     | 8        | 300      | O1      |


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md
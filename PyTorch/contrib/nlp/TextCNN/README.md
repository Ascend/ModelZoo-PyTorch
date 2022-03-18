# TextCNN

## TextCNN Detail

本仓库为使用Ascend NPU实现TextCNN迁移的仓库。请使用NPU运行该仓库。

## Requirements

- 安装ascend pytorch 1.5
- 安装 requirement.txt
- fork 本仓库

## Training

To train a model, run `run.py` with the desired model architecture and the path to the THUCNEWS dataset:

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


## TextCNN training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type | Loss_Scale |
| :------: | :------:  | :------: | :------: | :------: | :--------: |
|     -    | 5600      | 1        | 1        | O1       | 16.0       |
|  91.40   | 56500     | 8        | 20       | O1       | 16.0       |
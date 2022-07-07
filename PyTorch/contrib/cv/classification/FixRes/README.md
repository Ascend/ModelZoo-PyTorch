# FixRes

This implements training of FixRes on the ImageNet dataset, mainly modified from [facebookresearch/FixRes](https://github.com/facebookresearch/FixRes).

## FixRes Detail

FixRes first trains a ResNet50 model with 224x224 input images. Then 384x384 input images are used to finetune and evaluate the model.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

To train a model, run `main_resnet50_scratch.py` with the desired model architecture and the path to the ImageNet dataset.Before training, you need to create a 'train_cache' folder under the model directory:

```bash
# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=real_data_path

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path

# finetuning 1p 
bash test/finetune_full_1p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path

# finetuning 1p performance
bash ./test/finetune_performance_1p.sh --data_path=real_data_path

# finetuning 8p 
bash test/finetune_full_8p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path

# finetuning 8p performance
bash ./test/finetune_performance_8p.sh --data_path=real_data_path

#test 8p accuracy
bash test/train_eval_8p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path

```

Log path:
    test/output/devie_id/train_${device_id}.log           # training detail log
    test/output/devie_id/FixRes_bs512_8p_perf.log         # 8p training performance result log
    test/output/devie_id/FixRes_bs512_8p_acc.log          # 8p training accuracy result log



## FixRes training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 590 (train) / 711(fineutune) | 1        | 120 + 60 | O1       |
| 72.9% (72.1% before finetune)    | 3318(train) / 3515(finetune) | 8        | 120 + 60 | O1       |
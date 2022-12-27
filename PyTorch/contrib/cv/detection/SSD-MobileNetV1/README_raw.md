# SSD-MobileNetV1·

This implements training of SSD-MobileNetV1 on the VOC dataset, mainly modified from [pytorch/examples](https://github.com/qfgaohao/pytorch-ssd.git).

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the VOC dataset from pjreddie
- Download pretrained models ([models](https://storage.googleapis.com/models-hao/mobilenet_v1_with_relu_69_5.pth))

## Training

To train a model, run `main.py` with the path to the VOC dataset:

```bash
# a dirctory to save model and label
mkdir models
# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=real_data_path  --validation_data_path=real_validation_path

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path  --validation_data_path=real_validation_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path --validation_data_path=real_validation_path --loss_scale=128.0

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path  --validation_data_path=real_validation_path --loss_scale=128.0

#test 8p accuracy
bash test/train_eval.sh --data_path=real_data_path --pth_path=real_pre_train_model_path

# finetuning 1p 
bash test/train_finetune_1p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path 
```

Log path:
    test/output/devie_id/train_${device_id}.log           # training detail log
    test/output/devie_id/SSD-MobileNetV1_bs32_8p_perf.log  # 8p training performance result log
    test/output/devie_id/SSD-MobileNetV1_bs32_8p_acc.log   # 8p training accuracy result log



## SSD-MobileNetV1 training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| 0.67662        | 54       | 1        | 240        | O1       |
| 0.6783   | 1000     | 8        | 240      | O2       |

# Srcnn_x2_for_Pytorch

This implements training of Srcnn_x2 on the 91-image dataset and testing on the Set5 dataset, mainly modified from [SRCNN](https://github.com/yjn870/SRCNN-pytorch).

## Srcnn_x2_for_Pytorch Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.Therefore, Srcnn_x2 is re-implemented using semantics such as custom OP.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the 91-image dataset from [91-image-x2](https://www.dropbox.com/s/2hsah93sxgegsry/91-image_x2.h5?dl=0) as training dataset.
- Download the Set5 dataset from [Set5-x2](https://www.dropbox.com/s/r8qs6tp395hgh8g/Set5_x2.h5?dl=0) as testing dataset.

## Training

To train a model, run `train.py` with the desired model architecture and the path to the 91-image dataset:

```bash
# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path

# finetuning 1p 
bash test/train_finetune_1p.sh --data_path=real_data_path --pre_train_path=real_pre_train_model_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path

# testing 8p accuracy
bash test/train_eval_8p.sh --data_path=real_data_path --pre_train_path=real_pre_train_model_path

# demo
# 输出图片在real_image_path文件夹，xxxx_srcnn_x2.bmp
python3 demo.py --pre-train-path=real_pre_train_model_path --image-file=real_image_path
```

Log path:
    test/output/0/train_0.log                 # training detail log
    test/output/0/Srcnn_x2_bs256_8p_perf.log  # 8p training performance result log
    test/output/0/Srcnn_x2_bs256_8p_acc.log   # 8p training accuracy result log

## Srcnn_2 training result
使用动态lossscale会溢出

|   名称  | 精度(PSNR)| 性能(FPS) | 
| :---:  | :----:| :--------: | 
| GPU-1P |   -   | 4060.2765  |
| GPU-8P |   -   | 12944.0177 |
| NPU-1P |   -   | 5736.3541  |
| NPU-8P | 36.62 | 13953.0316 | 


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md
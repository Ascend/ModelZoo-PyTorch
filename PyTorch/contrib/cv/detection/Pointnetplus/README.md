# Pointnetplus

This implements training of Pointnetplus on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).

## Pointnetplus Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.Therefore, Pointnetplus is re-implemented using semantics such as custom OP.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip3 install -r requirements.txt`
- Download the ImageNet dataset from (https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) 

## Training

To train a model, run `train_classification_xP.py` with the desired model architecture and the path to the ImageNet dataset:
 then add [24,512,32,3],[24,1,128,259] transpose op to white list .pth is "/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/opp/op_impl/built-in/ai_core/tbe/impl/dynamic/transpose.py"
```bash
# training 1p accuracy
bash ./test/train_full_1p.sh --data=real_data_path

# training 1p performance
bash ./test/train_performance_1p.sh --data=real_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data=real_data_path

# finetuning 1p 
bash test/train_finetune_1p.sh --data=real_data_path --model_pth=real_pre_train_model_path
```

Log path:
    test/output/devie_id/train_${ASCEND_DEVICE_ID}_${Network}.log # training detail log
    test/output/devie_id/train_${ASCEND_DEVICE_ID}_${Network}.log  # 8p training performance result log
    test/output/devie_id/Pointnetplus_bs24_1p_acc.log   # 8p training accuracy result log



## WideResnet50_2 training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 10.04     | 1        | 1        | O2       |
| 92.0     | 64.1      | 8        | 200      | O2       |
# GENet

This implements training of GENET on the cifar10 dataset, mainly modified from [pytorch/examples](https://github.com/BayesWatch/pytorch-GENet).

## GENet Details

The configuration process and operation method of GENet are described in detail below.

## SoftWare Package
CANN 5.0.2

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the cifar10 dataset by referring the original [repository](https://github.com/BayesWatch/pytorch-GENet)
    - You can also without downloading them in advance. The cifar10 interface provided by torchvision will automatically download them for you.

## Training

To train a model, run `train.py` with the desired model architecture and the path to the cifar10 dataset.
Note: assuming that your dataset path is:**/opt/gpu/cifar10-batches-py/**, the real_data_path should be **/opt/gpu/**

```bash
# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=real_data_path

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training 1p with pretrained model
bash ./test/train_finetune_1p.sh --data_path=real_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path

# evaluating 8p performance
bash ./test/train_eval_8p.sh --data_path=real_data_path
```
### Remarks:
All bash instructions output log files correctly.

### Log path:
**training detail log:**
 test/output/devie_id/train_${device_id}.log           
**8p training performance result log:** test/output/devie_id/GENet_bs128_8p_perf.log
**8p training accuracy result log :** test/output/devie_id/GENet_bs128_8p_acc.log   

## GENET training result

| Acc@1    | FPS       | Device Type| Device Nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |:------:
|   94.73     | 1894.827   |   NPU | 1        | 300        | O2       |
| 95.23   | 7858.025     |NPU  |8       | 300      | O2       |
| 94.76    |  1350.074  |GPU  |1       | 300      | O2       |
| 94.81    |  6536.289  |GPU  |8       | 300      | O2       |
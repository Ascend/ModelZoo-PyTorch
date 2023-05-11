# 3D HRNet

note
- please download data from origin repo if necessary:
- https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR/data


This implements training of HRnet on the Cityscapes dataset, mainly modified from [pytorch/examples](https://github.com/HRNet/HRNet-Semantic-Segmentation).

## 3D HRNet Detail

The configuration process and operation method of 3D HRNet are described in detail below.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- The models are initialized by the weights pretrained on the ImageNet.You can download the pretrained models by BaiduYun [hrnetv2_w48_imagenet_pretrained.pth code:68g2](https://pan.baidu.com/s/13b8srQn8ARF9zHsaxvpRWA) 
- `mkdir pretrained_models` and put the pretrained models into this directory.
- Download the Cityscapes dataset from https://www.cityscapes-dataset.com/ 
  - your directory tree should be look like this:
  ```c
  ${data_path}
  |-- cityscapes
      |-- gtFine
      |    |-- test
      |    |-- train
      |    |-- val
      |-- leftImg8bit
           |-- test
           |-- train
           |-- val  

## Training

To train a model, run `train_npu.py` with the desired model architecture and the path to the Cityscapes dataset:

```bash
# training 1p performance
bash test/train_performance_1p.sh --data_path=real_data_path

# training 8p performance
bash test/train_performance_8p.sh --data_path=real_data_path

# training 8p accuracy
bash test/train_full_8p.sh --data_path=real_data_path

# finetuning 1p 
bash test/train_finetune_1p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path

# online inference demo
python3 tools/demo.py --data_path=real_data_path --pth_path=model_pth
```

Log path:
- `test/output/devie_id/train_${device_id}.log`           # training detail log
- `test/output/devie_id/train_${device_id}_8p_perf.log`             # 8p training performance result log
- `output/cityscapes/seg_hrnet*/*.pth`                    # training saved model



## 3D HRNet training result

| mIou     | FPS      | Device Type |Device nums | Epochs   | AMP_Type |
| :------: | :------: | :------:    |  :------:  | :------: | :------: |
| 81.04    |   28     |   NPU       |     8      |   484    |    O1    |
| 81.26    |   42     |   GPU       |     8      |   484    |    O1    |

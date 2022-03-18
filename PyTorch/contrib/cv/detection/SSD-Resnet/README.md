# #SSD-Resnet34

This implements training of SSD-Resnet34 on the 2017 COCO dataset.



## #SSD-Resnet34 Detail

On the basis of resnet34, a part of the feature layer is added for single target detection. 

## Requirements

* Install Pytorch==1.5.0 and torchvision 

* Install requirements.txt

* Steps to download pretrain-pth

  ```
  wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E8%AE%AD%E7%BB%83/cv/image_object_detection/SSD-ResNet34/resnet34-333f7ec4.pth
  ```

* Steps to download data

  ```
  source download_dataset.sh
  ```

  

# Training

### To train a model, run `training.py` with the desired model architecture and the path to the coco dataset:

```
# training 1p accuracy
cd ./test
bash train_full_1p.sh --data_path=real_data_path
# training 1p performance
cd ./test
bash train_performance_1p.sh --data_path=real_data_path
# training 8p accuracy
cd ./test
bash train_full_8p.sh --data_path=real_data_path
# training 8p performance
cd ./test
bash train_performance_8p.sh --data_path=real_data_path
#test 8p accuracy
bash test/train_eval_8p.sh --data_path=real_data_path --checkpoint_path=real_pre_train_model_path
```

Log path:
test/output/{device_id}/train_{device_id}.log
test/output/{device_id}/train_performance_{device_id}.log 

## SSD-Resnet34 training result

| Acc@1  | FPS  | Npu_nums | Epochs | AMP_Type |
| ------ | ---- | -------- | ------ | -------- |
| -      | 265  | 1        | 1      | O2       |
| 0.2301 | 1700 | 8        | 90     | O2       |
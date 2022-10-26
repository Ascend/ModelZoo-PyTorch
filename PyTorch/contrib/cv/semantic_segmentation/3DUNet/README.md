# 3DUNet
This implements training of 3DUNet on the Brats2018 dataset, mainly modified from [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch).

## Requirements
- Install Ascend PyTorch
- `pip install -r requirements.txt`
- Download the BraTS2018 dataset

## Training
```
#### 1p train perf
bash test/train_performance_1p.sh   --data_path=./datasets

#### 8p train perf
bash test/train_performance_8p.sh   --data_path=./datasets

#### 8p train full
bash test/train_full_8p.sh          --data_path=./datasets
```
--data_path以用户数据集实际存放路径为准。


## 3DUnet training result

| Name     | Dsc      | FPS       |
| :------: | :------: | :------:  | 
| NPU-1p   | -        | 18        | 
| NPU-8p   | 66       | 148       | 

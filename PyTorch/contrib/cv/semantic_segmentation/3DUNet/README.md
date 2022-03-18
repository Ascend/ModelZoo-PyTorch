# 3DUNet
This implements training of 3DUNet on the Brats2018 dataset, mainly modified from [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch).

## 3DUnet Detail
1. Until CANN 5.0.3, In order to make transpose op more efficient, you need to add a shape into white list to the following files:
```python
# open op script
vim usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/impl/dynamic/transpose.py 
# add shape at the end of line 160 whithin _by_dynamic_static_union_version function
[8,64,64,64,4]
```
2. FrameworkPTAdapter version should > 2.0.3.tr5.2021120812 to solve accuracy problem.

## Requirements
- Install Ascend PyTorch
- `pip install -r requirements.txt`
- Download the BraTS2018 dataset

## Training
#### 1p train perf
bash test/train_performance_1p.sh

#### 8p train perf
bash test/train_performance_8p.sh 

#### 8p train full
bash test/train_full_8p.sh 


## 3DUnet training result

| Name     | Dsc      | FPS       |
| :------: | :------: | :------:  | 
| NPU-1p   | -        | 18        | 
| NPU-8p   | 66       | 148       | 

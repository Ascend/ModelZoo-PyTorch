## PointNet

This implements training of AlignedReID on the a subset of shapenet dataset.
- Reference implementation
```
url=https://github.com/fxia22/pointnet.pytorch
branch=master
commit_id=f0c2430b0b1529e3f76fb5d6cd6ca14be763d975
```
### Requirements
- Install PyTorch (pytorch.org)
- pip install -r requirements.txt
- Download a subset of shapenet, you can use the following command:
```shell
bash test/download.sh
```
### Training
To train a model, run train_1p.py and train_8p.py with the desired model architecture and the path to the subset of shapenet:
```shell
# training 1p performance
bash test/train_performance_1p.sh --data_path=real_data_path

# training 8p performance
bash test/train_performance_8p.sh --data_path=real_data_path

# training 8p accuracy, pth file will be saved in the current path
bash test/train_full_8p.sh --data_path=real_data_path

#test 8p accuracy
bash test/train_eval_8p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path 

# finetuning 1p, input other cunstomed pkl file by adding pkl_path parameter to test finetuning function
bash test/train_finetune_1p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path --num_classes=num_classes

# online inference demo 
python3 demo.py
```

### PointNet training result

Accuracy|FPS|Npu_nums|Epochs|AMP_Type
-|-|-|-|-
97.39|225|1|80|O1
97.80|1615|8|80|O1

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
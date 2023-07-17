# PointRend

- 参考实现：
```
url=https://github.com/facebookresearch/detectron2
branch=master 
commit_id=f5e4c452bba854b8ed14a7240f51720ca7857e91
```

## PointRend Detail

- 增加了混合精度训练
- 增加了多卡分布式训练
- 优化了loss在NPU上的计算效率

## Requirements

- CANN 5.0.2及对应版本的PyTorch
- `pip3.7 install -r requirements.txt`

安装完其他依赖后，请使用代码仓中的源码编译安装detectron2：
```
source test/env_npu.sh
python3 -m pip install -e .
```


## 准备数据集
创建一个存放数据集的文件夹（如/root/datasets/cityscapes），再从cityscapes官网获取gtFine_trainvaltest.zip和leftImg8bit_trainvaltest.zip，将这两个压缩包解压到创建的文件夹中，随后参考detectron2官网的方法创建labelTrainIds.png ：
```
python3 createTrainLabelIds.py /root/datasets/cityscapes
```

## 准备预训练权重
```
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E8%AE%AD%E7%BB%83/cv/semantic_segmentation/PointRend/R-101.pkl
```
脚本默认权重文件位于工程根目录下，如果放在其他路径请在之后运行脚本时指定权重路径。
## Training


```
source test/env_npu.sh
# 1p train perf
bash test/train_performance_1p.sh --data_path=数据集路径(如/root/datasets) --pth_path=权重路径（非必需） --output_path=输出文件路径（非必需，默认为./output）

# 8p train perf
bash test/train_performance_8p.sh --data_path=数据集路径(如/root/datasets) --pth_path=权重路径（非必需） --output_path=输出文件路径（非必需，默认为./output）

# 1p train full
bash test/train_full_1p.sh --data_path=数据集路径(如/root/datasets) --pth_path=权重路径（非必需） --output_path=输出文件路径（非必需，默认为./output）

# 8p train full
bash test/train_full_8p.sh --data_path=数据集路径(如/root/datasets) --pth_path=权重路径（非必需） --output_path=输出文件路径（非必需，默认为./output）

# eval
bash test/train_eval_8p.sh --data_path=数据集路径(如/root/datasets) --pth_path=权重路径（非必需） --output_path=输出文件路径（非必需，默认为./output）

# finetuning
bash test/train_finetune_1p.sh --data_path=数据集路径(如/root/datasets) --pth_path=权重路径（非必需） --output_path=输出文件路径（非必需，默认为./output）



## PointRend training result

|IOU       | FPS       | Npu_nums | Steps    | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| 78.08    | 8.47      | 1        | 65000    | O1       |
| 77.58    | 29.01     | 8        | 65000    | O1       |

```

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
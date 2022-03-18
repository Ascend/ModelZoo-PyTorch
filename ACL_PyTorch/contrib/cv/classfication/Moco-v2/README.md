# MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
This implements inference of [MoCo v2](https://github.com/facebookresearch/moco) on the Imagenet dataset.

## Requirements
- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

## 推理步骤

### 下载模型
```
https://github.com/Lucas-Wye/moco/releases/download/model/model_lincls_best.pth.tar
```
模型精度信息：Acc@1 67.589 Acc@5 87.990

### 处理数据集

使用[ImageNet](http://www.image-net.org/)的5万张验证集进行测试，图片与标签分别存放在/root/datasets/imagenet/val与/root/datasets/imagenet/val_label.txt
```
mkdir -p prep_dataset
python3.7 imagenet_torch_preprocess.py datasets/ImageNet/val_union ./prep_dataset
python3.7 get_info.py bin ./prep_dataset ./dataset_prep_bin.info 224 224
```

### 离线推理
[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  
将benchmark.x86_64或benchmark.aarch64放到当前目录 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  

```
bash test/pth2om.sh
bash test/eval_acc_perf.sh --datasets_path=/root/datasets 
```

**评测结果：**   
| 模型      | pth精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| Moco v2 bs1  | Acc@1 67.589, Acc@1 67.589 Acc@5 87.990 | Acc@1 67.3, Acc@5 87.81% |  897.328fps | 1619.264fps | 
| Moco v2 bs16 | Acc@1 67.589, Acc@1 67.589 Acc@5 87.990 | Acc@1 67.3, Acc@5 87.81% | 1849.666fps | 2422.756ps | 

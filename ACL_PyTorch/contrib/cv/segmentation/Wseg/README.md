# Wseg模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，并且安装以下可能已经安装过的需求包
```
pip3.7 install -r requirements.txt  
```

2.获取开源模型的代码,并修改文件夹名称为wseg
```
git clone https://github.com/visinf/1-stage-wseg -b master   
git reset cfe5784f9905d656e0f15fba0e6eb76a3731d80f --hard
mv 1-stage-wseg wseg
```

3.获取权重文件
1. 获取经过预训练的基础网络权重文件并且放在代码仓的以下路径中：`<project>/models/weights/`.

    | Backbone | Initial Weights |
    |:---:|:---:|
    | WideResNet38 | [ilsvrc-cls_rna-a1_cls1000_ep-0001.pth (402M)](https://download.visinf.tu-darmstadt.de/data/2020-cvpr-araslanov-1-stage-wseg/models/ilsvrc-cls_rna-a1_cls1000_ep-0001.pth) |
      
2. 获取功能网络权重（作者提供的pth模型）并放置于代码仓的以下路径中：（初始代码仓无snapshots文件夹，需要自己新建路径）`<project>/snapshots/`
    
    | Backbone | Val | Link |
    |:---:|:---:|---:|
    | WideResNet38 | 62.7 |  [model_enc_e020Xs0.928.pth (527M)](https://download.visinf.tu-darmstadt.de/data/2020-cvpr-araslanov-1-stage-wseg/models/model_enc_e020Xs0.928.pth) |

3. 移动上述两个权重文件到代码仓指定位置，以待加载使用
```
mkdir ./models/weights
mv ilsvrc-cls_rna-a1_cls1000_ep-0001.pth ./models/weights
mkdir ./snapshots
mv model_enc_e020Xs0.928.pth ./snapshots
```

4.下载数据集，解压，将文件名改为voc，并将其放于代码仓中的以下路径： `<project>/data/`    
- VOC: [Training/Validation (2GB .tar file)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
    
```
tar -zxvf VOCtrainval_11-May-2012.tar
mv VOCtrainval_11-May-2012 voc
mkdir ./data
mv voc ./data
```

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  
将benchmark.x86_64或benchmark.aarch64放到当前目录  

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  
```
bash test/pth2om.sh
bash test/eval_acc_perf.sh --datasets_path=./data
```
 **评测结果：** 
   
| 模型      | 官网pth精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| WideResNet38 bs1 | [IOU: 62.7](https://github.com/visinf/1-stage-wseg) | IOU:63.7 | 5.270fps | 3.496fps | 
| WideResNet38 bs4 | [IOU: 62.7](https://github.com/visinf/1-stage-wseg) | IOU:63.7 | 5.460fps | 3.912fps | 

 **备注：** 
- 由于分辨率大内存使用多，故仅用bs1与bs4进行评测
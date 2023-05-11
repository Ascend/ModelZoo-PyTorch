# FairMOT

- 参考实现：
```
url=https://github.com/ifzhang/FairMOT
branch=master 
commit_id=815d6585344826e0346a01efd57de45498cfe52b
```

## FairMOT Detail

- 增加了混合精度训练
- 增加了多卡分布式训练
- 优化了loss在NPU上的计算效率

## Requirements

- CANN 5.0.2及对应版本的PyTorch
- `pip install -r requirements.txt`
依赖里有torchvision==0.6.0,如果在arm64上运行，需要手动安装


## 准备数据集
首先创建一个数据集目录dataset,数据集放在这个文件夹下。
- 下载[MOT17数据集](https://motchallenge.net/data/MOT17.zip)
    - 下载得到MOT17.zip 解压，然后将数据集处理成如下的文件结构
    ```
    MOT17
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train(empty)
    ```
  接下来需要生成标注文件，需要先修改/FairMOT/src/gen_labels_16.py
  将这个文件的seq_root 修改为dataset文件夹的目录+'/MOT17/images/train' 例如：/root/dataset/MOT17/images/train

  然后将label_root 修改为dataset文件夹的目录+'MOT16/labels_with_ids/train' 例如/root/dataset/MOT17/labels_with_ids/train
  然后执行 
  ```
  python3 gen_labels_16.py
  ```
  下载https://github.com/ifzhang/FairMOT模型， 将刚下载下的FairMOT/src下的data文件夹放至本模型的src目录下

## 准备预训练权重
下载[DLA-34 official] 下载链接参考源码仓。

然后放到/FairMOT/models/文价夹下

## Training


```bash
source test/env_npu.sh
# 1p train perf
bash test/train_performance_1p.sh --data_path=数据集路径 如/root/dataset

# 8p train perf
bash test/train_performance_8p.sh --data_path=数据集路径 如/root/dataset

# 8p train full  跑完之后再跑 bash test/train_eval_8p.sh --data_path=数据集路径 可以得到精度数据
bash test/train_full_8p.sh --data_path=数据集路径 如/root/dataset

# eval
bash test/train_eval_8p.sh --data_path=数据集路径 如/root/dataset

# finetuning
bash test/train_finetune_1p.sh --data_path=数据集路径 如/root/dataset



## FairMOT training result

|MOTA      | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| 90       | 3.8       | 1        | 30       | O1       |
| 84.8     | 28       | 8        | 30       | O1       |

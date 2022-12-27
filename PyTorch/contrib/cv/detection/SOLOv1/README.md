# SOLOv1

This implements training of SOLOv1 on the Coco dataset, mainly modified from [pytorch/examples](https://github.com/WXinlong/SOLO).

## SOLOv1 Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.
Therefore, SOLOv1 model need to be modified in the following aspects:

1. Converting tensors with the dynamic shapes into tensors with fixed shapes. (This is the hardest one)
2. Several operations, like the sum of `INT64`, are not supported on the NPU, so we modified tensors' `dtype` when needed
3. Framework bottlenecks lead to poor performance, so we improve the original code to improve the performance of the model
4. We used Apex for mmdtection due to the hardware defects of the NPU
5. ...


## Requirements

- NPU配套的run包安装
- Python 3.7.5
- PyTorch(NPU版本)
- apex(NPU版本)
- MMCV v0.2.16
### Document and data preparation
1. 下载压缩modelzoo\contrib\PyTorch\cv\instance_segmentation\SOLOv1 文件夹
2. 于npu服务器解压SOLOv1压缩包
3. 下载coco2017数据集
4. 将coco数据集放于SOLOv1/data目录下，目录结构如下：
```
GFocalV2
├── configs
├── data
│   ├── coco
│       ├── annotations   796M
│       ├── train2017     19G
│       ├── val2017       788M
```
### Configure the environment
```
进入SOLOv1目录下，source环境变量
cd SOLOv1
source test/env_npu.sh  
```
1. 配置安装mmcv
```
cd mmcv
python3.7 setup.py build_ext
python3.7 setup.py develop
cd ..
pip3 list | grep mmcv  # 查看版本和路径
``` 
2. 配置安装mmdet
```
pip install -r requirements/build.txt
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install -v -e .
```
## Train MODEL
进入SOLOv1目录下
### 1p
导入环境变量，修改train_full_1p.sh权限并运行
```
chmod +x ./test/train_full_1p.sh
bash ./test/train_full_1p.sh --data_path=./data/coco
```

### 8p
导入环境变量，修改train_full_8p.sh权限并运行
```
chmod +x ./test/train_full_8p.sh
bash ./test/train_full_8p.sh --data_path=./data/coco
```

### Eval
修改train_eval_1p.sh权限并运行
```
chmod +x ./test/train_eval_1p.sh
bash ./test/train_eval_1p.sh --data_path=./data/coco
```
### finetuning
修改train_finetune_1p.sh权限并运行
```
chmod +x ./test/train_eval_1p.sh
bash ./test/train_finetune_1p.sh --data_path=./data/coco
```
### 多机多卡性能数据获取流程

```shell
     1. 安装环境
     2. 开始训练，每个机器所请按下面提示进行配置
             bash ./test/train_performance_multinodes.sh  --data_path=数据集路径 --batch_size=单卡batch_size*所有卡数 --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
```
## SOLOv1 training result 

| Acc@1    | FPS       | Npu/Gpu_nums | Epochs   | AMP_Type | Loss_Scale |
| :------: | :------:  | :------:     | :------: | :------: | :------:   |
|          | 2.75      | 1p Gpu       | 1        | O1       | 128.0      |
| 32.3     | 16.3      | 8p Gpu       | 12       | O1       | 128.0      |
|          | 1.42      | 1p Npu       | 1        | O1       | 128.0      |
| 32.1     | 9.4       | 8p Npu       | 12       | O1       | 128.0      |

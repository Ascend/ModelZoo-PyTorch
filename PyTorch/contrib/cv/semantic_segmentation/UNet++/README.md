# UNet++

This implements training of UNet on the 2018 Data Science Bowl dataset, mainly modified from [UNet++](https://github.com/4uiiurz1/pytorch-nested-unet).

## UNet Detail 

For details, see [UNet++](https://github.com/4uiiurz1/pytorch-nested-unet).


## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))

- pip install -r requirements.txt

- 若遇到module 'cv2' has  no attribute '_registerMatType' 报错，可按照以下步骤解决


	  pip install -r requirements.txt
	  pip uninstall opencv-python
	  pip uninstall opencv-python-headless
	  pip install opencv-python-headless==4.6.0.66
	  pip install tqdm
	  pip install pillow==6.2.2

- get dataset from [data-science-bowl-2018](https://www.kaggle.com/c/data-science-bowl-2018/data).The file structure is the following:
```
── data-science-bowl-2018
    ├── stage1_train
    |   ├── 00ae65...
    │   │   ├── images
    │   │   │   └── 00ae65...
    │   │   └── masks
    │   │       └── 00ae65...            
    │   ├── ...
    |
    ...
```
- the data-science-bowl-2018 dataset need preprocess. 
```bash
python3 preprocess_dsb2018.py  --data_dir=./data-science-bowl-2018
数据会保存在./data-science-bowl-2018文件夹下
```

## Training

一、训练流程

单卡训练流程：

    1.安装环境  
    2.开始训练
        bash ./test/train_full_1p.sh  --data_path=数据集路径         # 精度训练
        eg: bash ./test/train_full_1p.sh --data_path=./inputs/dsb2018_96/
        bash ./test/train_performance_1p.sh  --data_path=数据集路径  # 性能训练


​    
多卡训练流程

    1.安装环境
    2.开始训练
        bash ./test/train_full_8p.sh  --data_path=数据集路径         # 精度训练
        eg:bash ./test/train_full_8p.sh --data_path=./inputs/dsb2018_96/
        bash ./test/train_performance_8p.sh  --data_path=数据集路径  # 性能训练

单卡或多卡在线评估：

		bash  ./test/eval.sh   数据集路径    模型文件路径

		eg: bash ./test/eval.sh ./inputs/dsb2018_96/ ./models/dsb2018_96_NestedUNet_woDS/model_best.pth.tar

demo在线评估：

		python demo.py --data_path=./inputs/dsb2018_96
		

二、Docker容器训练

1.导入镜像二进制包docker import ubuntuarmpytorch.tar REPOSITORY:TAG, 比如:

    docker import ubuntuarmpytorch.tar pytorch:b020

2.执行docker_start.sh后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：


    ./docker_start.sh pytorch:b020 /train/peta /home/DeepMar

3.执行步骤一训练流程（环境安装除外）



三、测试结果
训练日志路径：网络脚本test下output文件夹内。例如：
      test/output/devie_id/train_${device_id}.log          # 训练脚本原生日志
      test/output/devie_id/UNet++_bs1024_8p_perf.log  # 8p性能训练结果日志
      test/output/devie_id/UNet++_bs1024_8p_acc.log   # 8p精度训练结果日志

训练模型：训练生成的模型默认会写入到和test文件同一目录下。当训练正常结束时，checkpoint.pth.tar为最终结果。

### UNet++ training result

|  IOU  | FPS  | Npu_nums | Epochs | AMP_Type |
| :---: | :--: | :------: | :----: | :------: |
|   -   |  -   |    1     |  100   |    O2    |
| 83.59 |  -   |    8     |  100   |    O2    |
# SETR

- 参考实现：
```
url=https://github.com/fudan-zvg/SETR 
branch=master 
commit_id=23f8fde88182c7965e91c28a0c59d9851af46858
```

## SETR_Naive Detail

- 采用原仓FP16实现混合精度训练
- 迁移了多卡分布式训练

## Requirements

- CANN 5.0.3.1
- torch 1.5.0+ascend.post3.20210930
- apex 0.1+0.1+ascend.20210930
- tensor_fused_plugin、te、topi
- python3.7.5
- 2to3 , 使用"apt install 2to3"安装

- 编译安装mmcv1.2.7版本

```
source env_npu.sh 
cd SETR
git clone git@github.com:open-mmlab/mmcv.git
cd mmcv
git checkout v1.2.7
cd ..

# 用mmcv-need文件替换替换mmcv中的对应文件
cp -f mmcv-need/_functions.py mmcv/mmcv/parallel/_functions.py
cp -f mmcv-need/scatter_gather.py mmcv/mmcv/parallel/scatter_gather.py
cp -f mmcv-need/distributed.py mmcv/mmcv/parallel/distributed.py
cp -f mmcv-need/optimizer.py mmcv/mmcv/runner/hooks/optimizer.py
cp -f mmcv-need/iter_based_runner.py mmcv/mmcv/runner/iter_based_runner.py
cp -f mmcv-need/fp16_utils.py mmcv/mmcv/runner/fp16_utils.py
cp -f mmcv-need/dist_utils.py mmcv/mmcv/runner/dist_utils.py
cd mmcv
MMCV_WITH_OPS=True python3 setup.py build_ext --inplace
# 报错的话可能需要升级下pip版本
MMCV_WITH_OPS=1 pip3 install -e .
cd ..
```

- 安装SETR所需要的环境

```
pip3 install -e .  
# 若出现红字jpeg error，可能需要安装下 yum -y install libjpeg-turbo-devel
pip3 install -r requirements/optional.txt
pip3 install torchvision==0.2.0
```

### 配置数据集路径

cityscape数据集

参考源码仓的方式获取数据集：
https://github.com/fudan-zvg/SETR

新下载的数据集需要进行格式转换，产生_labelTrainIds.png后缀的输入文件

```
python3 tools/convert_datasets/cityscapes.py ./data/cityscapes --nproc 8 
```


### 配置预训练模型

本模型是在预训练模型vit-large的基础上进行训练的，需要将vit模型下载到本地进行训练。

获取地址：https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth

```
SETR
├── pre_train_models
│   ├── jx_vit_large_p16_384-b3be5167.pth
```

### NPU 1卡训练指令

结果会保存在SETR/work_dirs文件夹中。

```shell
# data_path 默认为'data/cityscapes/'
bash test/train_full_1p.sh --data_path='data/cityscapes/'
```

性能

```shell
bash test/train_performance_1p.sh --data_path='data/cityscapes/'
```

### NPU 8卡训练指令

训练
```shell
# data_path 默认为'data/cityscapes/'
bash test/train_full_8p.sh --data_path='data/cityscapes/'
```
评估
```shell
# data_path 默认为""
bash test/train_eval_8p.sh --data_path='data/cityscapes/' --check_point_file_name='work_dirs/training_npu_1p_job_20211120043053'
# “work_dirs/training_npu_1p_job_20211120043053”为保存checkpoint的文件夹，自动会处理文件中的latest.pth文件
```

性能

```shell
bash test/train_performance_8p.sh --data_path='data/cityscapes/'
```

### Demo

```
python demo.py --img='munster_000139_000019_leftImg8bit.png' --checkpoint='work_dirs/77.pth' 
```

```
python demo.py --img='dtk.jpg' --checkpoint='work_dirs/77.pth' 
```

### FAQ

- 模型训练到1个epoch时，突然出现import mmcv错误，程序随即停止训练？

  答：在新环境上可能会出现这样的问题。重新编译安装下mmcv就不会了。


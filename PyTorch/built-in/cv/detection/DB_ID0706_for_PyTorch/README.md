# db模型使用说明

## 1. Requirements
* NPU配套的run包安装(C20B030)

* Python 3.7.5

* PyTorch(NPU版本)

* apex(NPU版本)

* 安装geos，可按照环境选择以下方式：

  1. ubuntu系统：

     ```
     sudo apt-get install libgeos-dev
     ```

  2. euler系统：

     ```
     sudo yum install geos-devel
     ```

  3. 源码安装：

     ```
     wget http://download.osgeo.org/geos/geos-3.8.1.tar.bz2
     bunzip2 geos-3.8.1.tar.bz2
     tar xvf geos-3.8.1.tar
     cd geos-3.8.1
     ./configure && make && make install
     ```

     

- 安装python依赖包

  ```
  pip3.7 install -r requirements.txt
  ```

  

## 2. Models

下载预训练模型MLT-Pretrain-Resnet50, [Google Drive](https://drive.google.com/open?id=1T9n0HTP3X3Y_nJ0D1ekMhCQRHntORLJG). 放在文件夹path-to-model-directory下。

```
__ path-to-model-directory
  |__ MLT-Pretrain-ResNet50
```



## 3. Dataset Prepare

下载icdar2015数据集，放在文件夹datasets下。

```
__ datasets
  |__icdar2015
```



## 4. 1P

按需要编辑device_list，运行run1p.sh
以下是db的1p训练脚本

```
source env.sh
export PYTHONPATH=./:$PYTHONPATH
export TASK_QUEUE_ENABLE=0
export DYNAMIC_OP="ADD"
python3.7 train.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml \
        --resume path-to-model-directory/MLT-Pretrain-ResNet50 \
        --data_path datasets/icdar2015/ \
        --seed=515 \
        --amp \
        --device_list "0"
```

**注意**：如果发现打屏日志有报checkpoint not found的warning，请再次检查第二章节的设置，以免影响精度。

## 5. 8P

运行run8p.sh

```
source env.sh
export PYTHONPATH=./:$PYTHONPATH
export TASK_QUEUE_ENABLE=0
export DYNAMIC_OP="ADD"
python3.7 -W ignore train.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml \
        --resume path-to-model-directory/MLT-Pretrain-ResNet50 \
        --data_path datasets/icdar2015/ \
        --seed=515 \
        --distributed \
        --device_list "0,1,2,3,4,5,6,7" \
        --num_gpus 8 \
        --local_rank 0 \
        --dist_backend 'hccl' \
        --world_size 1 \
        --batch_size 128 \
        --lr 0.056 \
        --addr $(hostname -I |awk '{print $1}') \
        --amp \
        --Port 29502 \
```

## 6. eval_precision

模型结束后，要测试训练结果的精度，可执行eval_precision.sh

```
python3.7 eval.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume outputs/workspace/${PWD##*/}/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/final --box_thresh 0.6
```


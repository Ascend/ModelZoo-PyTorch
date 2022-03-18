# 目标检测与目标分割类模型使用说明

## Requirements
* NPU配套的run包安装(C20B030)
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)


## Dataset Prepare
1. 下载COCO数据集，放在datasets中。如已有下载可通过设置环境变量DETECTRON2_DATASETS=“coco所在数据集路径”进行设置，如export DETECTRON2_DATASETS=/home/sample，则coco数据集放在/home/sample目录中

## Pre-Train Weights File 
1. 模型脚本会自动下载预训练权重文件。若下载失败，请自行准备R-101.pkl等权重文件，同时修改相应配置文件中预训练权重文件路径。

### Build Detectron2 from Source

编译器版本：gcc & g++ ≥ 5
```
python3.7 -m pip install -e Faster_Mask_RCNN_for_PyTorch

```
在重装PyTorch之后，通常需要重新编译detectron2。重新编译之前，需要使用`rm -rf build/ **/*.so`删除旧版本的build文件夹及对应的.so文件。

## 1P
1. 编辑并运行run.sh
以下是mask_rcnn的训练脚本
```
source ./env_b031.sh
source ./env_new.sh
export PYTHONPATH=./:$PYTHONPATH
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
export DYNAMIC_OP="ADD#MUL"
/usr/local/Ascend/driver/tools/msnpureport -g error
/usr/local/Ascend/driver/tools/msnpureport -e disable

nohup python3.7 tools/train_net.py \                                                  # 运行入口文件
        --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml \ # 选择mask_rcnn的模型yaml文件
        AMP 1\                                                                        # 是否使用apex优化训练
        OPT_LEVEL O2 \                                                                # apex优化级别
        LOSS_SCALE_VALUE 64 \ 
        MODEL.DEVICE npu:5 \                                                          # 指定对应的device_id
        SOLVER.IMS_PER_BATCH 8 \                                                      # 设置batch_size
        SOLVER.MAX_ITER 82000 \                                                       # 设置训练iter数
        SEED 1234 \                                                                   # 随机数种子
        MODEL.RPN.NMS_THRESH 0.8 \                                                    # NMS 阈值
        MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO 2 \                                  # proposal采样率
        MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO 2 \
        DATALOADER.NUM_WORKERS 4 \                                                    # 图片读取进程数
        SOLVER.BASE_LR 0.0025 &                                                       # 初始学习率 

```
若将以上的--config-file参数指定为：configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml，则可训练对应的faster_rcnn模型。其他参数含义一致。

## 8P
1. 编辑并运行run8p.sh

```
source ./env_b031.sh
source ./env_new.sh
export PYTHONPATH=./:$PYTHONPATH
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
export DYNAMIC_OP="ADD#MUL"
/usr/local/Ascend/driver/tools/msnpureport -g error
/usr/local/Ascend/driver/tools/msnpureport -e disable

nohup python3.7 tools/train_net.py \
        --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml \
        --device-ids 0 1 2 3 4 5 6 7 \
        --num-gpus 8\
        AMP 1\
        OPT_LEVEL O2 \
        LOSS_SCALE_VALUE 64 \
        SOLVER.IMS_PER_BATCH 64 \
        SOLVER.MAX_ITER 10250 \
        SEED 1234 \
        MODEL.RPN.NMS_THRESH 0.8 \
        MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO 2 \
        MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO 2 \
        DATALOADER.NUM_WORKERS 8 \
        SOLVER.BASE_LR 0.02 &

```
若将以上的--config-file参数指定为：configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml，则可训练对应的faster_rcnn模型。其他参数含义一致。

## Docker容器训练：

1.导入镜像二进制包docker import ubuntuarmpytorch_maskrcnn.tar REPOSITORY:TAG, 比如:

    docker import ubuntuarmpytorch_maskrcnn.tar pytorch:b020_maskrcnn
2.执行docker_start.sh 后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：

    ./docker_start.sh pytorch:b020_maskrcnn /train/coco /home/MaskRCNN


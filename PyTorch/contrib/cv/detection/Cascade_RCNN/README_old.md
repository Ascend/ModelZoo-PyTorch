# CascadeRCNN

本项目实现了 CascadedRCNN 在 NPU 上的训练, 迁移自 detectron2-0.2.1.
[detectron2 github链接](https://github.com/facebookresearch/detectron2)

## CascadeRCNN Detail

本项目对 detectron2-0.2.1 做了如下更改：
1. 迁移到 NPU 上
2. 使用混合精度训练、测试
3. 对于一些操作，固定动态 shape 、使用 NPU 算子优化性能、同时将一些操作转移到 CPU 上进行


## Requirements

- NPU 配套的 run 包安装(20211018)
- Python 3.7.5
- PyTorch(NPU20210930 版本)
- apex(NPU20210930 版本)
- torchvision 0.6.0
- decorator
- sympy
- 安装 detectron2
```bash
pip3.7 install torch-1.5.0+ascend.post3.20210930-cp37-cp37m-linux_x86_64.whl
pip3.7 install apex-0.1+ascend.20210930-cp37-cp37m-linux_x86_64.whl
pip3.7 install -r Cascade_RCNN/requirements.txt
source Cascade_RCNN/test/env_npu.sh
cd Cascade_RCNN
python3 setup.py build develop
```
- 下载 COCO 数据集，放在 datasets 中。如已有下载可通过设置环境变量DETECTRON2_DATASETS=“coco 所在数据集路径”进行设置，如 export DETECTRON2_DATASETS=/opt/npu/，则 coco 数据集放在 /opt/npu/ 目录中
- 下载预训练模型 R-101.pkl ,configs/COCO-Detection/cascade_rcnn_R_101_FPN_1x.yaml配置文件中MODEL.WEIGHTS 设置为R-101.pkl的绝对路径

## Training

```bash
cd Cascade_RCNN

# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=/opt/npu/

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=/opt/npu/

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=/opt/npu/

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=/opt/npu/

#test 8p accuracy
bash test/train_eval_8p.sh --data_path=/opt/npu/ --pth_path=./output/model_final.pth

# finetuning 1p
bash test/train_finetune_1p.sh --data_path=/opt/npu/ --pth_path=./output/model_final.pth
```

Log path:
    test/output/devie_id/train_device_id.log           # training detail log
    test/output/devie_id/CascadeRCNN_bs64_8p_perf.log  # 8p training performance result log
    test/output/devie_id/CascadeRCNN_bs64_8p_acc.log   # 8p training accuracy result log



## CascadeRCNN training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 5         | 1        | 0.07     | O2       |
| 42.445   | 42        | 8        | 24.34    | O2       |
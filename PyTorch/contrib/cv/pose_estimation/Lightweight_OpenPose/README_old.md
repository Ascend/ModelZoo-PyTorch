# Lightweight OpenPose

该目录为Lightweight_OpenPose在coco2017数据集上的训练与测试，主要参考实现[Daniil-Osokin/lightweight-human-pose-estimation.pytorch](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)

##  Lightweight OpenPose相关细节

* 添加DDP多卡训练代码
* 添加混合精度训练
* 添加训练过程中的验证代码，以保存训练过程中的最佳精度的模型
* 使用multi_epochs_dataloaders,解决模型在NPU上训练时，每epoch间卡顿的问题
* 增加模型精度验证脚本

## 环境准备

* 执行本样例前，请确保已安装有昇腾910处理器的硬件环境，CANN包版本CANN toolkit_5.1.RC1.alpha001
* 该目录下的实现是基于PyTorch框架，其中torch版本为torch 1.8.1+ascend.rc2.20220505，使用的混合精度apex版本apex 0.1+ascend.20220505，固件驱动版本22.0.0
* pip install -r requirements.txt

## 训练准备

* 获取数据集coco_2017
,解压到合适的目录(假设路径名为<coco_home>)下，应包含`train2017`，`val2017`,`test2017`,`annotations`四个文件夹。
* 获取预训练的[mobilenetv1权重文件](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E8%AE%AD%E7%BB%83/cv/pose_estimation/openpose/mobilenet_sgd_68.848.pth.tar) 
,该文件置于主目录下。

```shell
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E8%AE%AD%E7%BB%83/cv/pose_estimation/openpose/mobilenet_sgd_68.848.pth.tar
```
* 将训练标准文件转化为内部格式,在主目录下生成文件`prepared_train_annotation.pkl`
```shell
python3 scripts/prepare_train_labels.py --labels <coco_home>/annotations/person_keypoints_train2017.json
```
* 从完整的5000样本数量的验证集中随机生成一个样本量250的子集。在主目录下生成文件`val_subset.json`。
```shell
python3 scripts/make_val_subset.py --labels <coco_home>/annotations/person_keypoints_val2017.json
```

## 快速运行

模型的训练文件详见train.py, 运行以下脚本能够进行单/多卡的训练和性能测试:

```shell
# train 1p performance,结果位于主目录下文件夹“perf_1p_checkpoints”
bash test/train_performance_1p.sh --data_path=<coco_home>

# train 1p full,模型经过三步step训练，依次执行以下脚本
# step one,结果位于主目录下文件夹“step_one_checkpoints”
bash test/train_full_1p.sh --data_path=<coco_home> --step=1
#step two,结果位于主目录下文件夹“step_two_checkpoints”
bash test/train_full_1p.sh --data_path=<coco_home> --step=2
#step three,结果位于主目录下文件夹“step_three_checkpoints”
bash test/train_full_1p.sh --data_path=<coco_home> --step=3

# train 8p performance,结果位于主目录下文件夹“perf_8p_checkpoints”
bash test/train_performance_8p.sh --data_path=<coco_home>

# train 8p full,模型经过三步step训练，依次执行以下脚本
# step one,结果位于主目录下文件夹“step_one_checkpoints”
bash test/train_full_8p.sh --data_path=<coco_home> --step=1
#step two,结果位于主目录下文件夹“step_two_checkpoints”
bash test/train_full_8p.sh --data_path=<coco_home> --step=2
#step three,结果位于主目录下文件夹“step_three_checkpoints”
bash test/train_full_8p.sh --data_path=<coco_home> --step=3

# 验证各阶段的最佳模型的精度，依次执行以下脚本
# eval step one,结果位于主目录下文件夹“eval_step1”
bash test/eval.sh --data_path=<coco_home> --step=1 --device_id=0 --checkpoint_path=./step_one_checkpoints/model_best.pth
# eval step two,结果位于主目录下文件夹“eval_step2”
bash test/eval.sh --data_path=<coco_home> --step=2 --device_id=1 --checkpoint_path=./step_two_checkpoints/model_best.pth
# eval step three,结果位于主目录下文件夹“eval_step3”
bash test/eval.sh --data_path=<coco_home> --step=3 --device_id=2 --checkpoint_path=./step_three_checkpoints/model_best.pth
```
训练的脚本需要在前一步骤结束后再接着启动。因为依赖于前一步保存的模型。

验证的脚本使用单卡验证，所以训练完成后，可以分别启动三个脚本在不同卡上运行。单次验证时间约为3小时。

## Lightweight OpenPose的训练结果
8p-NPU上各阶段训练后的模型精度

| Acc@1    | step       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| 0.3973        | 1      | 8        | 280      | O1       |
| 0.4132     | 2     | 8        | 280      | O1      |
| 0.4289     | 3     | 8        | 280      | O1      |

1p-NPU和8P-NPU训练的step-3结果

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 216.209      | 1        | 1      | O1       |
| 0.4289     | 1800.749     | 8        | 280      | O1      |

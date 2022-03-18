# ShuffleNetV2Plus (size=Small)

## ImageNet training with PyTorch

This implements training of ShuffleNetV1 on the ImageNet dataset, mainly modified from [Github](https://github.com/pytorch/examples/tree/master/imagenet).

## ShuffleNetV2Plus Detail

Base version of the model from [the paper author's code on Github](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2%2B).
The training script is adapted from [the ShuffleNetV2 script on Gitee](https://gitee.com/ascend/modelzoo/tree/master/built-in/PyTorch/Official/cv/image_classification/Shufflenetv2_for_PyTorch).

## Requirements

- pytorch_ascend, apex_ascend, tochvision
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).

## Training
一、训练流程：
        
单卡训练流程：

    1.安装环境
    2.修改参数device_id（单卡训练所使用的device id），为训练配置device_id，比如device_id=0
    3.开始训练
        bash ./test/train_full_1p.sh  --data_path=数据集路径    # 精度训练
        bash ./test/train_performance_1p.sh  --data_path=数据集路径  # 性能训练
    
    
多卡训练流程

    1.安装环境
    2.修改参数device_id_list（多卡训练所使用的device id列表），为训练配置device_id，例如device_id=0,1,2,3,4,5,6,7
    3.执行train_full_8p.sh开始训练
        bash ./test/train_full_8p.sh  --data_path=数据集路径         # 精度训练
        bash ./test/train_performance_8p.sh  --data_path=数据集路径  # 性能训练
  
二、测试结果
    
训练日志路径：网络脚本test下output文件夹内。例如：

      test/output/devie_id/train_${device_id}.log           # 训练脚本原生日志
      test/output/devie_id/ShuffleNetV1_bs8192_8p_perf.log  # 8p性能训练结果日志
      test/output/devie_id/ShuffleNetV1_bs8192_8p_acc.log  # 8p精度训练结果日志

训练模型：训练生成的模型默认会写入到和test文件同一目录下。当训练正常结束时，checkpoint.pth.tar为最终结果。



## ShufflenetV2Plus training result

| Acc@1    | FPS       | Npu_nums| Epochs   | Type     |
| :------: | :------:  | :------ | :------: | :------: |
| 73.132   | 6306      | 8       | 360      | O2       |

备注：由于模型开发中发现NPU上clamp算子反向错误，以上结果为使用自行编写的clamp函数训练获得。见blocks.py中的注释掉的函数clamp。
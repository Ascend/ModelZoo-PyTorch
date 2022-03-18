# EfficientDet_D0-Pytorch

This implements training of EfficientDet on the coco dataset, mainly modified from https://github.com/rwightman/efficientdet-pytorch.

## Requirements

- Install PyTorch 1.5.0 ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the coco dataset from http://cocodataset.org/
    - 2017 Train images(http://images.cocodataset.org/zips/train2017.zip)
    - 2017 Val images(http://images.cocodataset.org/zips/val2017.zip)
    - 2017 Train/Val annotations(http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
    - coco数据集目录结构需满足:
      - coco
        - ├── annotations
        - ├── train2017
        - └── val2017
    - 下载完毕后，需要将test文件夹中运行脚本的`data_path`参数路径自行修改为coco数据集所在路径
  
- ，timm版本为0.4.12，其中timm源码有修改，需要替换timm中以下文件，修改后的文件可从timm_modify文件夹中复制替换到对应位置：
  - timm
    - ├── models
      - ├── layers
        - ├── conv2d_same.py
        - ├── pool2d_same.py
        - ├── padding.py
        - └── activations_me.py 
    - ├── optim
      - ├── optim_factory.py
  - 由于原始timm包存在算子不适配和性能的问题，使用修改后的timm，具体见Code modification。
  

##Code modification
- effdet文件中的修改：
  - /anchors.py: 使用npu_multiclass_nms算子，替换原来的nms
  - /data/loader.py: 替换所有.cuda -> .npu
  - /evaluator.py: 修改class CocoEvaluator(Evaluator) ->  def evaluate(self) -> metric dtype = torch.float32
  - /loss.py: 替代one_hot算子为torch.npu_one_hot算子
  - /efficientdet.py： 修改class FpnCombine(nn.Module) -> def forward(self, x: List[torch.Tensor]) -> stack算子
  
- timm文件中的修改：
  - /models/layers文件夹中针对padv3d算子不适配的情况有三处规避修改,使用自行修改的pad，规避padsame算子：
    - /conv2d_same.py
    - /pool2d_same.py
    - /padding.py
  - /optim/optim_factory.py: 替换optimizers为NpuFusedSGD

## Training

- To train a model, run `run_train.py`,`train_npu.py`,`train_npu_peformance.py` and `eval_npu.py` with the desired model architecture and the path to the coco dataset:

```bash
#env
cd EfficientDetD0
dos2unix ./test/*.sh

# 1p train perf
# 是否正确输出了性能log文件
bash ./test/train_performance_1p.sh --data_path=./datasets/coco
# 验收结果：OK
# 备注：验收测试性能16fps，约为GPU-1p性能 48fps的0.4倍，日志在./test/output/0/train_0_1p_perf.log

# 8p train perf
# 是否正确输出了性能log文件
bash ./test/train_performance_8p.sh --data_path=./datasets/coco
# 验收结果：OK
# 备注：验收测试性能112fps，约为GPU-8p性能 270fps的0.4倍，运行后将输出日志在./test/output/0/train_0_8p_perf.log

# 8p train full
# 是否正确输出了性能精度log文件，是否正确保存了模型文件
bash ./run_train.sh --data_path=./datasets/coco
# 验收结果：OK
# 备注：目标精度0.3346，验收测试精度0.3289，约为目标精度的98.3%
# 运行后将会先后运行test文件夹中train_full_8p_0-120.sh和train_full_8p_121-300.sh,将输出日志在./test/output/0/train_0_8p_120.log和./test/output/1/train_0_8p_300.log内
# 运行后，将会生成./output/train/目录存放模型文件

# 8p eval
# 是否正确输出了性能精度log文件
bash ./test/train_eval_8p.sh --data_path=./datasets/coco --pth_path=./model_best.pth.tar
# 验收结果：OK
# 备注：功能正确，运行后将输出日志在./train_0_eval_8p.log
# 可修改train_eval_8p.sh中训练参数“--resume ",指定训练完成的checkpoint为./output/train文件夹中生成的.pth.tar文件

# finetuning
# 是否正确执行迁移学习
bash ./test/train_finetune_1p.sh --data_path=./datasets/coco --pth_path=./model_best.pth.tar
# 验收结果：OK
# 备注：功能正确，运行后将输出日志在./test/output/0/train_0_finetune_1p.log

# online inference demo 
# 是否正确输出预测结果，请确保输入固定tensor多次运行的输出结果一致
python3.7 demo.py
# 验收结果：OK
# 备注：功能正确，无输出日志
```

- 参数说明：
```bash
#--root               //数据集路径,可自行修改为对应路径的coco数据集
#--resume             //加载模型checkpoint路径，可自行修改为对应路径的模型文件
#--addr               //主机地址 
#--model              //使用模型，默认：tf_efficientdet_d0 
#--opt                //优化器选择
#--epoch              //重复训练次数 
#--batch-size         //训练批次大小 
#--lr                 //初始学习率，默认：0.16
#--model-ema          //使用ema 
#--sync-bn            //是否使用sync-bn 
#--device-list        //多卡训练指定训练用卡 ,8卡：'0,1,2,3,4,5,6,7'
#--lr-noise           //学习率噪声
#--amp                //是否使用混合精度 
#--loss-scale         //lossscale大小 
#--opt-level          //混合精度类型
```



## EfficientDet_D0-Pytorch training result

|   名称    | 性能（fps）| 精度（map）|    BS    |  Epochs  | AMP_Type  |
| :------: | :------:  | :------: | :------: | :------: | :------:  |
| GPU_1P   |    48     | -        | 16       | 1        |     O1    |
| GPU_8P   |   270     | 0.3346   | 16       | 310      |     O1    |
| NPU_1P   |    16     | -        | 16       | 1        |     O1    |
| NPU_8P   |   112     | 0.3289   | 16       | 310      |     O1    |


说明：
- 代码仓最高精度0.336；NPU8卡最高精度出现在第270epoch，精度0.3289；GPU8卡最高精度出现在第250epoch，精度0.3346。
- 性能优化过程中，发现此模型在NPU上，基本的conv，bn，act算子耗时为GPU的3-5倍，因此优化较难。
- GPU训练时，若采用固定lossscale时会在训练时出现loss为nan的情况，故采用源码提供的动态lossscale，动态调整前期losssacle很小（实测为8），后期lossscale很大（万位级）的情况；NPU训练采用了固定的lossscale。其他参数保持一致。
- NPU_8P训练过程由于设备公用情况的限制，同时与GPU同步，防止lossscale较大出现loss为nan的情况，前130epoch在910B卡上训练，lossscale设置为8，性能较910A卡稍差；后180个epoch在910A卡上训练，lossscale设置为128。
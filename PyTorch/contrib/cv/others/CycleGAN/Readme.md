# CycleGAN 训练

## Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

it's an approach for learning to translate an image from a source domain $X$ to a target domain $Y$ in the absence of paired examples

For more detail：https://arxiv.org/abs/1703.10593v6

## 

## Requirements

 **You need to install CNN5.0.3 to ensure the normal training of the model！！** 
and use pytorch, you can use pip or conda to install the requirements

```
# for pip
torch
torchvision
dominate>=2.4.0
visdom>=0.1.8.8
```
注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision ,建议Pillow版本是9.1.0 torchvision版本是0.6.0
## 数据集准备

1.从以下网址获取maps.zip作为训练集

http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/

文件结构如下：


```
CycleGAN
|-- datasets
|   |-- maps
|   |   |-- testA
|   |   |-- testB
|   |   |--trainA
|   |   |--trainB
|   |   |--valA
|   |   |--valB
|-- test   
    |--train_full_1p.sh
    |--train_full_8p.sh
    |--train_performance_1p.sh
    |--train_performance_8p.sh
|-- models      
     |--cycle_gan_model_adapt.py
     |--networks_adapt.py
|--util
     |--html.py
     |--visualizer_adapt.py
     |--util.py
     |--visualizer_adapt.py
|-- dataloader.py
|-- parse.py
|-- train.py
|--env_npu.sh

```

将数据集按照以上结构放在代码目录下

## 安装

请注意，本模型使用了新版本的pytorch以及CANN包，具体版本为：torch-1.5.0+ascend.post3.20210930-cp37-cp37m-linux_aarch64.whl,Ascend-cann-toolkit_5.0.3_linux-aarch64.run；

source 环境变量

```
bash ./env_npu.sh
```


## TRAIN

### 单p训练

source 环境变量

```
bash./env_npu.sh
```

运行单p脚本

```
bash ./test/train_full_1p.sh
```



### 多p训练

source 环境变量

```
source ./env_npu.sh
```

运行8p脚本

```
bash ./test/train_full_8p.sh
```

模型保存在./checkpoints目录下，以数字命名的pth文件是当前epoch训练得到的权重文件，可用来恢复训练；

运行日志保存至./目录下

## TEST

测试精度 



```
由于论文为人眼观察生成效果的真假，所以这里省略，不过下面的demon提供将生成结果以网页的形式更为直观的展现出来
```




## Demo
然后运行以下脚本，执行demo.py：

```
python3 demon.py --pu_ids='0' \
	 --prof=0 \
	 --multiprocessing_distributed=0 \
	 --distributed=1 \
	 --npu=1 \
	 --dataroot=./datasets/maps \
	 --checkpoints_dir=./checkpoints_1pbs1_O1_sacle_1024_torchadam \
	 --model_ga_path=./checkpoints_8pbs1/maps_cycle_gan/175_pu0_net_G_A.pth  \
	 --model_gb_path=./checkpoints_8pbs1/maps_cycle_gan/175_pu0_net_G_B.pth >>npu8pbs1_demon.log 2>&1 &
```

请指定需要测试的模型路径，将--checkpoints_dir、--model_ga_path、--model_gb_path所指向的参数替换掉既可替换掉即可，最后的输出结果存放在根目录的result目录下，点击index.html既可查看，结果展示请在支持浏览器的系统查看。

## 注意事项
1、超参说明
```
--pu_ids='0,1,2,3,4,5,6,7'------------------------------------------指定几张卡训练，必须使用连续的卡号
--prof=0------------------------------------------------------------是否测试性能，当为0时，不测试性能，为1则在大于等于10个epoch后输出prof文件
--multiprocessing_distributed=1-------------------------------------是否执行多核训练，多卡必须为1，单卡设置为0既可
--distributed=1-----------------------------------------------------该参数不可更改
--npu=1-------------------------------------------------------------是否使用Npu开始训练，如果在Npu平台训练则必须使用1，GPU平台则必须为0
--dataroot=./datasets/maps------------------------------------------数据集的目录
--checkpoints_dir=./checkpoints_8pbs1_O1_sacle_1024_torchadam-------存放训练权重的目录
--batch_size=1------------------------------------------------------指定训练时每个step输入多少样本，多卡训练不建议调高，单卡可适当调高为2。bs过大， 
                                                                    会导致判别器过早收敛，进而造成生辰效果不佳                                                 
--isapex=True-------------------------------------------------------是否开启混合精度进行训练，一般是开启的
--apex_type="O1"----------------------------------------------------如果开启混合精度训练，建议使用O1模式，O2模式不收敛。当然O0也是可以的
--loss_scale=1024---------------------------------------------------指定混合精度训练时的loss放大倍数，loss放大倍数也可以被指定为dynamic
--log_path="npu8pbs1.txt"-------------------------------------------只存放与模型有关的日志，不存放与后台输出有关的其他调试日志
--num_epoch_start=0-------------------------------------------------从第几个epoch开始训练，如果开启继续训练，则需要指定该参数
--num_epoch=200-----------------------------------------------------默认训练200个epoch，不可调高，但可以调低
--n_epochs=100------------------------------------------------------权重衰减参数，默认前100个epoch保持学习率不变，后面开始慢慢线性衰减
--lr=1e-4-----------------------------------------------------------baseline的学习率
--line_scale=1------------------------------------------------------baseline的学习率的放大倍数，单卡为1，8卡训练建议设为2，其他卡酌情调参
--n_epochs=100------------------------------------------------------与n_epochs保持一致
--n_epochs_decay=100------------------------------------------------与n_epochs保持一致
--pool_size=50-------------------------------------------------------该参数如果为单卡，使用50既可，如果为8卡，建议设置为16，其他卡酌情调参，一般多卡要调低且数 
                                                                     值为4的倍数
--lambda_A=10--------------------------------------------------------论文超参
--lambda_B=10--------------------------------------------------------论文超参
--loadweight=199_pu0-----------------------------------------------------指定多少个epoch开始继续训练，重新训练默认参数既可
--model_ga_path=./checkpoints_8pbs1/maps_cycle_gan/175_pu0_net_G_A.pth--存放权重的目录，运行demon的时候需要
--model_gb_path=./checkpoints_8pbs1/maps_cycle_gan/175_pu0_net_G_B.pth--存放权重的目录，运行demon的时候需要_ 
```

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
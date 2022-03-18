

## 概述

ECAPA-TDNN基于传统TDNN模型进行了改进，主要有三个方面的优化，分别是：增加了一维SE残差模块（1-Dimensional Squeeze-Excitation Res2Block）;多层特征融合（Multi-layer feature aggregation and summation）;通道和上下文相关的统计池化（Channel- and context-dependent statistics pooling）

 

- 参考实现：
[https://github.com/speechbrain/speechbrain/tree/develop/templates/speaker_id](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fspeechbrain%2Fspeechbrain%2Ftree%2Fdevelop%2Ftemplates%2Fspeaker_id)

  



## 支持特性

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 并行数据   | 是       |

## 混合精度训练

昇腾910 AI处理器提供自动混合精度功能，模型使用 opt_level="O1", loss_scale=128, combine_grad=True的配置进行amp.initialize

脚本已默认开启混合精度，设置如下。

  ```
  parser.add_argument(       
  		"--auto_mix_prec",        
  		action="store_true",        
  		help="This flag enables training with automatic mixed-precision.",     
      )
  ```


<h2 id="训练环境准备.md">训练环境准备</h2>

CANN版本：5.0.2

昇腾torch版本：1.5.0

#### speechbrain环境配置

（详情请参考speechbrain官方文档安装方法。）

1. 安装torch 1.5.0

2. 安装torchaudio，npu安装方法请参考

   https://e.gitee.com/HUAWEI-ASCEND/dashboard?issue=I48AZM

3. cd tdnn

   pip install -r requirement.txt

   pip install --editable .


<h2 id="快速上手.md">快速上手</h2>

- 数据集准备

模型训练使用rirs_noises、train-clean-5数据集，数据集请用户自行获取。

- 模型训练

选择合适的下载方式下载源码包。

Before training, modify the data_folder in these scripts.

```bash
# training 1p loss
bash ./test/train_full_1p.sh --data_folder=""

# training 1p performance
bash ./test/train_performance_1p.sh --data_folder=""

# training 8p loss
bash ./test/train_full_8p.sh --data_folder=""

# training 8p performance
bash ./test/train_performance_8p.sh --data_folder=""
```

```
Log path:
    test/output/train_full_1p.log              # 1p training result log
    test/output/train_performance_1p.log       # 1p training performance result log
    test/output/train_full_8p.log              # 8p training result log
    test/output/train_performance_8p.log       # 8p training performance result log
```

## 训练结果

| acc |  FPS   | Npu_nums | Epochs | AMP_Type |
| :--------: | :----: | :------: | :----: | :------: |
|     -      |  8.25  |    1     |   1    |    O1    |
|  0.9062  | 43.863 |    8     |   5    |    O1    |

<h2 id="高级参考.md">高级参考</h2>

### 脚本和示例代码

```
├── README.md                                 //代码说明文档
├── speechbrain                               //框架支持文件
├── templates/speaker_id
│    ├──test                                 //测试脚本
│    ├──custom_model.py                      //简易TDNN模块
|    ├──mini_librispeech_prepare.py          //数据清单文件
│    ├──run_1p.sh                            //单卡运行启动脚本
│    ├──run_8p.sh                            //8卡运行启动脚本
│    ├──train.py                             //网络训练与测试代码
│    ├──train.yaml                           //网络训练参数脚本 
```

### 脚本参数

```
--seed                   制作参数对象seed
--rank_size              使用NPU卡数量，默认：1
--number_of_epochs       训练epoch次数，默认：5
--data_folder            数据集路径，默认：./data
--output_folder          结果输出保存的文件路径，默认：./results/speaker_id/<seed>
--batch_size             每个NPU的batch size，默认：64
```


## 训练过程

1.  通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡、8卡网络训练。

2.  参考脚本的模型存储路径为results/<seed>/save，训练脚本log中包括如下信息。

```
Epoch: 1, lr: 1.00e-03 - train loss: 2.70 - valid loss: 3.39, valid error: 9.47e-01
Epoch loaded: 1 - test loss: 3.43, test error: 9.54e-01
```
## 注意事项

 **该模型为了固定shape，修改了1、/speechbrain/dataio/dataio.py read_audio函数 2、/speechbrain/templates/speaker_id/train.py prepare_features函数 3、/speechbrain/core.py _train_loader_specifics里的sampler。其中第三个修改是因为数据集不足固定shape，实际使用模型务必还原回去。** 
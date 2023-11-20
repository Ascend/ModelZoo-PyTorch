# BertSum
## 1.概述

#### 模型概述
**Bertsum是一个经典的抽取式文本摘要网络，其主要特点是使用了Bert预训练+微调的方式进行文本摘要任务。**

**在网络结构上，为了适应摘要任务，作者对于Bert的输入部分进行了调整，在每一个句子的前后都插入了CLS**

**和SEP的符号，微调阶段使用了三种分类器，分别是全连接层、LSTM以及transformer。然后在CNN/Dailymail**

**数据集上取得了较好的效果。**

**参考文献： `Fine-tune BERT for Extractive Summarization`**(https://arxiv.org/pdf/1903.10318.pdf)

**源代码链接： https://github.com/nlpyang/BertSum**

**请下载url目录 https://github.com/nlpyang/BertSum/tree/master/urls 到 url文件件下 **

#### 默认配置
**训练的超参数**：  

* batch_size:3000  
* hidden_size:128  
* ff_size:512  
* heads:4  
* optimizer:Adam  
* decay_method noam  
* dropout:0.1  
* lr:2e - 2  
* beta1:0.9  
* beta2:0.999  
* warmup_steps:10000  
* train_steps:50000 

## 2.支持特性
支持的特性包括：1.分布式训练。2.混合精度。3.数据并行。

8P训练脚本，支持数据并行的分布式训练。脚本样例中默认开启了混合精度，参考示例，见“开启混合精度”。

#### 混合精度训练
昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动

将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

#### 开启混合精度
脚本已默认开启混合精度，设置`train_with_amp`参数的值为`true`，参考脚本如下:
```bash
python train.py \
         -mode train \
	     -encoder classifier  \
	     -dropout 0.1 \
         -bert_data_path ../bert_data/cnndm  \
	     -model_path ../models/bert_classifier  \
	     -lr 2e-3  \
	     -visible_gpus 0  \
         -gpu_ranks 0  \
	     -world_size 1  \
	     -report_every 1  \
	     -save_checkpoint_steps 1000  \
	     -batch_size 3000  \
	     -decay_method noam  \
	     -train_steps 50000  \
	     -accum_count 2  \
	     -log_file ../logs/bert_classifier  \
	     -use_interval true  \
	     -warmup_steps 10000  \
	     -train_npu true  \
         -train_with_amp true
```

## 3.准备工作
#### conda环境
* 使用`pip install -r requirements.txt`安装依赖环境

#### 验证环境准备
* apex安装好之后，启动训练脚本，可能会报错：`No module named "fused_layer_norm_cuda"`，这是由于安装apex

  时没有安装CUDA/c++扩展。然而安装这两个扩展时，你可能会遇到cuda版本过高的问题。因此为了规避此问题，我们

  将环境中的`pytorch_pretrained_bert`这个依赖包的`modeling.py`模块（参考链接：`/root/anaconda3/envs/bertsum/`

  `lib/python3.7/site-packages/pytorch_pretrained_bert/modeling.py`）的228行进行修改，使用**`torch.nn.Layernorm`**替换

  掉`FusedLayerNorm`。

* 验证阶段，需要使用ROUGE来计算精度，请提前安装并配置好**ROUGE**。  
**Ubuntu参考链接**：https://zhuanlan.zhihu.com/p/57543809     
**Centos参考链接**：https://zhuanlan.zhihu.com/p/428864759



#### 脚本和示例代码
```bash
├── bert_config_uncased_base.json    //bert的配置文件
├── bert_data                        //模型加载数据的目录
├── json_data                        //数据预处理阶段存放中间文件的目录
├── LICENSE                          //license
├── logs                             //训练阶段存放日志的目录
├── models                           //训练过程中，存放checkpoints的目录
├── raw_data                         //数据处理阶段，存放原始数据的目录
├── README.md                        //代码说明文件
├── README.raw.md                    //源代码仓的代码说明文件
├── results                          //计算rouge时，用来存放结果文件的目录
├── src                              
│   ├── distributed.py               //分布式训练相关的模块
│   ├── cal_performance.py           //计算性能的脚本
│   ├── cal_perf_1p.sh               //启动计算1p性能的脚本
│   ├── cal_perf_8p.sh               //启动计算8p性能的脚本
│   ├── models                 
│   │   ├── data_loader.py           //dataloader模块
│   │   ├── encoder.py               //finetune的encoder
│   │   ├── __init__.py
│   │   ├── model_builder.py         //model构造的模块
│   │   ├── neural.py                //模型中的一些神经元，包括激活函数和attention
│   │   ├── optimizers.py            //优化器模块
│   │   ├── reporter.py              //控制日志输出的模块
│   │   ├── rnn.py                   //rnn模块
│   │   ├── stats.py                 //记录训练过程中的loss以及lr等信息的模块
│   │   └── trainer.py               //控制训练、验证及测试的类
│   ├── others                      
│   │   ├── __init__.py
│   │   ├── logging.py               //日志模块
│   │   ├── pyrouge.py               //python和ROUGE的接口模块 
│   │   └── utils.py                 //一些工具
│   ├── prepro                       //数据预处理的一些模块
│   │   ├── data_builder.py    
│   │   ├── __init__.py
│   │   ├── smart_common_words.txt
│   │   └── utils.py
│   ├── preprocess.py                //数据预处理的脚本
│   └── train.py                     //训练脚本
├── test
│   ├── env_npu.sh                   //npu的环境变量
│   ├── train_performance_1p.sh      //单卡性能测试的启动脚本
│   ├── train_performance_8p.sh      //8卡性能测试的启动脚本
│   ├── train_full_8p.sh             //8卡训练流程的启动脚本
│   ├── train_finetune_1p.sh         //单卡加载预训练模型及训练流程的启动脚本
│   └── train_eval_8p.sh             //验证流程的启动脚本
├── temp                             //用来存放一些中间文件以及Bert的本地预训练模型
├── urls                             //数据预处理阶段使用到的一些映射文件的url
└── requirements.txt                 //依赖环境文件
```

## 4.训练及验证

#### 数据准备
* step1.下载好处理过的数据集或者参考`README.raw.md`处理完自己的数据集之后，将所有的*.pt文件放到`bert_data`目录下

* step2.将Bert预训练模型`"bert_base_uncased"`放到`temp`目录下

#### 模型训练
* step1.cd到src目录，所有的脚本都要在此目录下运行

* step2.加载预训练模型并启动单机单卡训练(`bash ../test/train_finetune_1p.sh`,将shell脚本中的`PRETRAINED_MODEL_PATH`
  换为预训练模型的路径及文件名)；启动单机8卡训练(`bash ../test/train_8p.sh`)；跑完单卡和多卡性能需要分别运行计算性能的
  脚本（`bash ./cal_perf_1p.sh`或者`bash ./cal_perf_8p.sh`）

* step3.启动验证脚本，计算rouge精度(`bash ../test/train_eval_8p.sh`)

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md 
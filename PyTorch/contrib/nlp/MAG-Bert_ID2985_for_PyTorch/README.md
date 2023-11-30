- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Natural Language Processing**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.8.12**

**大小（Size）：940160KB**

**框架（Framework）：PyTorch 1.7**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于PyTorch框架的情感识别训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

 网络整体框架为Bert模型，中间插入了MAG多模态适应门，以完成多模态数据的情感识别。

- 参考论文：

  https://www.aclweb.org/anthology/2020.acl-main.214.pdf

- 参考实现：

  https://github.com/WasifurRahman/BERT_multimodal_transformer

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/contrib/nlp/MAG-Bert_ID2985_for_PyTorch

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    - max_seq_length：50
    - train_batch_size：48
    - n_epochs: 10
    - dataset mosi
    - learning_rate 1e-5
    - data_path
    - dev_batch_size 128
    - output_path


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 否       |
| 数据并行   | 是       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

基于NPU芯片的架构特性，会涉及到混合精度训练，即混合使用float16和float32数据类型的应用场景。使用float16代替float32有如下好处：

-   对于中间变量的内存占用更少，节省内存的使用。
-   因内存使用会减少，所以数据传出的时间也会相应减少。
-   float16的计算单元可以提供更快的计算性能。

但是，混合精度训练受限于float16表达的精度范围，单纯将float32转换成float16会影响训练收敛情况，为了保证部分计算使用float16来进行加速的同时能保证训练收敛，这里采用混合精度模块Apex来达到以上效果。混合精度模块Apex是一个集优化性能、精度收敛于一身的综合优化库。

适配昇腾AI处理器的混合精度模块Apex除了上述优点外，还能提升运算性能。具体如下：

-   Apex在混合精度运算过程中，会对模型的grad进行运算。开启combine\_grad开关，可以加速这些运算。具体为将amp.initialize\(\)接口参数combine\_grad设置为True；
-   适配后的Apex针对adadelta/adam/sgd/lamb做了昇腾AI处理器亲和性优化，得到的NPU融合优化器与原生算法保持一致，但运算速度更快。使用时只需将原有优化器替换为apex.optimizers.\*（“\*”为优化器名称，例如NpuFusedSGD）。
-   适配后的Apex针对数据并行场景做了昇腾AI处理器亲和性优化，支持利用融合grad进行加速，同时保持计算逻辑一致性。通过开启combine\_ddp开关，也就是将amp.initialize\(\)接口参数combine\_ddp设置为True并关闭DistributedDataParallel，即可开启该功能。

**特性支持**<a name="section723462915303"></a>

混合精度模块功能和优化描述如[表1](#table10717173813332)所示。

**表 1**  混合精度模块功能

<a name="table10717173813332"></a>
<table><thead align="left"><tr id="row371716385333"><th class="cellrowborder" valign="top" width="32.269999999999996%" id="mcps1.2.3.1.1"><p id="p13717163815333"><a name="p13717163815333"></a><a name="p13717163815333"></a>功能</p>
</th>
<th class="cellrowborder" valign="top" width="67.73%" id="mcps1.2.3.1.2"><p id="p14400173910345"><a name="p14400173910345"></a><a name="p14400173910345"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1571763813334"><td class="cellrowborder" valign="top" width="32.269999999999996%" headers="mcps1.2.3.1.1 "><p id="p4502732153412"><a name="p4502732153412"></a><a name="p4502732153412"></a>O1配置模式</p>
</td>
<td class="cellrowborder" valign="top" width="67.73%" headers="mcps1.2.3.1.2 "><p id="p640053920348"><a name="p640053920348"></a><a name="p640053920348"></a>Conv，Matmul等使用float16计算，其他如Softmax、BN使用float32。</p>
</td>
</tr>
<tr id="row3717173817336"><td class="cellrowborder" valign="top" width="32.269999999999996%" headers="mcps1.2.3.1.1 "><p id="p11503103210344"><a name="p11503103210344"></a><a name="p11503103210344"></a>O2配置模式</p>
</td>
<td class="cellrowborder" valign="top" width="67.73%" headers="mcps1.2.3.1.2 "><p id="p164001639143419"><a name="p164001639143419"></a><a name="p164001639143419"></a>除了BN使用float32外，其他绝大部分使用float16。</p>
</td>
</tr>
<tr id="row14717193815334"><td class="cellrowborder" valign="top" width="32.269999999999996%" headers="mcps1.2.3.1.1 "><p id="p1950318328349"><a name="p1950318328349"></a><a name="p1950318328349"></a>静态Loss Scale功能</p>
</td>
<td class="cellrowborder" valign="top" width="67.73%" headers="mcps1.2.3.1.2 "><p id="p1440033983418"><a name="p1440033983418"></a><a name="p1440033983418"></a>静态设置参数确保混合精度训练收敛。</p>
</td>
</tr>
<tr id="row871733813317"><td class="cellrowborder" valign="top" width="32.269999999999996%" headers="mcps1.2.3.1.1 "><p id="p1550303243417"><a name="p1550303243417"></a><a name="p1550303243417"></a>动态Loss Scale功能</p>
</td>
<td class="cellrowborder" valign="top" width="67.73%" headers="mcps1.2.3.1.2 "><p id="p15400143963412"><a name="p15400143963412"></a><a name="p15400143963412"></a>动态计算loss Scale值并判断是否溢出。</p>
</td>
</tr>
</tbody>
</table>

>**说明：** 
>-   当前版本的实现方式主要为python实现，不支持AscendCL或者CUDA优化。
>-   当前昇腾AI设备暂不支持原始Apex的FusedLayerNorm接口模块，如果模型原始脚本文件使用了FusedLayerNorm接口模块，需要在模型迁移过程中将脚本头文件“from apex.normalization import FusedLayerNorm“替换为“from torch.nn import LayerNorm“。

**将混合精度模块集成到PyTorch模型中**<a name="section18578112873911"></a>

1.  使用apex混合精度模块需要首先从apex库中导入amp，代码如下：

    ```
    from apex import amp
    ```

2.  导入amp模块后，需要初始化amp，使其能对模型、优化器以及PyTorch内部函数进行必要的改动，初始化代码如下：

    ```
    model, optimizer = amp.initialize(model, optimizer，combine_grad=True)
    ```

3.  标记反向传播.backward\(\)发生的位置，这样Amp就可以进行Loss Scaling并清除每次迭代的状态，代码如下：

    原始代码：

    ```
    loss = criterion(…) 
    loss.backward() 
    optimizer.step()
    ```

    修改以支持loss scaling后的代码：

    ```
    loss = criterion(…) 
    with amp.scale_loss(loss, optimizer) as scaled_loss:     
        scaled_loss.backward() 
    optimizer.step()
    ```

<h3 id="Pytorch1.8.1  AMP 在NPU上的使用说明md">Pytorch1.8.1  AMP 在NPU上的使用说明</h3>

<h4 id=总体说明md">总体说明</h4>

Pytorch1.8.1版本的AMP，类似于Apex AMP的O1模式（动态 loss scale），也是通过将部分算子的输入转换为fp16类型来实现混合精度的训练。

<h4 id="AMP使用场景md">AMP使用场景</h4>

1. 典型场景。

2. 梯度累加场景。

3. 多Models,Losses,and Optimizers场景。

4. DDP场景（one NPU per process）。


目前针对pytorch1.8.1框架仅支持以上4种场景，更多场景使用请参考pytorch官方操作指南。

<h4 id="NPU上AMP的使用方法md">NPU上AMP的使用方法</h4>

1. 模型从GPU适配到NPU时，需要将代码torch.cuda.amp修改为torch_npu.npu.amp。
2. 当前Pytroch1.8.1 AMP工具中GradScaler增加了dynamic选项（默认为True）,设置为False时，AMP能支持静态Loss Scale。

<h4 id="注意事项md">注意事项</h4>

1. 1.8.1中AMP使用装饰器的方式实现。在train与test时需要通过添加with Autocast()将模型的入参转换为FP16，如果不添加，模型入参仍为FP32，在极限batchsize下，会出现内存不足的问题。
4. 当前1.8.1AMP不支持tensor融合功能。

<h2 id="训练环境准备.md">训练环境准备</h2>

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

1、用户自行准备好数据集，模型训练使用Mosei数据集，run ./download_datasets.sh to download MOSI and MOSEI datasets

2、 MAG-Bert训练的模型及数据集可以参考"简述 -> 参考实现"



## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练
    
      1. 配置训练参数。
    
         首先在脚本test/train_full_1p.sh中，配置batch_size、steps、epochs、data_path等参数，请用户根据实际路径配置data_path，或者在启动训练的命令行中以参数形式 
         下发。
    
         ```
         max_seq_length=50
         train_batch_size=48
         n_epochs=10
         data_path="/home/ma-user/modelarts/inputs/data_url_0"
         ```
    
      2. 启动训练。
    
         启动单卡训练 （脚本为BERT_multimodal_transformer-maste\test/train_full_1p.sh）
    
         ```
         bash train_full_1p.sh --data_path='/home/ma-user/modelarts/inputs/data_url_0'
         ```
      3. 精度指标。

        ```
       | 精度指标项 | 论文发布 | GPU实测 | NPU实测 |
       | ---------- | -------- | ------- | ------- |
       | F1_SCORE   | 84.1     | 84.27   | 82.59   |
           


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── multimodal_driver.py                      //网络训练与测试代码
├── README.md                                 //代码说明文档
├── modeling.py                               //MAG门代码
├── global_configs.py                         //全局参数设置
├── argparse_utils.py                         //局部参数设置
├── bert.py                                   //bert模型代码
├── requirements.txt                          //训练python依赖列表
├── modelarts_entry_acc.py                    //单卡全量训练启动代码
├── modelarts_entry_perf.py                   //单卡训练验证性能启动代码
├── test
│    ├──train_performance_1p.sh               //单卡训练验证性能启动脚本
│    ├──train_full_1p.sh                      //单卡全量训练启动脚本
```

## 脚本参数<a name="section6669162441511"></a>

```
--max_seq_length=50
--train_batch_size=48
--n_epochs=10
--data_path                  数据集路径，默认：/home/ma-user/modelarts/inputs/data_url_0
--max_seq_length             单句最大长度，默认：50
--train_batch_size           每个NPU的batch size，默认：48
--learing_rate          初始学习率，默认：0.00001
--n_epochs                训练epcoh数量，默认：10
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
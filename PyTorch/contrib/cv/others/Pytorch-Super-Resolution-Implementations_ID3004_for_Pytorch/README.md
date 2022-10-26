- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Object Detection**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.8.16**

**大小（Size）：3892580KB**

**框架（Framework）：PyTorch 1.5**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于PyTorch框架的残差密集网络的图像超分辨实现训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

一个非常深的卷积神经网络 (CNN) 最近在图像超分辨率 (SR) 方面取得了巨大成功，并且还提供了分层特征。然而，大多数基于深度 CNN 的 SR 模型并没有充分利用原始低分辨率 (LR) 图像的层次特征，从而实现了相对较低的性能。在本文中，我们提出了一种新的残差密集网络（RDN）来解决图像 SR 中的这个问题。我们充分利用了所有卷积层的分层特征。具体来说，我们提出残差密集块（RDB）通过密集连接的卷积层提取丰富的局部特征。RDB 进一步允许从前一个 RDB 的状态直接连接到当前 RDB 的所有层，从而形成连续内存 (CM) 机制。然后使用 RDB 中的局部特征融合从先前和当前的局部特征中自适应地学习更有效的特征，并稳定更广泛网络的训练。在充分获得密集的局部特征后，我们使用全局特征融合来联合和自适应地学习全局层次特征。在具有不同退化模型的基准数据集上进行的大量实验表明，我们的 RDN 与最先进的方法相比取得了良好的性能.

- 参考论文：

  https://arxiv.org/abs/1802.08797

- 参考实现：

  https://github.com/puffnjackie/pytorch-super-resolution-implementations

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/contrib/cv/others/Pytorch-Super-Resolution-Implementations_ID3004_for_Pytorch

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
   - lr = 0.001  
   - nEpochs=1000 
   - batchSize = 8 
   - testBatchSize = 4
   - isCuda = False 
   - nThread = 4 
   - datapath 
   - model 
    

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
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
    model, optimizer = amp.initialize(model,optimizer,opt_level='O2',loss_scale=32.0,combine_grad=True)
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
    loss = criterion(preds, labels)
    optimizer.zero_grad()
    # loss.backward()
    with amp.scale_loss(loss, optimizer) as scaled_loss:
          scaled_loss.backward()
    optimizer.step()

    ```



<h2 id="训练环境准备.md">训练环境准备</h2>

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

1、训练数据集DIV2K， 测试数据集Set5，URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=kMZ5rC1dFBGYHI6Ycr96rcKSqlXL/5HAm77GvUS9QYBHiqXrq0cOgWg0GfQ1aIoVF3O9EBVZh0CFvbtT8TFi70VjXynqXKrZ09bmxrLQ+NTsZlnaTqSt8Op3yJT243YpNXnx1IHPE8nynkP88cjdKKDMBAaKeyfmY6kC6zpzYllco7lLeHyA9fpVpRHdB4FP6C2a2t91MG1gxYOGlvSwxI1l7f/WbEbqJ7qZFWRpPatC2VX5v6jY/Cq6/jiI1UdWNiSUvDe4rmdZajB7d2ZpLzPWjhMg3/epTIFGvNnjGmsYFbEDrnc4Uh8qqsq4786QIUycBGTAkGgoQBavhcJ6jDkidMLgJucTxd9rM3OLeFLj7Er9TyW3q7NpqywF1JWqqEMWs6MNF3nZO8ew6Aca84QK5Fb0BYfevWocsaDc6/t19f5xu06SZqU4SJyUeUco8V9tZ/wiuJpLPNdKup9++tDSJkKdrGBeuWU3m170DReO2Dz44DhTJw/8OcSVYD545FEkGxwx8hTFqdG/5tR2Unq4nPbKxnEsumskgkBxAZo=

提取码:
123456

*有效期至: 2023/01/08 07:50:42 GMT-08:00
数据集下载后放入代码目录即可。

2、Pytorch-Super-Resolution-Implementations训练的模型及数据集可以参考"简述 -> 参考实现"


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
         -datapath=${data_path}
         -nEpochs = 250
         -batchSize = 8
         ```
    
      2. 启动训练。
    
         启动单卡训练 （脚本为Pytorch-Super-Resolution-Implementations_ID3004_for_Pytorch/test/train_full_1p.sh）
    
         ```
         bash train_full_1p.sh --data_path='dataset'
         ```
      3. 精度指标。

       | 精度指标项 | 论文发布 | GPU实测 | NPU实测 |
       | ---------- | -------- | ------- | ------- |
       | PSNR   | 32.47    | 16.978   | 17.35  |

      4. 性能指标。

       | 性能指标项 | 论文发布 | GPU实测 | NPU实测 |
       | ---------- | -------- | ------- | ------- |
       | s/epoch   | 无    | 26.136   | 22.506  |
           


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├─README.md  
├─LICENSE    
├─ckpt    模型保存文件夹    
├─dataset       存放数据集文件夹  
├─test       test文件夹，存放脚本    
├─model 模型实现  
├─dataset.py  数据处理代码  
├─utils.py  工具类  
├─train.py  模型训练函数  
├─inference.py  模型推理函数  
├─train_npu.py  模型在NPU上训练函数
├─requirements.txt  所需包 
├── test
│    ├──train_performance_1p.sh               //单卡训练验证性能启动脚本
│    ├──train_full_1p.sh                      //单卡全量训练启动脚本
```

## 脚本参数<a name="section6669162441511"></a>

```
-- lr = 0.001  
-- nEpochs=1000 
-- batchSize = 8 
-- testBatchSize = 4
-- isCuda = False 
-- nThread = 4 
-- datapath 
-- model 
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。
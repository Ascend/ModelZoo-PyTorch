##  基本信息

**发布者（Publisher）：徐思源**

**应用领域（Application Domain）： Multimodal**

**版本（Version）：1**

**修改时间（Modified） ：2022.9.7**

**框架（Framework）：Pytorch 1.5**

**大小（Size）：905.57KB**

**模型格式（Model Format）：pt**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于iemocap数据集的低秩多模态融合代码**

## 概述

```
该模型完成了基于iemocap数据集的情感识别任务
```

LMF首先通过将三个模态的单峰输入分别传递到三个子嵌入网络中获得单峰表示，然后通过使用模态特定因子执行低秩多模态融合来产生多模态输出表示，该表示可用于生成预测任务。

- 参考论文：

    [https://arxiv.org/pdf/1806.00064.pdf](https://arxiv.org/pdf/1806.00064.pdf)

- 参考实现：

  [https://github.com/Justin1904/Low-rank-Multimodal-Fusion](https://github.com/Justin1904/Low-rank-Multimodal-Fusion)

- 适配昇腾 AI 处理器的实现：

   [https://gitee.com/xusiyuan713/ModelZoo-PyTorch/tree/master/PyTorch/contrib/others/Low-rank-Multimodal-Fusion_ID2983_for_Pytorch](https://gitee.com/xusiyuan713/ModelZoo-PyTorch/tree/master/PyTorch/contrib/others/Low-rank-Multimodal-Fusion_ID2983_for_Pytorch)

- 通过Git获取对应commit_id的代码方法如下：

  ```
  git clone {repository_url}    # 克隆仓库的代码
  cd {repository_name}    # 切换到模型的代码仓目录
  git checkout  {branch}    # 切换到对应分支
  git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
  cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  
  ```

##  默认配置
`--epochs=500`：训练中的最大训练轮数。由于提前停止用于防止过拟合，在实际训练中，训练轮数可能小于此处指定的训练轮数。
`--pationce=20`： 如果模型性能在多次连续验证评估中没有改善，则训练将提前停止。
`--output_dim=2`：模型的输出维度。
`--emotion=b'neutral'`:指定要训练模型预测的情绪类别。可以是“快乐”、“悲伤”、“愤怒”、“中性”。

- angry：
  - audio hidden=8
  - video hidden=4
  - text hidden=64
  - audio dropout=0.3
  - video dropout=0.1
  - text dropout=0.15
  - factor lr=0.003
  - learning rate=0.0005
  - rank=8
  - batch size=64
  - weight decay=0.001

- sad：

  - audio hidden=8
  - video hidden=4
  - text hidden=128
  - audio dropout=0
  - video dropout=0
  - text dropout=0
   - factor lr=0.0005
  - learning rate=0.003
  - rank=4
  - batch size=256
  - weight decay=0.002


- happy：

  - audio hidden=4
  - video hidden=16
  - text hidden=128
  - audio dropout=0.3
  - video dropout=0.1
  - text dropout=0.5
   - factor lr=0.003
  - learning rate=0.001
  - rank=1
  - batch size=256
  - weight decay=0.002
    

- happy：

  - audio hidden=32
  - video hidden=8
  - text hidden=64
  - audio dropout=0.2
  - video dropout=0.5
  - text dropout=0.2
   - factor lr=0.003
  - learning rate=0.0005
  - rank=16
  - batch size=256
  - weight decay=0.002

##  支持特性

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是     |
| 并行数据   | 否       |
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
-   适配后的Apex针对adadelta/adam/sgd/lamb做了昇腾AI处理器亲和性优化，得到的NPU融合优化器与原生算法保持一致，但运算速度更快。使用时只需将原有优化器替换为apex.optimizers.\*（“\*”为优化器名称，例如NpuFusedAdam）。
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
    model, optimizer = amp.initialize(model,optimizer,opt_level='O2',loss_scale=128.0,combine_grad=True)
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

##  训练环境准备

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录



## 快速上手

- 数据集准备

1. 模型训练使用iemocap数据集，数据集可以从这个OBS链接获取：obs://cann-id2983/dataset/iemocap.pkl
2. 或者从https://drive.google.com/open?id=1CixSaw3dpHESNG0CaCJV6KutdlANP_cr获取
3. 将获得的数据放在项目文件的data文件下

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练
    
      1. 配置训练参数。
    
         首先在脚本test/train_full_1p.sh中，配置data_url等参数，请用户根据实际路径配置data_url，或者在启动训练的命令行中以参数形式 
         下发。
    
         ```

          --data_url=${data_path} 
          --train_url=${output_path} 
           --para_path=${para_path}_path}
         ```
    
      2. 启动训练。
    
         启动单卡训练 （脚本为MMSA_ID2979_for_PyTorch/test/train_full_1p.sh）
    
         ```
         bash train_full_1p.sh --data_url='/home/test_user08/Low-rank-Multimodal-Fusion-master/data/' --train_url="/home/test_user08/Low-rank-Multimodal-Fusion-master/outputs/" --para_path="/home/test_user08/Low-rank-Multimodal-Fusion-master/param_data.csv"
    

##  训练结果

- 精度结果比对

| 精度指标项 | 论文发布 | GPU实测 | NPU实测 |
| ---------- | -------- | ------- | ------- |
| F1_angry   | 89.0     | 87.86   | 89.38   |
| F1_sad   | 85.9       | 84.37   | 84.06  |
| F1_happy   | 85.8     | 84.06  | 83.74   |
| F1_neutral   | 71.7     | 69.33 | 71.34   |

- 性能结果比对

| 性能指标项 |  GPU实测 | NPU实测 |
| ----------  | ------- | ------- |
| step time_angry   |   0.374 |0.028 |
| step time_sad | 0.381   | 0.037 |
| step time_happy | 0.374 | 0.037   |
|  step time_neutral  | 0.377 | 0.025   |
| FPS_angry   |  171.112  |2224.38 |
| FPS_sad | 671.91   | 6872.65 |
| FPS_happy | 684.49  | 6959.98   |
| FPS_neutral   |  679.05  |10276.13 |


## 高级参考

## 脚本和示例代码

```
├── train_angry.py                      //网络训练与测试代码
├── README.md                                 //代码说明文档
├── model.py                               //模型代码
├── para_data.csv                         //参数设置
├── utils.py                         //数据集封装
├── requirements.txt                          //训练python依赖列表
├── modelarts_entry_acc.py                    //单卡全量训练启动代码
├── modelarts_entry_perf.py                   //单卡训练验证性能启动代码
├── test
│    ├──train_performance_1p.sh               //单卡训练验证性能启动脚本
│    ├──train_full_1p.sh                      //单卡全量训练启动脚本
├── data
│    ├──iemocap.pkl            //数据集


```

## 脚本参数

```
 --data_url=${data_path} 
 --train_url=${output_path} 
 --para_path=${para_path}_path}
```


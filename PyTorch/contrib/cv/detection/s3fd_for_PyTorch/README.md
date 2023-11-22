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

**修改时间（Modified） ：2022.11.30**

**大小（Size）：85.6MB**

**框架（Framework）：PyTorch 1.8**

**模型格式（Model Format）：pth**

**精度（Precision）：FP16**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于PyTorch框架的人脸检测训练代码**

## 概述<a name="section194554031510"></a>

 针对尺度不变人脸的单阶段检测器的PyTorch实现，在小尺度目标检测方面具有良好效果。

- 参考论文：

  https://arxiv.org/abs/1708.05237

- 参考实现：

  https://github.com/sfzhang15/SFD

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/contrib/cv/s3fd_ID2934_for_PyTorch
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参：
- batch_size 16
- num_workers 4
- learning-rate 0.0001
- dataset face
- basenet 
- save_folder


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 否       |
| 数据并行   | 否       |


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

1. 下载WIDER face数据集   http://shuoyang1213.me/WIDERFACE
2. 修改data/config.py中的数据集路径
```
_C.FACE.TRAIN_FILE = 'data/face_train.txt'
_C.FACE.VAL_FILE = 'data/face_val.txt'
_C.FACE.WIDER_DIR = '{data_path}/dataset'
```
3. 执行初始化代码
```
python prepare_wider_data.py
```
4. s3fd训练的模型及数据集可以参考"概述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练
    
    1. 配置训练参数。
    
         数据集路径的参数配置在data/config.py中进行，见"快速上手 -> 数据集准备"
         
    
	2. 启动训练。
    
         启动单卡GPU训练 （切换到code目录）
    
         ```
         python3 train.py --device gpu 
         ```
         启动单卡NPU训练 （切换到code目录）
    
         ```
         python3 train.py --device npu 
         ```
         若需要从checkpoint继续训练，则
         ```
         python3 train.py --device {device} --resume {checkpoint_path} 
         ```
   3. 精度指标。
   
   GPU训练loss如下：

   ```
    Timer: 0.3231
    epoch:8  ||  iter:6640  ||  Loss:2. 9980
    ->> conf loss:1.9670  ||  loc loss:1. 9074
    ->>lr:0.001000 
    Timer: 0. 3260
    epoch:8  ||  iter:6650  ||  Loss:3. 0034
    ->> conf loss:2.0115  ||  loc loss:1.9247
    ->>lr:0.001000
    Timer: 0.3225
    epoch:8  ||  iter:6660  ||  Loss:3. 0076
    ->> conf loss:1.4032  ||  loc loss:1.2627
    ->>lr :0.001000
   ```

   NPU训练loss如下：

   ```
    Timer:1.7223
    epoch:8  ||  iter:6440  ||  Loss:3.0316
    ->> conf loss:1. 7048  ||  loc loss: 1.3268
    ->>lr :0. 001000
    Timer :0.7953
    epoch:8  ||  iter:6441  ||  Loss:3.0733
    ->> conf loss: 1.6830  ||  loc loss: 1.4320
    ->>lr :0.001000
    Timer :0.8737
    epoch:8  ||  iter:6442  ||  Loss:2. 7748
    ->> conf loss: 1.1785  ||  loc loss:0.9995
    ->>lr:0. 001000
   ```
   
   同步数下loss对齐



      

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── data
│    ├──__init__.py              
│    ├──config.py  
│    ├──factory.py              
│    ├──widerface.py 
├── layers
│    ├── functions   
│    │    ├── __init__.py
│    │    ├── detection.py 
│    │    ├── prior_box.py 
│    ├── modules   
│    │    ├── __init__.py
│    │    ├── l2norm.py 
│    │    ├── multibox_loss.py 
│    ├── __init__.py 
│    ├── bbox_utils.py
├── utils                         //图像数据处理代码
│    ├── __init__.py              
│    ├── augmentations.py 
├── weights                       //权重文件，模型默认保存在此目录
│    ├── vgg16_reducedfc.pth      //basenet
├── prepare_wider_data.py         //数据集预处理
├── README.md                     //代码说明文档
├── requiremnets.txt              //训练python依赖列表
├── s3fd.py                       //模型
├── train.py                      //网络训练代码
```

## 训练过程<a name="section1589455252218"></a>

训练日志中包括如下信息。
```
Timer: 1.1087
epoch:5 || iter:4750 || Loss:5.8974
->> conf loss:1.1982 || loc loss:4.1487
->>lr:0.001000
Timer: 3.5912
epoch:5 || iter:4760 || Loss:5.8922
->> conf loss:1.3858 || loc loss:4.1821
->>lr:0.001000
Timer: 1.0676
epoch:5 || iter:4770 || Loss:5.8903
->> conf loss:1.2478 || loc loss:4.2127
->>lr:0.001000
Timer: 0.9416
epoch:5 || iter:4780 || Loss:5.8892
->> conf loss:0.8212 || loc loss:4.6874
->>lr:0.001000
Timer: 1.2547
epoch:5 || iter:4790 || Loss:5.8877
->> conf loss:1.4334 || loc loss:4.4520
->>lr:0.001000
Timer: 6.2712
epoch:5 || iter:4800 || Loss:5.8843
->> conf loss:1.5964 || loc loss:4.1946
->>lr:0.001000
Timer: 6.0671
epoch:5 || iter:4810 || Loss:5.8823
->> conf loss:1.7890 || loc loss:4.8760
->>lr:0.001000
Timer: 46.3256
epoch:5 || iter:4820 || Loss:5.8848
->> conf loss:2.5106 || loc loss:4.4186
->>lr:0.001000
Timer: 392.5459
test epoch:5 || Loss:6.2771
Saving best state,epoch 5
```


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
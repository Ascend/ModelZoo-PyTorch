##  基本信息

**发布者（Publisher）：郭岩河**

**应用领域（Application Domain）： Multimodal**

**版本（Version）：1**

**修改时间（Modified） ：2022.12.16**

**框架（Framework）：Pytorch 1.8**

**大小（Size）：422.57KB**

**模型格式（Model Format）：pth**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于FiveK数据集的Laplacian金字塔图片实时转换网络代码**

## 概述

```
该模型完成了基于FiveK数据集的图片美化转换代码
```

LPTN利用GAN网络来转换图片，GAN的generator用于将原始图片转换成美化的图片，discriminator用于判别转换的图片是否符合美化图片数据集的要求。
因此，不需要成对的训练集，LPTN就可以实现图片转换任务。其中，美化的图片换成其它风格，比如白天/黑夜，冬/夏，就可以实现
其它类型的风格转换任务。

- 参考论文：

    [https://arxiv.org/pdf/2105.09188.pdf](https://arxiv.org/pdf/2105.09188.pdf)

- 参考实现：

  [https://github.com/csjliang/LPTN](https://github.com/csjliang/LPTN)

- 适配昇腾 AI 处理器的实现：

   [https://gitee.com/jiaolizc/ModelZoo-PyTorch/tree/master/PyTorch/contrib/cv/others/LPTN_ID2780_for_PyTorch](https://gitee.com/jiaolizc/ModelZoo-PyTorch/tree/master/PyTorch/contrib/cv/others/LPTN_ID2780_for_PyTorch)

- 通过Git获取对应commit_id的代码方法如下：

  ```
  git clone {repository_url}    # 克隆仓库的代码
  cd {repository_name}    # 切换到模型的代码仓目录
  git checkout  {branch}    # 切换到对应分支
  git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
  cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  
  ```

##  默认配置
```
name: LPTN_FiveK_paper 自定义项目名字
model_type: LPTNModel  自定义模型名字
num_gpu: 1  # set num_gpu: 0 for cpu mode 1表示GPU模式，0表示CPU模式
manual_seed: 10
```
## 数据集加载设置
```
  对于训练集:
    name: FiveK 数据集名字
    type: UnPairedImageDataset  不成对数据类型
    # (for lmdb)
    dataroot_gt: datasets/FiveK/FiveK_train_target.lmdb 转成lmdb格式的目标数据路径
    dataroot_lq: datasets/FiveK/FiveK_train_source.lmdb 转成lmdb格式的原始数据路径
    if_fix_size: true # 数据大小固定能够加快模型的训练
    gt_size: 256 # 训练集大小
    use_flip: true#翻转来增强训练集
    use_rot: true#旋转来增强训练集
    use_shuffle: true#数据清洗
    num_worker_per_gpu: 16#调节GPU利用率
    batch_size_per_gpu: 16#调节GPU显存率
    dataset_enlarge_ratio: 100#数据增强率
    prefetch_mode: npu#提前加载的模式
    pin_memory: true#锁页内存

  校验集:
    name: FiveK_val数据集名字
    type: PairedImageDataset 成对数据类型
    dataroot_gt: datasets/FiveK/FiveK_test_target.lmdb 转成lmdb格式的目标数据路径
    dataroot_lq: datasets/FiveK/FiveK_test_source.lmdb 转成lmdb格式的原始数据路径
```
# 网络结构参数
```
  type: LPTNPaper
  nrb_low: 5  #低频信息residual block数目
  nrb_high: 3 #高频信息residual block数目
  num_high: 3 #laplacian金字塔层数
```
# 训练参数
```
    type: Adam#优化器类型
    lr: !!float 1e-4#学习率
    weight_decay: 0#
    betas: [0.9, 0.99]#偏置
  
    type: MultiStepLR#学习率变化方式
    milestones: [50000, 100000, 200000, 300000]
    gamma: 0.5

  total_iter: 300000#总迭代次数
  warmup_iter: -1  # no warm up

  损失函数：
  pixel_opt:
    type: MSELoss#损失函数类型
    loss_weight: 1000
    reduction: mean
  gan_opt:
    type: GANLoss#GAN优化类型
    gan_type: standard
    real_label_val: 1.0
    fake_label_val: 0.0
    loss_weight: 1
  gp_opt:
    loss_weight: 100

```
# 校验集设置
```
val:
  val_freq: !!float 5e3#每迭代多少次执行校验
  save_img: true

  评价指标:
    psnr:
      type: calculate_psnr
      crop_border: 4
      test_y_channel: false
```
# logging设置
```
logger:
  print_freq: 100
  save_checkpoint_freq: !!float 5e3
  use_tb_logger: true
```

# 分布式训练设置
```
dist_params:
  backend: nccl
  port: 29500
```
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

```
cd LPTN 
```

## 快速上手

- 数据集准备

1. 模型训练使用FiveK数据集，可以通过执行如下代码实现数据集下载和数据格式创建
```
PYTHONPATH="./:${PYTHONPATH}" python3 scripts/data_preparation/download_datasets.py
PYTHONPATH="./:${PYTHONPATH}" python3 scripts/data_preparation/create_lmdb.py
```
2. 数据集也可从 [这里](https://drive.google.com/file/d/1oAORKd-TPnPwZvhcnEEJqc1ogT7KgFtx/view?usp=sharing)获取
3. 将获得的数据放在项目文件的datasets文件下

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练
    
      1. 配置训练参数可以在```options/train/LPTN/train_FiveK_paper.yml```文件里修改。
         
    
      2. 启动训练。
    
         启动单卡训练 
    
         ```
         PYTHONPATH="./:${PYTHONPATH}"  python3 train.py -opt options/train/LPTN/train_FiveK.yml
         ```
      3. log日志和训练文件会保存在./experiments/{name}
    - 单卡精度和性能测试
      1. 配置训练参数可以分别在```options/test/LPTN/test_FiveK.yml```和```options/test/LPTN/test_speed_FiveK.yml```文件里修改。
      2. 执行精度测试
         ```
         PYTHONPATH="./:${PYTHONPATH}" python3 test.py -opt options/test/LPTN/test_FiveK.yml
         ```
      3. 执行性能测试
         ```
         PYTHONPATH="./:${PYTHONPATH}" python3 test_speed.py -opt options/test/LPTN/test_speed_FiveK.yml
         ```
      4. 精度和性能测试结果会保存在```./results/{name}```目录下

##  训练结果

- 精度结果比对

| 精度指标项 | 论文发布 | GPU实测 | NPU实测 |
| ---------- | -------- | ------- | ------- |
| PSNR   | 22.12    | 22.8  | 22.3   |
| SSIM   | 0.878      | 0.885   | 0.8715 |


- 性能结果比对

| 性能指标项 |  GPU实测 | NPU实测 |
| ----------  | ------- | ------- |
|average duration(秒)   |   0.06675|0.07552 |



## 高级参考

## 脚本和示例代码

```
│  LICENSE
│  README.md
│  requirement.txt 安装依赖
│  test.py 精度测试
│  test_speed.py 性能测试
│  train.py 训练模型
│
├─ascend_function 昇腾相关函数
│  │  similar_api.py
│  │  __init__.py
│  │
│  └─__pycache__
│          similar_api.cpython-37.pyc
│          __init__.cpython-37.pyc
│
├─codes 代码脚本
│  │  test.py
│  │  test_speed.py
│  │  train.py
│  │
│  ├─data 数据处理
│  │  │  data_sampler.py
│  │  │  data_util.py
│  │  │  paired_image_dataset.py
│  │  │  prefetch_dataloader.py
│  │  │  transforms.py
│  │  │  unpair_image_dataset.py
│  │  │  __init__.py
│  │  │
│  │  └─__pycache__
│  │          data_sampler.cpython-37.pyc
│  │          data_util.cpython-37.pyc
│  │          paired_image_dataset.cpython-37.pyc
│  │          prefetch_dataloader.cpython-37.pyc
│  │          transforms.cpython-37.pyc
│  │          unpair_image_dataset.cpython-37.pyc
│  │          __init__.cpython-37.pyc
│  │
│  ├─metrics 度量准则
│  │  │  metric_util.py
│  │  │  psnr_ssim.py
│  │  │  __init__.py
│  │  │
│  │  └─__pycache__
│  │          psnr_ssim.cpython-37.pyc
│  │          __init__.cpython-37.pyc
│  │
│  ├─models 模型结构
│  │  │  base_model.py
│  │  │  lptn_model.py
│  │  │  lptn_test_model.py
│  │  │  lr_scheduler.py
│  │  │  __init__.py
│  │  │
│  │  ├─archs
│  │  │  │  arch_util.py
│  │  │  │  discriminator_arch.py
│  │  │  │  LPTN_arch.py
│  │  │  │  LPTN_paper_arch.py
│  │  │  │  __init__.py
│  │  │  │
│  │  │  └─__pycache__
│  │  │          discriminator_arch.cpython-37.pyc
│  │  │          LPTN_arch.cpython-37.pyc
│  │  │          LPTN_paper_arch.cpython-37.pyc
│  │  │          __init__.cpython-37.pyc
│  │  │
│  │  ├─losses
│  │  │  │  losses.py
│  │  │  │  loss_util.py
│  │  │  │  __init__.py
│  │  │  │
│  │  │  └─__pycache__
│  │  │          losses.cpython-37.pyc
│  │  │          loss_util.cpython-37.pyc
│  │  │          __init__.cpython-37.pyc
│  │  │
│  │  └─__pycache__
│  │          base_model.cpython-37.pyc
│  │          lptn_model.cpython-37.pyc
│  │          lptn_test_model.cpython-37.pyc
│  │          lr_scheduler.cpython-37.pyc
│  │          __init__.cpython-37.pyc
│  │
│  ├─utils 
│  │  │  dist_util.py
│  │  │  download_util.py
│  │  │  file_client.py
│  │  │  flow_util.py
│  │  │  img_util.py
│  │  │  lmdb_util.py
│  │  │  logger.py
│  │  │  misc.py
│  │  │  options.py
│  │  │  __init__.py
│  │  │
│  │  └─__pycache__
│  │          dist_util.cpython-37.pyc
│  │          file_client.cpython-37.pyc
│  │          img_util.cpython-37.pyc
│  │          lmdb_util.cpython-37.pyc
│  │          logger.cpython-37.pyc
│  │          misc.cpython-37.pyc
│  │          options.cpython-37.pyc
│  │          __init__.cpython-37.pyc
│  │
│  └─__pycache__
│          train.cpython-37.pyc
│
├─datasets 数据集存放文件夹
│  └─FiveK
├─experiments 训练结果文件夹
│  └─pretrained_models
│          put_models_here
│
├─Figs
│      LPTN_pipeline.jpg
│
├─options yml配置文件存放文件夹
│  ├─test
│  │  └─LPTN
│  │          test_FiveK.yml
│  │          test_speed_FiveK.yml
│  │
│  └─train
│      └─LPTN
│              train_FiveK.yml
│              train_FiveK_paper.yml
│
├─results 测试结果文件夹
└─scripts 下载和处理数据脚本
    │  download_gdrive.py
    │
    └─data_preparation
            create_lmdb.py
            download_datasets.py


```

## 脚本参数

```
 -opt options/{name} 指定的yml文件
```

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md

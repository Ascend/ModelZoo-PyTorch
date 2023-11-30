-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Click-Through-Rate**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.10.26**

**大小（Size）：237KB**

**框架（Framework）：Pytorch 1.5.0**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于Pytorch框架的广告点击率预测模型xDeepFM训练代码** 

<h2 id="概述.md">概述</h2>

xDeepFM模型是用于预测广告点击率的模型，为了实现自动学习显式的高阶特征交互，同时使得交互发生在向量级上，xDeepFM首先提出了一种新的名为压缩交互网络（Compressed Interaction Network，简称CIN）的模型 。

- 参考论文：

    [Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang. “xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems.”](https://arxiv.org/pdf/1803.05170.pdf) 

- 参考实现

    

- 适配昇腾 AI 处理器的实现：
  
  [https://gitee.com/ascend/pytorch/blob/2.0.4.tr5/docs/zh/PyTorch%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97/PyTorch%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97.md#%E5%AE%89%E8%A3%85PyTorch%E6%A1%86%E6%9E%B6md](https://gitee.com/ascend/pytorch/blob/2.0.4.tr5/docs/zh/PyTorch%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97/PyTorch%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97.md#%E5%AE%89%E8%A3%85PyTorch%E6%A1%86%E6%9E%B6md)

## 默认配置<a name="section91661242121611"></a>

- 训练数据集、测试数据集预处理：

  - 运行 data/forXDeepFM/xDeepFM_dataPreprocess_PyTorch.py 实现数据集预处理
  - 运行命令(切换到forXDeepFM目录)：python3 xDeepFM_dataPreprocess_PyTorch.py

- 训练超参

  - Batch size: 2048
  - Learning rate(LR): 0.01
  - Optimizer: apex.optimizers.NpuFusedSGD
  - Train epoch: 5


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 是    |
| 并行数据  | 是    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

已默认开启混合精度。


<h2 id="训练环境准备.md">训练环境准备</h2>

- 硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://gitee.com/ascend/pytorch/blob/2.0.4.tr5/docs/zh/PyTorch%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97/PyTorch%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97.md#%E5%AE%89%E8%A3%85PyTorch%E6%A1%86%E6%9E%B6md)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。

    当前模型支持的镜像列表如表1所示。

    **表 1** 镜像列表

    <a name="zh-cn_topic_0000001074498056_table1519011227314"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001074498056_row0190152218319"><th class="cellrowborder" valign="top" width="47.32%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001074498056_p1419132211315"><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><em id="i1522884921219"><a name="i1522884921219"></a><a name="i1522884921219"></a>镜像名称</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="25.52%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001074498056_p75071327115313"><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><em id="i1522994919122"><a name="i1522994919122"></a><a name="i1522994919122"></a>AscendPytorch版本</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="27.16%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001074498056_p1024411406234"><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><em id="i723012493123"><a name="i723012493123"></a><a name="i723012493123"></a>配套CANN版本</em></p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001074498056_row71915221134"><td class="cellrowborder" valign="top" width="47.32%" headers="mcps1.2.4.1.1 "><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><ul id="zh-cn_topic_0000001074498056_ul81691515131910"><li><em id="i82326495129"><a name="i82326495129"></a><a name="i82326495129"></a>ARM架构：<a href="#" target="_blank" rel="noopener noreferrer">ascend-pytorch-arm</a></em></li><li><em id="i18233184918125"><a name="i18233184918125"></a><a name="i18233184918125"></a>x86架构：<a href="#" target="_blank" rel="noopener noreferrer">ascend-pytorch-x86</a></em></li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>2.0.2</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="#" target="_blank" rel="noopener noreferrer">5.0.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


<h2 id="快速上手.md">快速上手</h2>

- 数据集准备
1. 模型训练使用[Criteo数据集](https://www.kaggle.com/datasets/mrkmakr/criteo-dataset)。

2. 数据集训练前需要做预处理操作。

3. 数据集处理后，对应路径下会产生训练集和测试集数据。
   

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/pytorch/blob/2.0.4.tr5/docs/zh/PyTorch%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97/PyTorch%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97.md#%E5%AE%89%E8%A3%85PyTorch%E6%A1%86%E6%9E%B6md)

- 单卡训练 

  1. 配置训练参数。

     首先准备好超参数：指定具体NPU, 若果是GPU则不需要指定

  2. 启动训练。

     启动单卡GPU训练(切换到Model目录)

     ```
     python3 xDeepFM_PyTorch_GPU_1P.py
     ```

     启动单卡NPU训练(切换到Model目录)

     ```
     python3 xDeepFM_PyTorch_NPU_1P.py
     ```


<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

  数据集要求如下：

  1. 获取数据。

     将下载的train.txt文件放到 data/Criteo 目录后，进行预处理数据集，产生的数据集存放路径如下：

     - 训练集： /forXDeepFM/train_data
     - 测试集： /forXDeepFM/test_data

  2. 数据集文件结构，目录参考：

        ```
            |--|train_data
            |  part-0
            |  part-1
            |  part-2
            |  ...
            |--|test_data
            |  part-9
            |  part-42
            |  ...
        ```

<h2 id="高级参考.md">高级参考</h2>

## 示例代码<a name="section08421615141513"></a>

```
├── data
│    ├── Criteo                    
│         ├── forXDeepFM
│              ├──xDeepFM_dataPreprocess.py    //数据预处理代码
│              ├── aid_data   
│              ├── raw_data 
│              ├── test_data                   //测试集
│              ├── train_data                  //训练集
│         ├──train.txt
│         ├──util.py
│         ├──__init__.py   
│    ├──__init__.py   
├── Model
│    ├── common                    
│    ├── util   
│    ├──xDeepFM_PyTorch.py                     //xDeepFM_PyTorch原代码
│    ├──xDeepFM_PyTorch_GPU_1P.py              //GPU运行    
│    ├──xDeepFM_PyTorch_NPU_1P.py              //NPU迁移              
├── output
│    ├── log                                   //训练日志
│    ├── prof                          
├── util
│    ├──load_data_util.py                            
│    ├──train_model_util_PyTorch.py
│    ├──train_model_util_TensorFlow.py
├──LICENSE
├──README.md
```

## 训练过程<a name="section1589455252218"></a>

- 训练日志中包括如下信息。

```
[92m08-08 12:29:16[0m gpus_per_node:1
[92m08-08 12:29:31[0m Train Epoch: 1 [0 / 41130217 (0%)]	Loss:0.753818
[92m08-08 12:32:41[0m Train Epoch: 1 [2048000 / 41130217 (5%)]	Loss:0.485503
[92m08-08 12:35:50[0m Train Epoch: 1 [4096000 / 41130217 (10%)]	Loss:0.462001
[92m08-08 12:38:58[0m Train Epoch: 1 [6144000 / 41130217 (15%)]	Loss:0.480042
[92m08-08 12:42:07[0m Train Epoch: 1 [8192000 / 41130217 (20%)]	Loss:0.467268
...
[92m08-08 17:53:26[0m Train Epoch: 5 [34816000 / 41130217 (85%)]	Loss:0.450160
[92m08-08 17:56:35[0m Train Epoch: 5 [36864000 / 41130217 (90%)]	Loss:0.476156
[92m08-08 17:59:30[0m Train Epoch: 5 [38912000 / 41130217 (95%)]	Loss:0.465982
[92m08-08 18:02:38[0m Train Epoch: 5 [40960000 / 41130217 (100%)]	Loss:0.475123
[92m08-08 18:02:42[0m FPS: 0.66
[92m08-08 18:08:46[0m Roc AUC: 0.80064
[92m08-08 18:08:46[0m Test set: Average loss: 0.44814
```

## 运行结果<a name="section1465595372416"></a>
- GPU
|Epochs|AUC|LogLoss|
|-----|---|-------|
|1st|0.79872|0.45048|
|2nd|0.79959|0.44973|
|3rd|0.80052|0.44896|
|4th|0.80058|0.44855|
|5th|0.80064|0.44814|

- NPU
|Epochs|AUC|LogLoss|
|-----|---|-------|
|1st|0.79868|0.45095|
|2nd|0.79981|0.44992|
|3rd|0.80034|0.44933|
|4th|0.80041|0.44931|
|5th|0.80045|0.44927|



# 公网地址说明

代码涉及公网地址参考 public_address_statement.md

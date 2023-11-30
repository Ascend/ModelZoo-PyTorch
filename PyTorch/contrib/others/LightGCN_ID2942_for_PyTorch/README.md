-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Recommendation system**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.11.7**

**大小（Size）：63.9MB**

**框架（Framework）：Pytorch 1.5.0**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于Pytorch框架的用于推荐任务的简化并增强的图卷积网络训练代码** 

<h2 id="概述.md">概述</h2>

LightGCN是将图[卷积神经网络](https://so.csdn.net/so/search?q=卷积神经网络&spm=1001.2101.3001.7020)应用于推荐系统当中，是对神经图协同过滤（NGCF）算法的优化和改进。

- 参考论文：

    [SIGIR 2020. Xiangnan He, Kuan Deng ,Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang(2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126)

- 参考实现

    

- 适配昇腾 AI 处理器的实现：
  
  [https://gitee.com/ascend/pytorch/blob/2.0.4.tr5/docs/zh/PyTorch%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97/PyTorch%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97.md#%E5%AE%89%E8%A3%85PyTorch%E6%A1%86%E6%9E%B6md](https://gitee.com/ascend/pytorch/blob/2.0.4.tr5/docs/zh/PyTorch%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97/PyTorch%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97.md#%E5%AE%89%E8%A3%85PyTorch%E6%A1%86%E6%9E%B6md)

## 默认配置<a name="section91661242121611"></a>

- 训练数据集、测试数据集预处理：

  
  - 运行命令(切换到code目录)：python3 main.py

- 训练超参

  - Batch size: 2048
  - Learning rate(LR): 0.001
  - Optimizer: apex.optimizers.NpuFusedSGD
  - Layer:4


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

  提供了三个经过处理的数据集：Gowalla、Yelp2018和Amazon book，以及一个小数据集LastFM（dataloader.py）。数据集存放在项目目录下data文件中。


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/pytorch/blob/2.0.4.tr5/docs/zh/PyTorch%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97/PyTorch%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97.md#%E5%AE%89%E8%A3%85PyTorch%E6%A1%86%E6%9E%B6md)

- 单卡训练 

  1. 配置训练参数。

     首先准备好超参数：指定具体NPU, 若果是GPU则不需要指定
     
     在文件根目录下code文件夹内修改world.py中的ROOT_PATH 

  2. 启动训练。

     启动单卡GPU训练(切换到code目录)

     ```
     python3 main.py
     ```

     启动单卡NPU训练(切换到code目录)

     ```
     python3 main.py
     ```

<h2 id="高级参考.md">高级参考</h2>

## 示例代码<a name="section08421615141513"></a>

```
├── code
│    ├── checkpoints 
│    ├── sources              
│         ├── sampling.cpp
│    ├── dataloader.py                //数据预处理代码
│    ├── main.py                      //主方法
│    ├── model.py                     //模型
│    ├── parse.py                     //参数
│    ├── Procedure.py                 
│    ├── register.py
│    ├── utils.py
│    ├── world.py  
│    ├──__init__.py   
├── data
│    ├── amazon-book 
│         ├── train.txt               //训练集
│         ├── test.txt                //测试集       
│         ├── user_list.txt
│         ├── item_list.txt
│         ├── README.md
│    ├── gowalla   
│    ├── lastfm                     
│    ├── yelp2018                                     
├── imgs
│    ├── tf.jpg                                   
│    ├── torch.png                         
├──.gitignore
├──requirements.txt
├──README.md
```

## 训练过程<a name="section1589455252218"></a>

- 训练日志中包括如下信息。
```
[TEST]
{'precision': array([0.04360305]), 'recall': array([0.16279419]), 'ndcg': array([0.10995968])}
EPOCH[981/1000] loss0.014-|Sample:19.08|-15.86081600189209-129.12324307625076
EPOCH[982/1000] loss0.014-|Sample:20.66|-15.889200210571289-128.89257941613948
EPOCH[983/1000] loss0.014-|Sample:19.26|-15.9640634059906-128.28813992504422
EPOCH[984/1000] loss0.014-|Sample:21.07|-15.882123708724976-128.9500093035363
EPOCH[985/1000] loss0.014-|Sample:20.87|-15.907534122467041-128.7440268386728
EPOCH[986/1000] loss0.014-|Sample:19.24|-15.966394901275635-128.2694066295689
EPOCH[987/1000] loss0.014-|Sample:19.34|-15.967133283615112-128.26347495336452
EPOCH[988/1000] loss0.014-|Sample:19.55|-16.01818013191223-127.85472401573699
EPOCH[989/1000] loss0.014-|Sample:21.30|-15.967095851898193-128.26377564186356
EPOCH[990/1000] loss0.014-|Sample:21.88|-15.978184938430786-128.17475876588105

```
## 运行结果<a name="section1465595372416"></a>
- GPU

  | EPOCH | loss  | Recall | ndcg   |
  | ----- | ----- | ------ | ------ |
  | 110   | 0.024 | 0.1257 | 0.0752 |
  | 120   | 0.023 | 0.1282 | 0.0771 |
  | 130   | 0.022 | 0.1298 | 0.0779 |

- NPU

  | EPOCH | loss  | Recall | ndcg   |
  | ----- | ----- | ------ | ------ |
  | 110   | 0.024 | 0.1085 | 0.0674 |
  | 120   | 0.023 | 0.1107 | 0.0690 |
  | 130   | 0.022 | 0.1128 | 0.0702 |

  
# 公网地址说明

代码涉及公网地址参考 public_address_statement.md

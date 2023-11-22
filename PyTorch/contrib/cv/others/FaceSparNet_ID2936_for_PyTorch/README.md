
- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)

<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：人脸超分辨**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.9.7**

**大小（Size）：139KB**

**框架（Framework）：Pytroch**

**模型格式（Model Format）：ckpt**

**精度（Precision）：混合精度**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于Pytroch框架的SparNet网络复现**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

一般的图像超分辨率技术在应用于低分辨率人脸图像时难以恢复详细的人脸结构。最近针对人脸图像定制的基于深度学习的方法通过与额外任务（如人脸解析和界标预测）联合训练，实现了改进的性能。然而，多任务学习需要额外的手动标记数据。此外，现有的大多数工作只能生成相对低分辨率的人脸图像（例如128×128），因此其应用受到限制。在本文中，我们介绍了一种新的空间注意力剩余网络（SPARNet），该网络基于我们新提出的人脸注意力单元（FAU），用于人脸超分辨率。具体而言，我们将空间注意力机制引入到香草剩余块。这使得卷积层能够自适应地自举与关键面部结构相关的特征，并且较少关注那些特征不丰富的区域。这使得训练更加有效和高效，因为关键面部结构仅占面部图像的很小部分。注意力图的可视化显示，我们的空间注意力网络可以很好地捕捉关键人脸结构，即使是非常低分辨率的人脸（如16×16）。对各种度量（包括PSNR、SSIM、身份相似性和地标检测）的定量比较证明了我们的方法优于现有技术。我们进一步使用多尺度鉴别器扩展SPARNet，称为SPARNetHD，以产生高分辨率结果（即512×512）。我们表明，使用合成数据训练的SPARNetHD不仅可以为合成退化的人脸图像生成高质量和高分辨率的输出，而且对真实世界的低质量人脸图像具有良好的泛化能力。

- 参考论文：

  [https://arxiv.org/abs/2012.01211](https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fabs%2F1810.04805)

- 参考实现：

   https://github.com/chaofengc/Face-SPARNet
- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ddwolf0526/ModelZoo-PyTorch_git/tree/master/PyTorch/contrib/cv/others/FaceSparNet_ID2936

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换

    

## 默认配置<a name="section91661242121611"></a>

 -   网络结构
	多层网络结构，具体请参考论文。

-   训练超参（单卡）：
   --  name SPARNet_S16_V4_Attn2D 
   --model sparnet 
    --Gnorm "bn" 
    --lr 0.0002 
    --beta1 0.9 
    --scale_factor 8 
    --load_size 128 
    --dataroot ../celeba_crop_train 
    --dataset_name celeba 
    --batch_size 32 
    --total_epochs 20 
    --visual_freq 100 
    --print_freq 10 
    --save_latest_freq 500


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 数据并行   | 是       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

该代码训练时已经采用混合精度训练，主要代码如下：

    self.optimizer_G  =  apex.optimizers.NpuFusedAdam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
    
    self.optimizers  = [self.optimizer_G]
    
    self.netG, self.optimizer_G  = amp.initialize(self.netG, self.optimizer_G, opt_level='O2', loss_scale=128.0, combine_grad=True)
    
    with amp.scale_loss(self.loss_Pix, self.optimizer_G) as  scaled_loss:
    
		    scaled_loss.backward()

<h2 id="训练环境准备.md">训练环境准备</h2>

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

1.训练使用数据集为Celeba
2.测试数据集下载提供以下两种方式：

- [GoogleDrive](https://drive.google.com/drive/folders/1PZ_TP77_rs0z56WZausgK0m2oTxZsgB2?usp=sharing)  
- [BaiduNetDisk](https://pan.baidu.com/s/1zYimaAnIgMIKBf9KANpxog), extract code: `2nax` 



## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练

         网络共包含1个训练，即SparNet的网络训练。
	 ```
			python newtrain.py
	```
		
 <h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
|--LICENSE
|--README.md									#说明文档
|--newtrain.py									#训练代码
|--modelarts_entry_perf.py
|--modelarts_entry_acc.py 								
|--requirements.txt		   						#所需依赖

|--data			           						#数据预处理
|	|--__init__.py
|	|--base_dataset.py
|	|--celeba_dataset.py
|	|--ffhq_dataset.py
|	|--image_folder.py
|	|--single_dataset.py

|--models			           					#训练所需模型
|	|--__init__.py
|	|--base_model.py
|	|--blocks.py
|	|--loss.py
|	|--networks.py
|	|--sparnet.py
|	|--sparnet_model.py
|	|--sparnethd_model.py

|--options		           						#模型参数设定
|	|--__init__.py
|	|--base_options.py
|	|--test_options.py
|	|--train_options.py

|--utils		           					    #工具包
|	|--logger.py
|	|--timer.py
|	|--utils.py

|--test		           						   #数据预处理
|	|--train_full_1p.sh
|	|--train_performance_1p.sh

```

## 脚本参数<a name="section6669162441511"></a>

```
 --name SPARNet_S16_V4_Attn2D    #模型名称
 --model sparnet 				 #采用的模型
 --Gnorm "bn" 
 --lr 0.0002 					 #学习率
 --beta1 0.9 					 #衰减率
 --scale_factor 8 				 #尺度
 --load_size 128 				 #载入数据集尺寸
 --dataroot ../celeba_crop_train #训练数据集路径
 --dataset_name celeba           #训练集名称
 --batch_size 32 				 #batchsize大小
 --total_epochs 20               #epoch大小
 --visual_freq 100 
 --print_freq 10 			     #打印频率
 --save_latest_freq 500			 #模型保存频率
```


## 训练过程<a name="section1589455252218"></a>

如果要进行SparNet网络训练，请先把训练数据和测试数据放到对应的路径下面。之后运行newtrain.py文件即可
		训练SparNet网络，该网络设置的npu卡号为0，请根据实际需要修改卡号。


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md
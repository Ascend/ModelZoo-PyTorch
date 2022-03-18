-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：huawei**

**应用领域（Application Domain）：CV**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.05.24**

**大小（Size）**_**：【深加工】**

**框架（Framework）：PyTorch 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：【深加工】**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：【深加工】**

**描述（Description）：基于PyTorch框架的AutoAugment训练代码**

<h2 id="概述.md">概述</h2>

-    该存储库包含基于AutoAugment：从PyTorch中实现的数据中学习增强策略的AutoAugment（仅使用纸张的最佳策略）的代码。

    -   参考实现：
        
        ```
        https://github.com/4uiiurz1/pytorch-auto-augment
        ```
    
-   适配昇腾 AI 处理器的实现：【深加工】
    
        ```
        https://gitee.com/ascend/modelzoo/tree/master/built-in/PyTorch/Official/cv/image_classification/AUTOAUGMENT_ID0792_for_PyTorch
        branch=master
        commit_id=
        ```


    -   通过Git获取对应commit\_id的代码方法如下：
    
        ```
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
        ```

## 默认配置【深加工】<a name="section91661242121611"></a>
-   网络结构
    -   初始学习率为0.06，使用Cosine learning rate
    -   优化器：Momentum
    -   单卡batchsize：256
    -   8卡batchsize：128*8
    -   总Epoch数设置为150
    -   Weight decay为0.0001，Momentum为0.9
    -   Label smoothing参数为0.1

-   训练数据集预处理（当前代码以ImageNet/train为例，仅作为用户参考示例）：
    -   图像的输入尺寸为224\*224
    -   随机裁剪图像尺寸
    -   随机水平翻转图像
    -   根据平均值和标准偏差对输入图像进行归一化

-   测试数据集预处理（当前代码以ImageNet验证集为例，仅作为用户参考示例）：
    -   图像的输入尺寸为224\*224（将图像最小边缩放到256，同时保持宽高比，然后在中心裁剪图像）
    -   根据平均值和标准偏差对输入图像进行归一化

-   训练超参（单卡）：
    -   Batch size: 128
    -   Momentum: 0.9
    -   LR scheduler: cosine
    -   Learning rate\(LR\): 0.06
    -   Weight decay: 0.0001
    -   Label smoothing: 0.1
    -   Train epoch: 150


## 支持特性【深加工】<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 数据并行   | 是       |


## 混合精度训练【深加工】<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度【深加工】<a name="section20779114113713"></a>
相关代码示例。



```
run_config = NPURunConfig(
        model_dir=self.config.model_dir,
        session_config=session_config,
        keep_checkpoint_max=5,
        save_checkpoints_steps=5000,
        enable_data_pre_proc=True,
        iterations_per_loop=iterations_per_loop,
        precision_mode='allow_mix_precision',
        hcom_parallel=True
      ）
```

<h2 id="训练环境准备.md">训练环境准备</h2>

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

    当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

    **表 1** 镜像列表

    <a name="zh-cn_topic_0000001074498056_table1519011227314"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001074498056_row0190152218319"><th class="cellrowborder" valign="top" width="47.32%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001074498056_p1419132211315"><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><em id="i1522884921219"><a name="i1522884921219"></a><a name="i1522884921219"></a>镜像名称</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="25.52%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001074498056_p75071327115313"><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><em id="i1522994919122"><a name="i1522994919122"></a><a name="i1522994919122"></a>镜像版本</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="27.16%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001074498056_p1024411406234"><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><em id="i723012493123"><a name="i723012493123"></a><a name="i723012493123"></a>配套CANN版本</em></p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001074498056_row71915221134"><td class="cellrowborder" valign="top" width="47.32%" headers="mcps1.2.4.1.1 "><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><ul id="zh-cn_topic_0000001074498056_ul81691515131910"><li><em id="i82326495129"><a name="i82326495129"></a><a name="i82326495129"></a>ARM架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm" target="_blank" rel="noopener noreferrer">ascend-tensorflow-arm</a></em></li><li><em id="i18233184918125"><a name="i18233184918125"></a><a name="i18233184918125"></a>x86架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86" target="_blank" rel="noopener noreferrer">ascend-tensorflow-x86</a></em></li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>20.2.0</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">20.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

1. 在训练脚本中指定数据集路径，可正常使用。

## 模型训练【深加工】<a name="section715881518135"></a>
- 下载训练脚本。
- 检查scripts/目录下是否有存在8卡IP的json配置文件“8p.json”。
  
```
 {"group_count": "1","group_list":     
                [{"group_name": "worker","device_count": "8","instance_count": "1", "instance_list":      
                                         [{"devices":                               
                                         [{"device_id":"0","device_ip":"192.168.100.101"},                
                                         {"device_id":"1","device_ip":"192.168.101.101"},                   
                                         {"device_id":"2","device_ip":"192.168.102.101"},                  
                                         {"device_id":"3","device_ip":"192.168.103.101"},                
                                         {"device_id":"4","device_ip":"192.168.100.100"},                 
                                         {"device_id":"5","device_ip":"192.168.101.100"},                  
                                         {"device_id":"6","device_ip":"192.168.102.100"},                   
                                         {"device_id":"7","device_ip":"192.168.103.100"}],                 
                                     "pod_name":"npu8p",        "server_id":"127.0.0.1"}]}],"status": "completed"}
```

- 开始训练。
  
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
    

    2. 单卡训练
       
        2.1 设置单卡训练参数（脚本位于./AUTOAUGMENT_ID0792_for_PyTorch/test/train_performance_1p.sh），示例如下。请确保下面例子中的“input_file和bert_config_file”修改为用户数据集的路径。
            
        
        ```
        `nohup python3 train.py \
            --epochs $train_epochs`
        ```
        
        
        
        2.2 单卡训练指令（脚本位于./AUTOAUGMENT_ID0792_for_PyTorch/test/train_performance_1p.sh） 

```
        `bash train_performance_1p.sh`
```

3. 8卡训练【深加工】

   3.1 设置8卡训练参数（脚本位于AlexNet_for_TensorFlow/scripts/train_alexnet_8p.sh），示例如下。

   请确保下面例子中的“--data_dir”修改为用户生成的tfrecord的实际路径。
   ```
   ` --data_dir=/data/slimImagenet`
   ```
   3.2 8卡训练指令（脚本位于AlexNet_for_TensorFlow/scripts/run_npu_8p.sh）

            ` bash run_npu_8p.sh`

<h2 id="开始测试.md">开始测试【深加工】</h2>

 - 参数配置
    1. 修改脚本启动参数（脚本位于AlexNet_for_TensorFlow/scripts/train_alexnet_8p.sh），将mode设置为evaluate，如下所示：

        `--mode=evaluate`

    2. 增加checkpoints的路径，请用户根据checkpoints实际路径进行配置。
       
        `--checkpoint_dir=./results/0/model_8p/`

    3. 将“`rm -rf ${EXEC_DIR}/${RESULTS}/${device_id}/*`”替换为“`#rm -rf ${EXEC_DIR}/${RESULTS}/${device_id}/*`”，如下所示：

        `#rm -rf ${EXEC_DIR}/${RESULTS}/${device_id}/*`

- 执行测试指令
  
    1. 上述文件修改完成之后，执行8卡测试指令（脚本位于AlexNet_for_TensorFlow/scripts/run_npu_8p.sh）
       
        `bash scripts/run_npu_8p.sh`

<h2 id="迁移学习指导.md">迁移学习指导【深加工】</h2>

- 数据集准备。

    1.  获取数据。
        请参见“快速上手”中的数据集准备，需要将数据集转化为tfrecord格式。类别数可以通过训练参数中的num_classes来设置。
    2.  数据集每个类别所占比例大致相同。
    3.  数据目录结构如下：
        
        ```
                |--|imagenet_tfrecord
                |   train-00000-of-01024
                |   train-00001-of-01024
                |   train-00002-of-01024
                |   ...
                |   validation-00000-of-00128
                |   validation-00000-of-00128
                |   ...
        
        ```

    4.  设置合理的数据集预处理方法（裁剪大小、随机翻转、标准化）。
        
        ```
                def parse_and_preprocess_image_record(config, record, height, width,brightness, contrast, saturation, hue,
                                              distort, nsummary=10, increased_aug=False, random_search_aug=False):
                with tf.name_scope('preprocess_train'):
                    image = crop_and_resize_image(config, record, height, width, distort)   #解码，80%中心抠图并且Resize[224 224]
                    if distort:
                        image = tf.image.random_flip_left_right(image)            #随机左右翻转
                        image = tf.clip_by_value(image, 0., 255.)                     #归一化
                image = normalize(image)                  #减均值[121.0, 115.0, 100.0]，除方差[70.0, 68.0, 71.0]
                image = tf.cast(image, tf.float16)
                return image
        ```


- 修改训练脚本。
    1.  修改配置文件。

        1.1 使用自有数据集进行分类，如需将分类类别修改为10，修改alexnet/model.py，将depth=1000设置为depth=10。
        
            `labels_one_hot = tf.one_hot(labels, depth=1000)`

        1.2 修改alexnet/alexnet.py，将num_classes=1000修改为num_classes=10。将“x, 1000”设置为“x, 10”。

        
        ```
        def inference_alexnet_impl(inputs, num_classes=1000, is_training=True):
                    .  .  .
                    def inference_alexnet_impl_he_uniform(inputs,num_classes=1000, is_training=True):
                    .  .  .
                    x = tf.layers.dense(x, 1000, activation=tf.nn.relu, use_bias=True,
                    kernel_initializer= tf.variance_scaling_initializer(scale=scale, mode ='fan_in',distribution='uniform'))
                    .  .  .
                    def inference(inputs,version="xavier",num_classes=1000, is_training=False):
        ```


​    
​    2.  加载预训练模型。
​        
​        配置文件增加参数，修改文件train.py（具体配置文件名称，用户根据自己实际名称设置），增加以下参数。    
​    
​        ```
​        parser.add_argument('--restore_path', default='/code/ckpt0/model.ckpt-188000',
​                    help="""restore path""")            #配置预训练ckpt路径
​        parser.add_argument('--restore_exclude', default=['dense_2'],
​                    help="""restore_exclude""")  #不加载预训练网络中FC层权重
​        ```
​    3. 模型加载修改，修改文件alexnet/model.py，增加以下代码行。
​    
        ```
        assert (mode == tf.estimator.ModeKeys.TRAIN)
        #restore ckpt for finetune，
        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=self.config.restore_exclude)
        tf.train.init_from_checkpoint(self.config.restore_path,{v.name.split(':')[0]: v for v in variables_to_restore})
        ```


-  模型训练。

    请参考“快速上手”章节。

-  模型评估。
   
    可以参考“模型训练”中训练步骤。

<h2 id="高级参考.md">高级参考【深加工】</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

    ├── README.md                                //说明文档
    ├── requirements.txt						 //依赖
    ├──test										 
    │    ├──train_performance_1p.sh				 //单卡训练脚本
    │    ├──env.sh								 //环境变量
    ├──albert_config                     	     //网络配置
    ├──create_pretraining_data.sh           	 //预处理执行脚本
    ├──create_pretraining_data.py          	     //预处理脚本
    ├──run_pretraining.py              		     //预训练脚本


## 脚本参数【深加工】<a name="section6669162441511"></a>

```
    --data_dir                        train data dir, default : path/to/data
    --num_classes                     number of classes for dataset. default : 1000
    --batch_size                      mini-batch size ,default: 128 
    --lr                              initial learning rate,default: 0.06
    --max_epochs                      total number of epochs to train the model:default: 150
    --warmup_epochs                   warmup epoch(when batchsize is large), default: 5
    --weight_decay                    weight decay factor for regularization loss ,default: 1e-4
    --momentum                        momentum for optimizer ,default: 0.9
    --label_smoothing                 use label smooth in CE, default 0.1
    --save_summary_steps              logging interval,dafault:100
    --log_dir                         path to save checkpoint and log,default: ./model_1p
    --log_name                        name of log file,default: alexnet_training.log
    --save_checkpoints_steps          the interval to save checkpoint,default: 1000
    --mode                            mode to run the program (train, evaluate), default: train
    --checkpoint_dir                  path to checkpoint for evaluation,default : None
    --max_train_steps                 max number of training steps ,default : 100
    --synthetic                       whether to use synthetic data or not,default : False
    --version                         weight initialization for model,default : he_uniorm
    --do_checkpoint                   whether to save checkpoint or not, default : True
    --rank_size                       number of npus to use, default : 1
```

## 训练过程【深加工】<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。
将训练脚本（train_alexnet_1p.sh,train_alexnet_8p.sh）中的data_dir设置为训练数据集的路径。具体的流程参见“模型训练”的示例。
模型存储路径为results/1p或者results/8p，包括训练的log以及checkpoints文件。以8卡训练为例，loss信息在文件results/8p/0/model_8p/alexnet_training.log中，示例如下。

```
step:   700  epoch:  0.6  FPS:33558.6, loss: 6.754, total_loss: 7.786  lr:0.00671  batch_time:3.051382
step:   800  epoch:  0.6  FPS:33828.3, loss: 6.840, total_loss: 7.871  lr:0.00767  batch_time:3.027051
step:   900  epoch:  0.7  FPS:32321.9, loss: 6.785, total_loss: 7.814  lr:0.00863  batch_time:3.168133
step:  1000  epoch:  0.8  FPS:34254.1, loss: 6.777, total_loss: 7.805  lr:0.00959  batch_time:2.989423
step:  1100  epoch:  0.9  FPS:32974.6, loss: 6.770, total_loss: 7.795  lr:0.01055  batch_time:3.105422
step:  1200  epoch:  1.0  FPS:32637.9, loss: 6.715, total_loss: 7.738  lr:0.01151  batch_time:3.137457
step:  1300  epoch:  1.0  FPS:33125.1, loss: 6.629, total_loss: 7.650  lr:0.01247  batch_time:3.091308
step:  1400  epoch:  1.1  FPS:32611.2, loss: 6.574, total_loss: 7.593  lr:0.01343  batch_time:3.140028
step:  1500  epoch:  1.2  FPS:33074.4, loss: 6.516, total_loss: 7.531  lr:0.01439  batch_time:3.096053
step:  1600  epoch:  1.3  FPS:35500.1, loss: 6.504, total_loss: 7.517  lr:0.01535  batch_time:2.884497
step:  1700  epoch:  1.4  FPS:32006.8, loss: 6.348, total_loss: 7.358  lr:0.01631  batch_time:3.199324
step:  1800  epoch:  1.4  FPS:34176.2, loss: 6.363, total_loss: 7.371  lr:0.01727  batch_time:2.996241
```


## 推理/验证过程【深加工】<a name="section1465595372416"></a>

在150 epoch训练执行完成后，请参见“模型训练”中的测试流程，需要修改脚本启动参数（脚本位于scripts/train_alexnet_8p.sh）将mode设置为evaluate，增加checkpoints的路径，“rm -rf ${EXEC_DIR}/${RESULTS}/${device_id}/*”替换为“#rm -rf ${EXEC_DIR}/${RESULTS}/${device_id}/*”然后执行脚本。

bash run_npu_8p.sh

该脚本会自动执行验证流程，验证结果若想输出至文档描述文件，则需修改启动脚本参数，否则输出至默认log文件（./results/8p/0/model_8p/alexnet_training.log）中。

```
Evaluating
Validation dataset size: 49921
step      epoch  top1    top5     loss   checkpoint_time(UTC)
6300     1.0    18.502   39.33    4.78  2020-06-18 11:18:45
12600    10.0   29.946   54.90    3.99  2020-06-18 11:42:07
125200   100.0  53.015   77.11    2.91  2020-06-18 12:40:13
187700   150.0  60.120   82.06    2.57  2020-06-18 13:12:14
Finished evaluation

```
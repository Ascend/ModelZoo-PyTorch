# TextCNN网络迁移

```shell
网络源码路径：https://github.com/gaussic/text-classification-cnn-rnn
论文地址：https://arxiv.org/abs/1408.5882
```

### 1、模型概述：

使用卷积神经网络进行中文文本分类 

#####  

### 2、数据集：

名称：全量数据集[THUCTC](http://thuctc.thunlp.org/) 

路径：训练子集 10.136.165.4服务器:/turingDataset/CarPeting_Textcnn/cnews

### 3、依赖安装: 

`requirements.txt随网络模型归档至gitlab仓。`

```shell
tensorflow==1.15.0
numpy
scikit-learn
scipy

```

### 4、训练详细调测步骤：

#### 训练步骤：

(是否需要手工迁移)

执行：

```shell
全量精度：bash run_npu_1p_acc.sh
CI性能： bash run_npu_1p_perf.sh
```

#### 3、NPU训练结果：

##### `ckpt`文件和`graph`文件存放路径：

` 10.136.165.4服务器:/turingDataset/results/CarPeting_TF_Textcnn/ckpt_npu路径`

```shell
精度性能示例：（部分打屏结果如下）

Iter:    100, Train Loss:   0.87, Train Acc:  75.00%, Val Loss:    1.1, Val Acc:  68.00%, Time: 0:00:36 *
Iter:    200, Train Loss:   0.33, Train Acc:  87.50%, Val Loss:    0.7, Val Acc:  77.28%, Time: 0:00:38 *
Iter:    300, Train Loss:   0.12, Train Acc:  96.88%, Val Loss:   0.48, Val Acc:  85.34%, Time: 0:00:41 *
Iter:    400, Train Loss:   0.34, Train Acc:  90.62%, Val Loss:   0.41, Val Acc:  88.04%, Time: 0:00:43 *
Iter:    500, Train Loss:    0.2, Train Acc:  92.19%, Val Loss:   0.35, Val Acc:  90.38%, Time: 0:00:45 *
Iter:    600, Train Loss:   0.35, Train Acc:  95.31%, Val Loss:   0.31, Val Acc:  90.90%, Time: 0:00:47 *
Iter:    700, Train Loss:   0.29, Train Acc:  92.19%, Val Loss:   0.32, Val Acc:  90.64%, Time: 0:00:49
2021-03-18 19:51:57.214150: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SOURCE is null.
2021-03-18 19:51:57.214226: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SINK is null.


```

#### 4、GPU训练结果：

##### 环境：Tesla V100卡 

##### `ckpt`文件，loss+perf_npu.txt存放路径：

` 10.136.165.4服务器:/turingDataset/results/CarPeting_TF_Textcnn/ckpt_gpu路径`

```shell
精度性能示例：（部分打屏结果如下）
Iter:      0, Train Loss:    2.3, Train Acc:  10.94%, Val Loss:    2.3, Val Acc:   8.92%, Time: 0:00:01 *
Iter:    100, Train Loss:   0.88, Train Acc:  73.44%, Val Loss:    1.2, Val Acc:  68.46%, Time: 0:00:04 *
Iter:    200, Train Loss:   0.38, Train Acc:  92.19%, Val Loss:   0.75, Val Acc:  77.32%, Time: 0:00:07 *
Iter:    300, Train Loss:   0.22, Train Acc:  92.19%, Val Loss:   0.46, Val Acc:  87.08%, Time: 0:00:09 *
Iter:    400, Train Loss:   0.24, Train Acc:  90.62%, Val Loss:    0.4, Val Acc:  88.62%, Time: 0:00:12 *
Iter:    500, Train Loss:   0.16, Train Acc:  96.88%, Val Loss:   0.36, Val Acc:  90.38%, Time: 0:00:15 *
Iter:    600, Train Loss:  0.084, Train Acc:  96.88%, Val Loss:   0.35, Val Acc:  91.36%, Time: 0:00:17 *
Iter:    700, Train Loss:   0.21, Train Acc:  93.75%, Val Loss:   0.26, Val Acc:  92.58%, Time: 0:00:20 *
```

#### 5、GPU/NPU loss收敛趋势：

| step | GPU loss | NPU loss |
| :--- | -------- | :------- |
|      |          |          |
|      |          |          |

#### 6、单step耗时 NPU/GPU：

```shell
NPU/GPU=2
```


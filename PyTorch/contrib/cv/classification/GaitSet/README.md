# GaitSet

gaitset是一个灵活、有效和快速的跨视角步态识别网络，迁移自https://github.com/AbnerHqC/GaitSet



## Detail

1、修改为使用混合精度进行训练

2、修改代码使模型适应NPU环境下的训练

3、使用DDP实现分布式8P训练



## Requirements

1、`torch`、`torchvision`、`opencv-python`和`xarray`，`python3.7.5`

2、具体的本地conda环境：

apex                0.1+ascend

opencv-python       4.5.3.56

torch               1.5.0+ascend.post3

...，完整列表在`conda_env.txt`

3、下载`CASIA-B`数据集：

> 下载地址：http://www.cbsr.ia.ac.cn/GaitDatasetB-silh.zip
>
> 数据集主页：http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp



## Training

### 一、数据预处理

​	1、下载后的数据集内的压缩文件需要全部解压，解压后数据集内部的目录应为（`CASIA-B`数据集）：数据集路径/对象序号/行走状态/角度，例如`CASIA-B/001/nm-01/000/ `。

​	2、使用`pretreatment.py`进行数据处理：其中，包括括号"{}"需要替换为数据集的路径。

```bash
$ python3 pretreatment.py --input_path {downloaded_path} \
                          --output_path {output_path}
```

>  预处理过程中提示`--WARNING--`属于预期现象，请等待处理完成



### 二、训练

1、设置环境变量

```bash
$ source test/npu_set_env.sh
```

2、修改`config.py`文件

根据训练NPU`1P/8P`相应修改`conf_1p/conf_8p`中以下内容：

> **WORK_PATH**：改为保存训练模型文件的路径
>
> **ASCEND_VISIBLE_DEVICES**：改为使用的NPU卡号，用","分隔；例如2卡为"0,1"
>
> **dataset_path**：预处理后的数据集存放路径（建议使用**绝对路径**）
>
> **total_iter**（可选）：训练的总迭代数。如需更改，训练脚本也做相同更改
>
> **restore_iter**（可选）：需要恢复训练时改为保存的模型文件的代数，否则默认设为0
>
> **num_workers**（可选）：根据需要设置worker数

对参数`batch_size`的修改有可能导致性能和精度下降

由于测试时只使用单卡，因此8P也需要修改conf_1p的两个路径



3、NPU1P训练

每`200`次迭代保存1次模型，可手动把`/model/model.py`的第`346`行的`200`调整为其他代数

```bash
$ bash test/train_full_1p.sh
```

RT训练脚本可以外部指定数据集路径${data_path}和迭代数${iters}
RT1脚本1p训练
```bash
$ bash train_ID4118_GaitSet_RT1_performance_1p.sh --data_path=${data_path} --iters=${iters}
```

RT2脚本1p训练
```bash
$ bash train_ID4118_GaitSet_RT2_performance_1p.sh --data_path=${data_path} --iters=${iters}
```


4、NPU8P训练

```bash
$ bash test/train_full_8p.sh
```



### 三、测试

```bash
$ bash test/train_eval_8p.sh
```

注意：需要手动把`train_eval_8p.sh`中`--iter`参数改为训练保存的模型想要加载的代数。

精度、性能结果参考：

|       | 训练代数(Iters) | 精度(RANK-1, %) | 性能(FPS) |
| ----- | --------------- | --------------- | --------- |
| NPU1P | 6w              | 95.488          | 270       |
| NPU8P | 4w              | 95.525          | 1000      |


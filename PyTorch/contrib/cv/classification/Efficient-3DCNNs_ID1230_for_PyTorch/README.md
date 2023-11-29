# Efficient-3DCNNs 使用说明


## 任务
- 开源链接：https://github.com/okankop/Efficient-3DCNNs
- 交付内容：训练+推理
- 交付标准：精度+性能
- 目标配置：MobileNetV2-1.0x
- 目标数据集：UCF101
- 参考精度：Accuracy 81%


--- 

## 数据集

please download from origin repo:
https://github.com/okankop/Efficient-3DCNNs/tree/master/annotation_UCF10

### 处理数据
参考：https://github.com/okankop/Efficient-3DCNNs
- 步骤1：创建目录
```python
    mkdir ./annotation_UCF101/UCF-101/ 
    mkdir./annotation_UCF101/UCF-101-image/
```
- 步骤2：下载视频和训练/测试分割 [here](https://www.crcv.ucf.edu/data/UCF101.php)放置于annotation_UCF101目录下.
- 步骤3：解压下载的UCF-101数据集(需安装解压rar的工具)
```python
    unrar x UCF-101.rar
```
- 步骤4：从avi转换为jpg文件
```python
    python utils/video_jpg_ucf101_hmdb51.py
```
- 步骤5：生成n_frames文件
```python
    python utils/n_frames_ucf101_hmdb51.py
```
- 步骤6：生成类似于ActivityNet的json格式的注释文件。（包括classInd.txt, trainlist0{1,2,3}.txt, testlist0{1,2,3}.txt）
```python
    python utils/ucf101_json.py
```


--- 

## 预训练文件
参考：https://github.com/okankop/Efficient-3DCNNs
下载预训练文件 kinetics_mobilenetv2_1.0x_RGB_16.pth 放置于pretrain目录


--- 

## 目录检查
确认最终的目录结构是否是如下格式：

```
Efficient-3DCNNs_ID1230_for_PyTorch
├── annotation_UCF101 # 数据集
|   |── UCF-101 （原始视频数据。提示：如果使用处理后的数据不需要此文件夹）
|   |   |── ApplyEyeMakeup
|   |   |   |── v_ApplyEyeMakeup_g01_c01.avi
|   |   |   |── ...
|   |── UCF-101-image （预处理后的帧数据）
|   |   |── ApplyEyeMakeup
|   |   |   |── v_ApplyEyeMakeup_g01_c01
|   |   |   |   |── image_00001.jpg
|   |   |   |   |── image_00002.jpg
|   |   |   |   |── ...
|   |   |   |   |── n_frames
|   |── classInd.txt
|   |── testlist01.txt # 测试集1
|   |── testlist02.txt
|   |── testlist03.txt
|   |── trainlist01.txt # 训练集1
|   |── trainlist02.txt
|   |── trainlist03.txt
|   |── ucf101_01.json
|   |── ucf101_02.json
|   |── ucf101_03.json
├── datasets # 数据集加载
|   |── ucf101.py
├── models # 模型定义
|   |── mobilenetv2.py
├── pretrain # 预训练文件
|   |── kinetics_mobilenetv2_1.0x_RGB_16_best.pth
├── results # 结果保存
├── run # 运行依赖程序
|   |── __init__.py
|   |── dataset.py
|   |── eval_ucf101.py
|   |── getmodel.py
|   |── hook.py
|   |── mean.py
|   |── opts.py
|   |── spatial_transforms.py
|   |── target_transforms.py
|   |── temporal_transforms.py
|   |── test.py
|   |── train.py
|   |── utils.py
|   |── validation.py
|   |── video_accuracy.py
├── test # 测试脚本
|   |── env_npu.sh
|   |── gpu 
|   |   |── gpu_train_full_1p.sh      # GPU 1P（包含了精度和性能）
|   |   |── gpu_train_full_1p_withgroup.sh # GPU 1P（包含了精度和性能）注：这里使用了预训练文件
|   |   |── gpu_train_full_8p.sh      # GPU 8P（包含了精度和性能）
|   |   |── gpu_train_full_8p_withgroup.sh # GPU 8P（包含了精度和性能）注：这里使用了预训练文件
|   |── env_npu.sh
|   |── train_full_1p.sh      # NPU 1P完整训练（精度）
|   |── train_full_8p.sh      # NPU 8P完整训练（精度）
|   |── train_performance_1p.sh # NPU 1P训练（性能）
|   |── train_performance_8p.sh # NPU 8P训练（性能）
├── utils # 视频数据转化代码
|   |── __init__.py
|   |── n_frames_ucf101_hmdb51.py
|   |── ucf101_json.py
|   |── video_jpg.py
|   |── video_jpg_ucf101_hmdb51.py
├── main.py # 主函数入口
├── README.md
├── requirements.txt
├── test_acc.py # 测试集评估（自动通过main.py调用）
```


--- 

## 检查NPU环境
- CANN: 5.0.3
- torch：1.5.0+ascend.post3
- torchvison: 0.2.2.post3
- apex: 0.1+ascend.20210930
- topi: 0.4.0
- te: 0.4.0

注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision ,建议Pillow版本是9.1.0 torchvision版本是0.6.0
**注：附录中列出了完整的执行环境详细版本。**


--- 

## 训练执行

<span style="font-family: Source Code Pro; padding: 20px; background-color: #000000; color: #ff0000;">
说明：NPU现在不支持Conv3D算子中group大于1的场景，但GPU是支持的，原始模型也是使用了group参数！！
</span> 

因此要在**GPU 1P/8P**上测试带有group参数的模型，需要将'/models/mobilenetv2.py'文件中：
```python
    第23行： nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, bias=False), 
    修改为：nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
```

```python
    第28行：nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, bias=False),
    修改为：nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
```


### GPU 1P带group参数训练
提示：确保'/models/mobilenetv2.py'中第23行和28代码按照上述方式进行了修改！

执行命令：
```python
    cd test 
    source set_npu_env.sh 
    bash gpu/gpu_train_full_1p_withgroup.sh root_path
```

执行日志保存在'results/gpu_train_full_1p_withgroup.log'中

### GPU 8P带group参数训练
提示：确保'/models/mobilenetv2.py'中第23行和28代码按照上述方式进行了修改！

执行命令：
```python
    cd test 
    source set_npu_env.sh
    bash gpu/gpu_train_full_8p_withgroup.sh root_path
```

执行日志保存在'results/gpu_train_full_8p_withgroup.log'中

### GPU 1P不带group参数训练
执行命令：
```python
    cd test
    source set_npu_env.sh 
    bash gpu/gpu_train_full_1p.sh  root_path
```

执行日志保存在'results/gpu_train_full_1p.log'中


### GPU 8P不带group参数训练
执行命令：
```python
    cd test 
    source set_npu_env.sh 
    bash gpu/gpu_train_full_8p.sh root_path
```

执行日志保存在'results/gpu_train_full_8p.log'中


### NPU 1P精度
执行命令：
```python
    cd test 
    source set_npu_env.sh 
    bash train_full_1p.sh root_path
```

执行日志保存在'results/npu_train_full_1p.log'中

### NPU 1P性能
执行命令：
```python
    cd test 
    source set_npu_env.sh 
    bash train_performance_1p.sh root_path
```

执行日志保存在'results/npu_train_performance_1p.log'中

### NPU 8P精度
执行命令：
```python
    cd test 
    source set_npu_env.sh 
    bash train_full_8p.sh root_path
```

执行日志保存在'results/npu_train_full_8p.log'中

### NPU 8P性能
执行命令：
```python
    cd test 
    source set_npu_env.sh 
    bash train_performance_8p.sh root_path
```

执行日志保存在'results/npu_train_performance_8p.log'中


--- 

## 3D MobileNet-v2 训练结果
注：withgroup表示conv3d中使用group参数，withoutgroup表示conv3d没有使用group参数

|  Type     | Top1 Acc            | FPS     | NPUs/GPUs | BatchSize | Epochs | AMP_Type | AMP_LossScale |
| :------:  | :----:     | :-----: | :------: | :----:    | :----: | :------: |:------: |
|  GPU-1P-withgroup   |   -   | 46.80     |    1     |  80      |  2   |   O2    | 128.0 |
|  GPU-8P-withgroup   |   80.35%   | 355.57     |    8     |  80      |  30   |   O2    | 128.0 |
||||||||
|  GPU-1P-withoutgroup   |   -   | 90.54     |    1     |  80      |  2   |   O2    | 128.0 |
|  GPU-8P-withoutgroup   |   24.02%    |356.05      |    8     |  80      |  30   |   O2    | 128.0 |
||||||||
|  NPU-1P-withoutgroup   |   -     | 111.87    |    1     |  80      |  2   |    O2    | 128.0 |
|  NPU-8P-withoutgroup   |   23.79%    | 484.65     |    8     |  80      |  30   |   O2    | 128.0 |

自测结果说明：
- 项目交付精度要求：Top1 Acc达到 81% 。目前在GPU中Top1 Acc为 80.35% ，已达到目标精度的 99% 以上（即 81% * 99% = 80.19%）。
> NPU由于不支持group参数，无法与目标精度对齐。故与GPU 8P不加group参数进行对齐。目前GPU 8P不加group参数精度为 24.02%， NPU 8P不加group参数精度为 23.79%。NPU精度以达到GPU精度 99% 以上（即 24.02% * 99% = 23.77%）。符合验收标准。
- 项目交付性能要求：NPU 8P性能达到GPU 8P性能 99% 以上。目前GPU 8P不加group参数 FPS=356.05 ， NPU 8P不加group参数 FPS=484.65 。NPU性能以达到GPU性能 99% 以上。符合验收标准。

--- 

## 附录
### 完整环境版本细节
- apex            0.1+ascend.20210930
- attrs           21.2.0
- certifi         2018.10.15
- cffi            1.15.0
- cycler          0.10.0
- Cython          0.29.24
- decorator       5.1.0
- ffmpeg          0.2.0
- ffprobe         0.1.3
- future          0.18.2
- joblib          1.1.0
- kiwisolver      1.3.2
- matplotlib      3.4.3
- mpmath          1.2.1
- numpy           1.21.2
- opencv-python   4.5.3.56
- pandas          1.3.4
- Pillow          8.3.1
- pip             21.2.4
- psutil          5.8.0
- pycocotools     2.0.2
- pycparser       2.20
- pyparsing       2.4.7
- python-dateutil 2.8.2
- pytz            2021.3
- scikit-learn    1.0
- scipy           1.7.1
- setuptools      40.4.3
- six             1.16.0
- sklearn         0.0
- SoundFile       0.10.3.post1
- sympy           1.8
- te              0.4.0
- threadpoolctl   3.0.0
- topi            0.4.0
- torch           1.5.0+ascend.post3
- torchvision     0.2.2.post3
- tqdm            4.62.3
- wheel           0.32.1

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
# 3DMPPE_ROOTNET

在数据集MuCo、MPII和MuPoTS上实现对3DMPPE_ROOTNET的训练。
- 实现参考：
```
url=https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE
branch=master 
commit_id=a199d50be5b0a9ba348679ad4d010130535a631d
```

## 3DMPPE_ROOTNET 细节

3DMPPE_ROOTNET是一个经典姿态估计网络，它可以对一张图片上多人的三维姿态进行估计，这也其名称的而来。该网络基于一种相机距离感知的自上而下的方法，从单个RGB图像进行三维多人姿态估计，它包括人物检测模块、人物绝对三维定位模块和基于三维的单人姿态估计模块。该模型取得了与最先进的三维单人姿态估计模型相当的结果。

## 环境准备

- 安装 PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
  注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision ,建议Pillow版本是9.1.0 torchvision版本是0.6.0
- 训练数据集是MuCo和MPII，评估数据集是MuPoTS，数据集较大，请在下载和解压时确保硬盘空间充足。
- 请在`data`文件夹中遵循以下的目录结构，如果连接到了已经准备好的数据文件夹，就不需要再构建下面的目录。
```
${3DMPPE_ROOTNET}
|-- data
|   |-- MPII
|   |   |-- images
|   |   |   |-- ...   ## 图片文件
|   |   |-- annotations
|   |   |   |-- test.json
|   |   |   |-- train.json
|   |-- MuCo
|   |   |-- data
|   |   |   |-- augmented_set
|   |   |   |   |-- ...   ## 图片文件
|   |   |   |-- unaugmented_set
|   |   |   |   |-- ...   ## 图片文件
|   |   |   |-- MuCo-3DHP.json
|   |-- MuPoTS
|   |   |-- data
|   |   |   |-- MultiPersonTestSet
|   |   |   |   |-- ...   ## 图片文件
|   |   |   |-- MuPoTS-3D.json
```
- 请在`output`文件夹中遵循以下目录结构：
```
${3DMPPE_ROOTNET}
|-- output
|   |-- model_dump  //用于保存.pth文件
|   |-- result
|   |-- log
|   |   |-- train_logs.txt   ## 训练的日志保存在这里
|   |   |-- test_logs.txt    ## 评估的日志保存在这里
|   |-- vis
|   |-- prof
```

## 训练模型

- 注意，`test`目录下的`output`文件夹也会保存代码运行的日志。
- 运行 `train_1p.py` 或 `train_8p.py` 进行模型训练：


```
# 1p train perf
bash test/train_performance_1p.sh --data_path=xxx

# 8p train perf
bash test/train_performance_8p.sh --data_path=xxx

# 1p train full
bash test/train_full_1p.sh --data_path=xxx

# 8p train full
bash test/train_full_8p.sh --data_path=xxx

```

## 训练结果

| AP_25(百分比)    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| 33(平均) 35.78(最高）        | 190      | 1        | 20      | O1       |
| -        | 230      | 2        | 3      | O1       |
| -        | 425      | 4        | 3      | O1       |
| 37(平均) 41.05(最高）        | 855      | 8        | 20      | O1       |

# 其它说明 # 

- 运行 `demo.py`：
进入 `demo` 文件夹。运行demo的输入文件已经提供（`input.jpg`），运行结束后会在该目录下得到输出的图片。将 `snapshot_XX.pth` 放置在 `./output/model_dump/` 目录下。
修改 `run_demo.sh` 中 `test_epoch` 的参数为 `XX` ，与刚才的 `.pth` 文件的数字对应。最后，运行指令：
```
bash demo/run_demo.sh
```
也可以直接运行：
```
python demo.py --test_epoch XX
```
- 单独进行模型评估：
将 `snapshot_XX.pth` 放置在 `./output/model_dump/` 目录下。cd到 `main` 文件夹，运行指令：
```
python test.py --test_epoch XX
```
记录也会保存在 `./output/log/test_logs.txt` 中。
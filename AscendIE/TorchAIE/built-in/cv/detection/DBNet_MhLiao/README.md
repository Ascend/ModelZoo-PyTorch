# DBNet_MhLiao

- [概述](#ABSTRACT)
- [环境准备](#ENV_PREPARE)
- [准备数据集](#DATASET_PREPARE)
- [快速上手](#QUICK_START)
- [模型推理性能&精度](#INFER_PERFORM)
  
***

## 概述 <a name="ABSTRACT"></a>
- 参考实现
  ```
  url=https://github.com/MhLiao/DB 
  commit_id=4ac194d0357fd102ac871e37986cb8027ecf094e  
  ```
### 输入输出数据
- 输入数据

  | 输入数据 | 数据类型 | 大小                 | 数据排布 |
  | ------- | -------- | ------------------- | ------- |
  | input   |          | bs x 3 x 736 x 1280 | NCHW    |

- 输出数据
  
  | 输出数据 | 数据类型 | 大小                 | 数据排布 |
  | ------- | -------- | ------------------- | ------- |
  | output  | Float32  | bs x 1 x 736 x 1280 | ND      |

## 环境准备 <a name="ENV_PREPARE"></a>
| 配套                   | 版本            | 
|-----------------------|-----------------| 
| CANN                  | 6.3.RC2.alph002 |
| Python                | 3.9        |                                                           
| Torch+cpu             | 2.0.1           |
| torchVison            | 0.15.2          | 
| Ascend-cann-torch-aie | --              |
| Ascend-cann-aie       | --              |
| 芯片类型               | Ascend310P3     |
### 配置OpenMMLab运行环境
#### 环境配置步骤
- 获取源码
  ```
  git clone https://github.com/MhLiao/DB 
  cd DB
  git reset 4ac194d0357fd102ac871e37986cb8027ecf094e --hard
  pip3 install -r requirements.txt
  ```
- 获取运行脚本和环境  
  下载本仓的 db.diff 和 dbnet_compile_run.py 到上方获取的 DB 路径里
  ```
  dos2unix db.diff
  dos2unix ./backbones/resnet.py
  patch -p1 < db.diff
  ```
- 获取权重文件  
  权重文件同样需要在 DB 路径下存放
  ```
  wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/DBnet/PTH/ic15_resnet50 -O ic15_resnet50
  ```

### 配置昇腾运行环境
下载对应版本的昇腾产品
#### 安装CANN包

```
chmod +x Ascend-cann-toolkit_6.3.RC2.alpha002_linux-aarch64.run 
./Ascend-cann-toolkit_6.3.RC2.alpha002_linux-aarch64.run --install
```

#### 安装推理引擎

```
chmod +x Ascend-cann-aie_6.3.T200_linux-aarch64.run
Ascend-cann-aie_6.3.T200_linux-aarch64.run --install
cd Ascend-cann-aie
source set_env.sh
```

#### 安装torch—aie

```
tar -zxvf Ascend-cann-torch-aie-6.3.T200-linux_aarch64.tar.gz
pip3 install torch-aie-6.3.T200-linux_aarch64.whl
```

## 准备数据集 <a name="DATASET_PREPARE"></a>
- 数据集下载地址
  ```
    url=https://rrc.cvc.uab.es/?ch=4&com=downloads
  ```
  这里我们使用的 ICDAR2015 的500张图片的测试数据集和标注。从链接中下载 Test Set Images 数据集和 Test Set Ground Truth 并根据下方排布对数据集进行处理。
- 数据集格式
  ```
    ├── datasets
    |   ├── icdar2015_images
    |   |   ├── test_images
    |   |   |    ├── img_1.jpg
    |   │   |    
    |   │   |    ├── ......
    |   |   ├── test_gts
    |   |   |    ├── gt_img_1.txt
    |   │   |    
    |   │   |    ├── ......
    |   |   ├── test_list.txt ("img_1.jpg\n...")
    |   |   ├── train_images
    |   |   |    ├── img_1.jpg
    |   │   |    ├── ......
    |   |   ├── train_gts
    |   |   |    ├── img_1.txt
    |   │   |    ├── ......
    |   |   ├── train_list.txt ("img_1\n...")
  ```
  这里因为代码仓设计会需要存在train路径，可以在其中放一两个文件即可。此处需要注意数据集中各个文件的命名区别。

## 快速上手 <a name="QUICK_START"></a>
一下所有命令均在上方获取的 DB 路径下运行。

- trace并compile 并运行 AIE模型
  ```
    python3 dbnet_compile_run.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume ./ic15_resnet50 --trace_compile
  ```

- load 并运行 AIE模型
  ```
    python3 dbnet_compile_run.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume ./ic15_resnet50
  ```

## 模型推理性能&精度 <a name="INFER_PERFORM"></a>
| 芯片型号 | Batch Size | 数据集    | 性能 | 精度 |
|---------|------------|-----------|------|------|
| 310P3   | 1          | ICDAR2015 | 10.94qps | 0.887 |
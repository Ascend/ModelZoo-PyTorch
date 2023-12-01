# CenterFace模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

  - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备工作](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

- 论文：[CenterFace: Joint Face Detection and Alignment Using Face as Point](https://arxiv.org/abs/1911.03599)


- 参考实现：

  ```
  url=https://gitee.com/Levi990223/center-face
  branch=master
  commit_id=063db90e844fa0271abc14067b871f5afcbe6c60
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FLOAT16 | batchsize x 3 x 800 x 800 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小       | 数据排布格式 |
  | -------- | -------- | ---------- | ------------ |
  | output  | FLOAT16  | batchsize x 15 x 200 x 200  | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套        |  版本    | 环境准备指导                                                 |
  | ----------- | -------  | ------------------------------------------------------------ |
  | 固件与驱动   | 23.0.rc3 | -                                                          |
  | CANN        |  7.0.RC1 | -                                                            |
  | Python      |  3.9.2   | -                                                            |
  | AIE         |  7.0.RC1 | -                                                            |
  

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/Levi990223/center-face
   ```

2. 整理代码结构

   ```
   mv centerface_torchaie_run.py centerface_postprocess.py ./center-face/src/lib
   mv patch.diff ./center-face
   cd ./center-face
   patch -p1 < patch.diff
   ```

## 准备工作<a name="section183221994411"></a>

1. 获取原始数据集。
> 提示：请遵循数据集提供方要求使用。

   获取WIDER_FACE的VAL数据集，解压到center-face目录下。目录结构如下：

   ```
   center-face
   ├── WIDER_val
   |   ├── images
   │   │   ├── 0--Parade
   │   |   |   ├── 0_Parade_Parade_0_102.jpg
   │   |   |   ├── 0_Parade_Parade_0_12.jpg
   │   │   ├── 1--Handshaking
   ...
   ```

2. 获取权重文件[model_best.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/CenterFace/PTH/model_best.pth)，放在center-face/src/lib目录下。

3. 在center-face/src/lib/external路径下，执行以下命令编译nms。
      ```
      python3 setup.py build_ext --inplace
      ```

4. 在center-face/evaluate目录下，执行以下命令编译bbox。
      ```
      python3 setup.py build_ext --inplace
      ```

## 模型推理<a name="section741711594517"></a>

1. 在center-face/src/lib目录下，执行centerface_torchaie_run.py文件进行模型推理测试
      ```
      python3 centerface_torchaie_run.py --gpus 0 --batch_size 4
      ```
    
    参数说明：

    gpus：推理使用的NPU device ID, 仅支持单个设备推理；

    batch_size：推理使用的batch_size。

2. 在center-face/src/lib目录下，执行centerface_postprocess.py文件进行后处理

      ```
      python3 centerface_postprocess.py
      ```

3. 在center-face/evaluate目录下，执行evaluation.py文件进行精度验证。

      ```
      python3 evaluation.py
      ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

| Batch Size | 数据集    | 精度                          | 310P3性能(数据集) | 310P3性能(随机数) | 
| ---------- | --------- | ----------------------------- | ------- | ------- |
| 1          | widerface | easy：92.27%<br/>Medium：91.02%<br/>hard：74.54% | 425 | 450 |
| 4          | widerface | - | 375 | 380 |
| 8          | widerface | - | 350 | 353 |
| 16         | widerface | - | 346 | 348 |
| 32         | widerface | - | 345 | 350 |

# FAQ
1、报错No module named 'datasets.dataset_factory'。

解决方法：
      ```
      cd center-face/src/lib/datasets
      touch __init__.py
      ```

2、报错module 'numpy' has no attribute 'float'。

解决方法一：numpy版本降级:
```
pip3 install numpy==1.23.1
```

解决方法二：修改源码evaluate/bbox.pyx、evaluate/box_overlaps.pyx第12行：

```
DTYPE = np.float64
```

修改完后重新编译bbox:

```
python3 setup.py build_ext --inplace
```

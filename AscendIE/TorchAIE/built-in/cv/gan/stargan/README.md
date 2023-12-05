# StarGAN模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

StarGAN是 Yunjey Choi 等人于 17年11月 提出的一个模型。该模型可以实现 图像的多域间的迁移（作者在论文中具体应用于人脸属性的转换）。在 starGAN 之前，也有很多 GAN模型 可以用于 image-to-image，比如 pix2pix（训练需要成对的图像输入），UNIT（本质上是coGAN），cycleGAN（单域迁移）和 DiscoGAN。而 starGAN 使用 一个模型 实现 多个域 的迁移，这在其他模型中是没有的，这提高了图像域迁移的可拓展性和鲁棒性。



- 参考实现：

  ```
  url=https://github.com/yunjey/stargan
  commit_id=94dd002e93a2863d9b987a937b85925b80f7a19f
  code_path=contrib/cv/gan/stargan
  model_name=stargan
  ```
  

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | img    | RGB_FP32 | batchsize x  3 x 128 x 128	 | NCHW         |
  | Attr    | INT | 	batchsize x 5	 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型        | 大小 | 数据排布格式 |
  | -------- | ------------ | -------- | ------------ |
  | output1  | RGB_FP32 | batchsize x 3 x 128 x 128  | NCHW           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套  | 版本  | 环境准备指导  |
  |---------| ------- | ------------------------------------------------------------ |
  | 固件与驱动 | 23.0.rc1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN | 7.0.RC1.alpha003 | - |
  | Python | 3.9.11 | - |
  | PyTorch | 2.0.1 | - |
  | Torch_AIE | 6.3.rc2 | - |

- 安装依赖

   ```
   pip install -r requirements.txt
   ```


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip） 数据集默认路径为 ./data/celeba.zip ，使用脚本 unzip_dataset.sh 解压数据集。

    ```
    bash unzip_dataset.sh
    ```

   数据目录结构请参考：
   ```
   ├──celeba
    ├──images
    ├──list_attr_celeba.txt
   ```

## 模型推理<a name="section741711594517"></a>

  1. 模型转换

  ```
  python3 export.py --input_file=./stargan/200000-G.pth
  ```
    将`input_file`改为自己的模型文件目录，执行完毕后会导出.ts文件


  2. 精度验证。

    调用脚本`StarGAN_pre_processing.py`，可以获得推理结果

  ```
  python3 StarGAN_pre_processing.py --result_dir './result_baseline' --attr_path './dataset/celeba/list_attr_celeba.txt' --celeba_image_dir './dataset/celeba/images'  --batch_size 16 --ts_model_path "./stargan.ts"
  ```

    - 参数说明：

      - ts_model_path：ts模型文件路径
  
  3. 性能测试。

```
python3 perf.py --mode ts --batch-size 
```
    - 参数说明：
    - --batch ：推理时输入`batch`, 可通过修改第1维测试不同batch的推理性能。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| Ascend310P3 | 1 | celeba | NA | 756 FPS |
| Ascend310P3 | 4 | celeba | NA | 1108 FPS |
| Ascend310P3 | 8 | celeba | NA | 1169 FPS |
| Ascend310P3 | 16 | celeba | NA | 1111 FPS |

精度参考如下图片：

![](imgs/1.png)

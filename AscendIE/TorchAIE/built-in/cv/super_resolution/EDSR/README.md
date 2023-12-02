# EDSR模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

论文通过提出EDSR模型移除卷积网络中不重要的模块并且扩大模型的规模，使网络的性能得到提升。

- 参考实现：

  ```
  url=https://github.com/sanghyun-son/EDSR-PyTorch.git
  branch=master
  commit_id=9d3bb0ec620ea2ac1b5e5e7a32b0133fbba66fd2
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- | -------- | -------------------------   | ------------ |
  | input    | FLOAT32  | batchsize x 3 x 1020 x 1020 | NCHW         |

- 输出数据

  | 输出数据 | 大小                         | 数据类型 | 数据排布格式 |
  | -------- | --------                     | -------- | ------------ |
  | output   | batch_size x 3 x 2040 x 2040 | FLOAT32  | NCHW        |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>
  **表 1**  版本配套表

| 配套                    | 版本               | 
|-----------------------|------------------| 
| CANN                  | 7.0.RC1.alpha003 | -                                                       |
| Python                | 3.9.0            |                                                           
| PyTorch               | 2.0.1            |
| torchVison            | 0.15.2           |-
| Ascend-cann-torch-aie | -                
| Ascend-cann-aie       | -                
| 芯片类型                  | Ascend310P3      | -                                                         |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git        # 克隆仓库的代码
   git checkout master         # 切换到对应分支
   cd AscendIE/TorchAIE/cv/super_resolution/EDSR              # 切换到模型的代码仓目录
   ```
2. 模型开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```
   git clone https://github.com/sanghyun-son/EDSR-PyTorch.git
   cd EDSR-PyTorch && git checkout 9d3bb0ec
   patch -p1 < ../edsr.diff
   cd ..
   ```
3. 获取权重文件
   下载地址[pth权重文件](https://github.com/aamir-mustafa/super-resolution-adversarial-defense/blob/master/models/edsr_baseline_x2-1bc95232.pt)

## 安装依赖。

下载Ascend-cann-torch-aie和Ascend-cann-aie得到run包和压缩包
- 安装Ascend-cann-aie
    ```
    chmod +x Ascend-cann-aie_6.3.T200_linux-aarch64.run
    ./Ascend-cann-aie_6.3.T200_linux-aarch64.run --install
    cd Ascend-cann-aie
    source set_env.sh
    ```
- 安装Ascend-cann-torch-aie
    ```
    tar -zxvf Ascend-cann-torch-aie-6.3.T200-linux_aarch64.tar.gz
    pip3 install torch-aie-6.3.T200-linux_aarch64.whl
    ```
- 安装其他依赖
   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。

   本模型采用[DIV2K官网](https://data.vision.ee.ethz.ch/cvl/DIV2K/)的100张验证集进行测试
   其中，低分辨率图像(LR)采用bicubic x2处理(Validation Data Track 1 bicubic downscaling x2 (LR images))，高分辨率图像(HR)采用原图验证集(Validation Data (HR images))。

   数据目录结构请参考：

   ```
   ├── DIV2K              
         ├──HR  
              │──图片1
              │──图片2
              │   ...
         ├──LR
              │──图片1
              │──图片2
              │   ...       
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行预处理脚本，生成数据集预处理后的bin文件:

   ```
   python3 edsr_preprocess.py -s DIV2K/LR -d ./prep_data
   ```

   - 参数说明：

     -s: LR(低分辨率)数据集文件位置。

     -d：输出文件位置。

## 模型推理<a name="section741711594517"></a>
   ```
   python3 cal_sample.py --HR ./DIV2K_valid_HR --prep_data ./prep_data/bin/ 
   ```
   - 参数说明： 
   - -HR: 原始高精度图片的文件夹路径。
   - -prep_data: 使用LR(低精度图片)预处理之后bin文件夹路径。 
   - -pth: 预训练权重，默认值：edsr_baseline_x2-1bc95232.pt。
   - pad_info: 执行edsr_preprocess.py脚本生成的中间文件，默认值 "./pad_info.json"。
   - batch_size: 批次大小，默认值 1。
   - device_id: 设备ID，默认值 0。


## 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

精度参考下列数据:

| device |   ACC |
|--------|-------|
| 310P   | 34.6% |

性能参考下列数据(memory限制只支持到bs8)。

| 模型       | 310P3性能 |
|----------|------|
| EDSR bs1 | 7.1818|
| EDSR bs4 | 8.2233|
| EDSR bs8 | 8.3908|

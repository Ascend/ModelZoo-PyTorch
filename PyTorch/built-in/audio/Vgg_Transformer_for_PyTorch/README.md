# Vgg_Transformer for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

Vgg_Transformer模型基于fairseq框架，可在librispeech数据集上完成自动语音识别任务。模型使用卷积学到的输入表征来代替原来的位置编码，相对于原来的绝对位置表征，这种相对位置的编码效果更有利于后面的transformer去发现长距离的依赖关系，以获得更高的语音识别精度。


- 参考实现：

  ```
  url=https://github.com/facebookresearch/fairseq
  tag=v0.10.2
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/audio
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      |      三方库依赖版本      |
  |:-----------------:| :--------------------: |
  | PyTorch 1.11 | torchaudio==0.11.0 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  1. 安装基础依赖

      安装好PyTorch后，在模型源码包根目录下执行命令，安装模型需要的依赖。
      ```
      pip install --editable ./
      ```

  2. 编译安装torchaudio

      ```
      git clone -b release/0.11 https://github.com/pytorch/audio.git
      cd audio
      python3 setup.py install
      ```

  3. 安装sentencepiece

      安装sentencepiece python包，并从源码编译安装以获取spm_train/spm_encode指令。
      ```
      pip install sentencepiece
      git clone https://github.com/google/sentencepiece.git
      cd sentencepiece
      mkdir build && cd build
      cmake ..
      make -j $(nproc)
      sudo make install 
      sudo ldconfig -v
      ```


## 准备数据集

1. 获取数据集。

   进入该模型文件夹，使用官方脚本获取数据集并进行预处理，完整执行完后，总数据集大小约130G。
   `$DIR_TO_SAVE_RAW_DATA`指定原始数据路径，`$DIR_FOR_PREPROCESSED_DATA`指定预处理数据路径。
   执行训练脚本时，数据集路径需指定为`$DIR_FOR_PREPROCESSED_DATA`。

   ```
   cd /${模型文件夹名称}
   ./examples/speech_recognition/datasets/prepare-librispeech.sh $DIR_TO_SAVE_RAW_DATA $DIR_FOR_PREPROCESSED_DATA
   ```

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   - 单机8卡全量训练
     ```
     bash test/train_full_8p.sh --data_path=$DIR_FOR_PREPROCESSED_DATA  # 8卡全量训练
     ```

     --data_path参数填写预处理数据路径`$DIR_FOR_PREPROCESSED_DATA`。

   ```
   公共参数：
   --train_epochs                      //训练总epoch数
   --token_size                        //单batch最大帧数
   ```
   训练完成后，权重文件保存在`test/output/saved_results`，推理结果保存在`test/output/infer_results`。
   并输出模型训练精度和性能信息。
   

# 训练结果展示

**表 2**  训练结果展示表

| Name   | WER  | WPS  | Device | Epochs | DataType | CPU |
|--------|:----:|:----:|:------:|:------:|:--------:|:---:|
| 8p-竞品A | 8.56 | 7744 |   -    |   80   |   fp32   | x86 |
| 8p-NPU | 8.58 | 3059 |  -  |   80   |   fp32   | ARM |

# 公网地址说明

  代码涉及公网地址参考 public_address_statement.md
# 版本说明

## 变更

2023.6.25：首次发布。

## FAQ

Q：运行数据集脚本`prepare-librispeech.sh`时出现`spm_train`报错：

```
spm_train: error while loading shared libraries: libsentencepiece.so.0: cannot open shared object file: No such file or directory。
```

A：可以通过将编译安装sentencepiece时，.so文件实际保存路径加入至`/etc/ld.so.conf`解决：

```
echo "/usr/local/lib64" >> /etc/ld.so.conf
sudo ldconfig -v
```
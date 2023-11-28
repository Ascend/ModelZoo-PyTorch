# FastPitch模型-推理指导

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

Fastpitch模型由双向 Transformer 主干（也称为 Transformer 编码器）、音调预测器和持续时间预测器组成。 在通过第一组 N 个 Transformer 块、编码后，信号用基音信息增强并离散上采样。 然后它通过另一组 N个 Transformer 块，目的是平滑上采样信号，并构建梅尔谱图。

- 参考实现：

  ```shell
    url=https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小              | 数据排布格式 |
  | -------- |-----------------| ------------------------- | ------------ |
  | input   | RGB_FP32 | batchsize x 200 | NCHW         |

- 输出数据

  | 输入数据 | 数据类型    | 大小                   | 数据排布格式 |
  |---------|----------------------|--------| ------------ |
  | output1   | FLOAT32 | batchsize x 80 x 900 | ND     |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下依赖

  **表 1**  版本配套表


  | 配套                                                            | 版本    | 环境准备指导                                                                                          |
  | --------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
  | 固件与驱动                                                      | 23.0.rc1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 7.0.RC1.alpha003 | -                                                                                                     |
  | Python                                                          | 3.9.11  | -                                                                                                     |
  | PyTorch                                                         | 2.0.1   | -                                                                                                     |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \   


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

      ```
        git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
        cd FastPitch
        git clone https://github.com/NVIDIA/DeepLearningExamples
        cd ./DeepLearningExamples
        git checkout master
        git reset --hard 6610c05c330b887744993fca30532cbb9561cbde
        mv ../p1.patch ./
        patch -p1 < p1.patch
        cd ..
        git clone https://github.com/NVIDIA/dllogger.git
        cd ./dllogger
        git checkout 26a0f8f1958de2c0c460925ff6102a4d2486d6cc
        cd ..
        export PYTHONPATH=dllogger:${PYTHONPATH}
    ```

2. 安装依赖。

      ```
       pip install -r requirements.txt
      ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）。
    ```
      wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
      tar -xvjf LJSpeech-1.1.tar.bz2
    ```
2. 数据预处理，计算Pitch
    ```
   python3 DeepLearningExamples/PyTorch/SpeechSynthesis/FastPitch/prepare_dataset.py --wav-text-filelists DeepLearningExamples/PyTorch/SpeechSynthesis/FastPitch/filelists/ljs_audio_text_val.txt --n-workers 16 --batch-size 1 --dataset-path ./LJSpeech-1.1 --extract-mels --f0-method pyin
    ```
   参数说明：
   * --wav-text-filelists：包含数据集文件路径的txt文件
   * --n-workers：使用的CPU核心数
   * --batch-size：批次数
   * --dataset-path：数据集路径
   * --extract-mels：默认参数
   * --f0-method：默认参数，代码中只包含了pyin选项，不可替换

2. 保存模型输入、输出数据

    为了后面推理结束后将om模型推理精度与原pt模型精度进行对比，脚本运行结束会在test文件夹下创建mel_tgt_pth用于存放pth模型输入数据，mel_out_pth用于存放pth输出数据，input_bin用于存放二进制数据集，input_bin_info.info用于存放二进制数据集的相对路径信息
   ```
    python3 data_process.py -i phrases/tui_val100.tsv --dataset-path=./LJSpeech-1.1 --fastpitch ./nvidia_fastpitch_210824.pt --waveglow ./nvidia_waveglow256pyt_fp16.pt
   ```
    参数说明：
    * -i：保存数据集文件的路径的tsv文件
    * -o：输出二进制数据集路径
    * --dataset-path：数据集路径
    * --fastpitch：fastpitch权重文件路径
    * --waveglow：waveglow权重文件路径

## 模型推理<a name="section741711594517"></a>
1. 获取权重文件。
    ```
     wget https://gitee.com/link?target=https%3A%2F%2Fascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com%2Fmodel%2F1_PyTorch_PTH%2FUnet%252B%252B%2FPTH%2Fnested_unet.pth
    ```


2. 生成trace模型(onnx, ts)
   
    首先使用本代码提供的pth2onnx.py替换原代码的同名脚本
    ```
    python3 pth2onnx.py -i phrases/tui_val100.tsv --fastpitch nvidia_fastpitch_210824.pt --waveglow nvidia_waveglow256pyt_fp16.pt --energy-conditioning --batch-size 1
    ```

3. 保存编译优化模型（非必要，可不执行。若不执行，后续执行推理脚本时需要包含编译优化过程，入参加上--need_compile）

    ```
     python export_torch_aie_ts.py
    ```
   命令参数说明（参数见onnx2om.sh）：
    ```
     --torch_script_path：编译前的ts模型路径
     --soc_version：处理器型号
     --batch_size：模型batch size
     --save_path：编译后的模型存储路径
    ```


4. 执行推理脚本

    （1）推理脚本，包含性能测试。
     ```
      python3 pt_val.py --model nested_unet_torch_aie_bs4.pt --batch_size=4
     ```
   命令参数说明：
    ```
   -i 输入text的完整路径，默认phrases/tui_val100.tsv 
   --dataset_path 数据集路径，默认./LJSpeech-1.1 
   --fastpitch checkpoint的完整路径，默认./nvidia_fastpitch_210824.pt 
   --model 模型路径
     --soc_version：处理器型号
     --need_compile：是否需要进行模型编译（若参数model为export_torch_aie_ts.py输出的模型，则不用选该项）
     --batch_size：模型batch size。注意，若该参数不为1，则不会存储推理结果，仅输出性能
     --device_id：硬件编号
     --multi：将数据扩展多少倍进行推理。注意，若该参数不为1，则不会存储推理结果，仅输出性能
    ```
5. 精度验证

    调用脚本分别对比input中创建的mel_tgt_pth输入数据和ais_bench推理结果./result/{}，以及pthm模型mel_out_pth输出数据，可以分别获得om和pth模型的Accuracy数据。
     ```
    python3 infer_test.py ./result/
    ```
   命令参数说明：
    ```
    ./result/：推理结果保存路径
    ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>



芯片型号 Ascend310P3。
dataloader生成未drop_last，已补满尾部batch

模型精度 bs1 = 11.2573

**表 2** 模型推理性能

| batch_size              | 性能（fps） | 数据集扩大倍数 |
|-------------------------|---------|---------|
| 1                       | 25.4367 | 8       |
| 4                       | 58.792  | 32      |
| 8                       | 70.5458 | 64      |
| 16                      | 68.4412 | 128     |
| 32                      | 71.5593 | 256     |
| 64                      | 70.0225 | 512     |

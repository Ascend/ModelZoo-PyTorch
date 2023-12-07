# DeepSpeech2模型-推理指导

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

Deepspeech是百度推出的语音识别框架，系统采用了端对端的深度学习技术，也就是说，系统不需要人工设计组件对噪声、混响或扬声器波动进行建模，而是直接从语料中进行学习，并达到了较好的识别效果。

- 参考实现：

  ```shell
    url=https://github.com/SeanNaren/deepspeech.pytorch
    branch=master
    commit_id=075a69ae66aa284c5c5a954c6c15efe6d56898dd
  ```

## 输入输出数据<a name="section540883920406"></a>

 参考
 
https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/audio/Deepspeech2


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下依赖

  **表 1**  版本配套表


  | 配套                                                            | 版本               | 环境准备指导                                                                                          |
  |------------------| ------- | ----------------------------------------------------------------------------------------------------- |
  | CANN                                                            | 7.0.RC1.alpha003 | -                                                                                                     |
  | Python                                                          | 3.9.11           | -                                                                                                     |
  | PyTorch                                                         | 2.0.0            | -                                                                                                     |
  | 说明：芯片类型：Ascend310P3 | \                | \   


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码并安装。

      ```
       获取推理部署代码
        git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
        cd ModelZoo-PyTorch/ACL_PyTorch/contrib/audio/Deepspeech2
       获取源码并安装
        git clone https://github.com/SeanNaren/deepspeech.pytorch.git -b V3.0
        cd deepspeech.pytorch
        pip3 install -e .
    ```

2. 安装依赖。

      ```
       使用本目录下的requirements.txt替换原始requirements.txt
       pip install -r requirements.txt
      ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）
    
    该模型使用AN4数据集验证模型精度，参考[此代码](https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/an4.py)下载并整理数据集，最终的数据目录结构如下：
   ```
    |——an4_test_manifest.json
    |——labels.json  
    |——an4_dataset
        |——val
        |——train
        |——test
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。
    
    数据预处理将原始数据集转换为模型输入的数据。 执行deepspeech_preprocess.py脚本，完成预处理。

   ```
    python3 deepspeech2_preprocess.py --data_file ./deepspeech.pytorch/data/an4_test_manifest.json --save_path ./deepspeech.pytorch/data/an4_dataset/test --label_file ./deepspeech.pytorch/labels.json
   ```
   命令参数说明：
    ```
     --data_file：json文件路径。
     --save_path：输出的二进制文件（.bin）所在路径。
     --label_file：标签文件路径。
    ```
   说明： 在预处理前，修改an4_test_manifest.json中root_path参数，改为当前an4_dataset中test数据集的路径，方便进行数据预处理。 如果linux系统缺少sox，需要安装sox。

   运行成功后在./deepspeech.pytorch/data/an4_dataset/test目录下生成供模型推理的bin文件。
## 模型推理<a name="section741711594517"></a>
1. 获取权重文件
   [an4_pretrained_v3.ckpt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/DeepSpeech2/PTH/an4_pretrained_v3.ckpt)。

2. 生成trace模型(ts)
   使用“an4_pretrained_v3.ckpt”导出onnx文件。 运行“ckpt2onnx.py”脚本。
    ```
     将pytorch2ts.py放在deepspeech2_ckpt2onnx.py同一目录下
     python3 pytorch2ts.py --ckpt_path ./an4_pretrained_v3.ckpt --out_file deepspeech_torchscript.pt
    ```

3. 保存编译优化模型（非必要，可不执行。若不执行，则后续执行的推理脚本需要包含编译优化过程，添加入参--need_compile）

    ```
     python export_torch_aie_ts.py --batch_size=1
    ```
   命令参数说明：
    ```
     --torch_script_path：编译前的ts模型路径
     --soc_version：处理器型号
     --batch_size：模型batch size
     --save_path：编译后的模型存储路径
    ```


4. 执行推理脚本（包括性能验证）

    将pt_val.py与model_pt.py放在Ecapa_Tdnn下
     ```
      python3 pt_val.py --model="deepspeech_torchscript_torch_aie_bs1.pt" --batch_size=1 --multi=4
     ```
   命令参数说明（参数见onnx2om.sh）：
    ```
     --soc_version：处理器型号
     --model：输入模型路径
     --need_compile：是否需要进行模型编译（若使用export_torch_aie_ts.py输出的模型，则不用选该项）
     --batch_size：模型batch size
     --device_id：硬件编号
     --multi：将数据扩展多少倍进行推理。注意，若该参数不为1，则不会存储推理结果，仅输出性能
     --data_file：json文件路径。
     --label_file：标签文件路径。
     --result_path：结果存储路径，默认result/dumpout
    ```
# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

 精度验证(执行旧脚本需要先将numpy回退到先前版本)

   ```
   pip3 install numpy==1.19.3
   python3 deepspeech2_postprocess.py --out_path ./result/dumpout --info_path ./deepspeech.pytorch/data/an4_dataset/test --label_file ./deepspeech.pytorch/labels.json
   ```
命令参数说明：
   ```
    --out_path：生成推理结果所在路径。
    --info_path：输出的二进制文件所在路径。
    --label_file：标签数据路径。
   ```

**表 2** ecapa_tdnn模型精度

| batchsize                                      | aie性能（fps） | aie精度                               |
|------------------------------------------------|------------|-------------------------------------|
| bs1                                            | 1.2357     | Average WER 9.573 Average CER 5.515 |
| bs4                                            | 4.9684     | /                                   |
| bs8                                            | 9.877      | /                                   |
| bs16                                           | 18.7566    | /                                   |
| bs32                                           | 6.6713     | /                                   |
| bs64                                           | 6.9273     | /                                   |

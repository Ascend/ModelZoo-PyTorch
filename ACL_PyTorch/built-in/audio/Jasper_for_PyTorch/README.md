# Jasper模型推理指导：

## 文件说明
  1. `acl_net.py`：PyACL推理工具代码
  2. `om_infer_acl.py`：Jasper推理代码，基于om推理
  3. `pth2onnx.py`：根据pth文件得到onnx模型

## 环境准备
  - 文件下载
    - 源码下载

      下载[Jasper源码](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper)或解压文件，切换到 `DeepLearningExamples/PyTorch/SpeechRecognition/Jasper` 目录下。

    - 权重下载

      从 [NGC model repository](https://ngc.nvidia.com/catalog/models/nvidia:jasperpyt_fp16) 下载Jasper的权重文件

    - 数据集下载

      下载 [LibriSpeech-test-other.tar.gz](https://www.openslr.org/resources/12/test-other.tar.gz)数据集并根据 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper#quick-start-guide) 的 `Download and preprocess the dataset` 部分进行数据预处理，将原始 `.flac` 文件转换成 `.wav` 文件

  - 文件拷贝

    拷贝 `acl_net.py`, `atc.sh`, `diff.patch`, `om_infer.sh`, `pth2onnx.py`文件到 `DeepLearningExamples/PyTorch/SpeechRecognition/Jasper` 目录下，创建 `checkpoints` 目录并将 `jasper_fp16.pt` 文件拷贝到该目录下。

## 推理端到端步骤

1. pth导出onnx
    ```python
    # 生成jasper_dynamic.onnx
    python3.7 pth2onnx.py
    ```

2. 利用ATC工具转换为om
    ```shell
    # 生成jasper.om，输入shape可以在脚本中修改，默认feats:4,64,4000  feat_lens:4
    bash atc.sh jasper_dynamic.onnx jasper
    ```

3. pyACL推理
    ```shell
    bash om_infer.sh
    ```

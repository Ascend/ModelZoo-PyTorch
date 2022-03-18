## Tacotron2模型推理指导：

### 文件说明
    1. atl_net.py：pyacl推理工具代码
    2. data_process.py：数据预处理代码
    3. onnx_infer.py：waveglow后处理代码，基于onnxruntime推理实现
    4. om_infer_acl.py：Tacotron2推理代码，基于om推理，Tacotron2输出经过waveglow处理写音频文件


### 环境依赖
    dllogger安装
    1. 命令行安装：pip3 install 'git+https://github.com/NVIDIA/dllogger'
    2. 源码安装：
    下载源码：https://github.com/NVIDIA/dllogger
    解压后进入dllogger目录，执行python3.7.5 setup.py install
    其他python包安装
    pip3 install librosa==0.8.0
    pip3 install scipy==1.6.2
    pip3 install torch==1.5.0
    pip3 install onnxruntime==1.7.0
    pip3 install numpy==1.19.5
    pip3 install onnx==1.7.0
    pip3 install wave==0.0.2

### 环境准备
    下载tacotron2源码或解压文件
    1. 下载链接：https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2
    接入进入DeepLearningExamples-master\PyTorch\SpeechSynthesis\Tacotron2目录下。
    
    2. 文件替换及新增
    拷贝内如如下：
    拷贝 acl_net.py, encoder_modify.py, decoder_iter_modify.py, om_infer_acl.py, atc_static.sh 文件到 DeepLearningExamples-master\PyTorch\SpeechSynthesis\Tacotron2目录下；
    拷贝convert_tacotron22onnx.py、convert_waveglow2onnx.py到DeepLearningExamples-master\PyTorch\SpeechSynthesis\Tacotron2\tensorrt目录下
    拷贝nvidia_tacotron2pyt_fp32_20190427, nvidia_waveglow256pyt_fp16到DeepLearningExamples-master\PyTorch\SpeechSynthesis\Tacotron2\checkpoints目录下pth导出ONNX
    从下载tacotron2和waveglow的权重文件，建议下载fp32的权重文件
    tacotron2：https://ngc.nvidia.com/catalog/models/nvidia:tacotron2_pyt_ckpt_fp32
    waveglow：https://ngc.nvidia.com/catalog/models/nvidia:waveglow_ckpt_fp32

### 推理端到端步骤

1. pth导出onnx
    tacotron2
    ```
    python3.7 tensorrt/convert_tacotron22onnx.py --tacotron2 ./checkpoints/nvidia_tacotron2pyt -o output/ --fp32 --batch_size=4
    ```

    waveglow
    ```
    python3.7 tensorrt/convert_waveglow2onnx.py --waveglow ./checkpoints/nvidia_waveglow256pyt --config-file config.json --wn-channels 256 -o output/ --fp32
    ```

2. ONNX改图优化
    参考encoder_modify.py代码
    实现LSTM双向改2个单向, 删除LSTM的两个输入（initial_h和initial_c）

    ```
    python3.7 encoder_modify.py ./output/encoder.onnx ./output/encoder_modify.onnx
    ```

    参考decoder_iter_modify.py代码，修改RandomUniform算子，替换为输入节点，提升网络性能
    ```
    python3.7 decoder_iter_modify.py ./output/decoder.onnx ./output/decoder_iter_modify.onnx
    ```

3. 利用ATC工具转换为om
    ```
    bash atc_static.sh Ascend310 4
    ```   

4. pyACL推理
    设置pyACL环境变量：
    ```
    export PYTHONUNBUFFERED=1
    export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/pyACL/python/site-packages/acl:$PYTHONPATH
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/:$LD_LIBRARY_PAT
    ```

    ```
    python3.7 om_infer_acl.py -i filelists/ljs_audio_text_test_filelist.txt -bs 4 -max_inputlen 128 --device_id 0
    ```


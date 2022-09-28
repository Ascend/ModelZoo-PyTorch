## Tacotron2模型推理指导：

### 文件说明
    1. atl_net.py：pyacl推理工具代码
    2. data_process.py：数据预处理代码
    3. onnx_infer.py：waveglow后处理代码，基于onnxruntime推理实现
    4. om_infer_acl.py：Tacotron2推理代码，基于om推理，Tacotron2输出经过waveglow处理写音频文件


### 环境依赖
    om_gener安装
    git clone https://gitee.com/peng-ao/om_gener.git
    cd om_gener
    pip3 install .
    dllogger安装
    1. 命令行安装：pip3 install 'git+https://github.com/NVIDIA/dllogger'
    2. 源码安装：
    下载源码：https://github.com/NVIDIA/dllogger
    解压后进入dllogger目录，执行python3.7.5 setup.py install
    其他python包安装
    librosa==0.8.0
    scipy==1.7.1
    torch==1.8.0
    onnxruntime==1.6.0
    numpy==1.21.5
    onnx=1.12.0
    wave=0.0.2
    Unidecode==1.3.3
    inflect==5.4.0
    onnxoptimizer==0.2.6
    其他包请按需安装


### 环境准备
    下载tacotron2源码或解压文件
    1. 
    git clone https://github.com/NVIDIA/DeepLearningExamples.git
    cd DeepLearningExamples
    git reset --hard 9a6c5241d76de232bc221825f958284dc84e6e35
    cd PyTorch/SpeechSynthesis/Tacotron2
    tacotron_path=$(pwd)
    mkdir output
    mkdir checkpoints
    2. 文件替换及新增
    拷贝ModelZoo上下载的Tacotron2_for_Pytorch目录下的文件到$tacotron_path路径：
    cp acl_net.py onnx_infer.py addweight.py om_infer_acl.py data_process.py atc_static.sh onnxsim.sh $tacotron_path
    cp get_out_node.py get_out_type.py $tacotron_path/output
    cp convert_tacotron22onnx.py convert_waveglow2onnx.py $tacotron_path/tensorrt
    权重下载链接
    tacotron2：https://ngc.nvidia.com/catalog/models/nvidia:tacotron2_pyt_ckpt_fp32
    waveglow：https://ngc.nvidia.com/catalog/models/nvidia:waveglow_ckpt_fp32
    cp nvidia_tacotron2pyt_fp32_20190427 nvidia_waveglowpyt_fp32_20190427 $tacotron_path/checkpoints

### 推理端到端步骤

首先需侵入式修改onnx文件，修改/usr/local/python3.7.5/lib/python3.7/site-packages/onnx/__init__.py， 在load_model函数中函数首添加load_external_data=False，并且在/usr/local/python3.7.5/lib/python3.7/site-packages/onnx/checker.py中的check_model中C.check_model(protobuf_string)注释掉，以上文件路径可能与描述不同，在自己安装的onnx路径下查找即可

1. pth导出onnx
    ①生成权重并将权重备份
    
    ```
    cd $tacotron_path
python3.7 tensorrt/convert_tacotron22onnx.py --tacotron2 ./checkpoints/nvidia_tacotron2pyt_fp32_20190427 -o output/ --fp32 --batch_size=4 --iter=1
    cd output
    mkdir iter
    mv 336 337 355 356 357 358 359 378 379 380 tacotron2.decoder.attention_layer.location_layer.location_conv.conv.weight tacotron2.decoder.gate_layer.linear_layer.weight tacotron2.decoder.linear_projection.linear_layer.weight ./iter
    注意使用不同版本的onnx生成的权重节点名称可能不同，请自行更改上述生成的权重名称
    ```

生成100层或其它层数叠加的decoder模型（注意该步骤生成的权重占用磁盘空间约7GB，请注意预留空间）    

    cd $tacotron_path
    python3.7 tensorrt/convert_tacotron22onnx.py --tacotron2 ./checkpoints/nvidia_tacotron2pyt_fp32_20190427
    -o output/ --fp32 --batch_size=4 --iter=100
    将ouput目录下新生成的权重删除
    
    生成waveglow模型
    python3.7 tensorrt/convert_waveglow2onnx.py --waveglow ./checkpoints/nvidia_waveglowpyt_fp32_20190427 --config-file config.json -o output/ --fp32

2. ONNX改图优化
   
    修改decoder_iter模型，将权重共享并加载到onnx中
```
    python3 addweight.py $tacotron_path/output/iter $tacotron_path/output/decoder_iter.onnx $tacotron_path/output/decoder_iter_weight.onnx
    ./onnxsim.sh $tacotron_path/output/decoder_iter_weight.onnx $tacotron_path/output/decoder_sim_100.onnx 4 128
```



1. 利用ATC工具转换为om

    310P上执行，执行时使npu-smi info查看设备状态，确保device空闲

${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)   

首先设置环境变量(默认路径为/usr/local/Ascend,其他路径请自行更改)以及日志等级： 

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_GLOBAL_LOG_LEVEL=3
/usr/local/Ascend/driver/tools/msnpureport -g error -d 0
```

转模型

    bash atc_static.sh Ascend${chip_name} 4 128# Ascend310P3
    #参数分别为芯片类型，模型batchsize, seq_len

4. pyACL推理
   

    ```
    python3.7 om_infer_acl.py -i filelists/ljs_audio_text_test_filelist.txt -bs 4 -max_inputlen 128 -max_decode_iter 20 --device_id 0
    ```


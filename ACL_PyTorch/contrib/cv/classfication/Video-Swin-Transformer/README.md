# VideoSwinTransformer模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

    
- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Video swin transformer是一种基于Transformer的视频识别主干网络结构，并且它在效率上超过了以前的分解时空建模的模型。其结构中利用了假设偏置，所以达到了更高的建模效率。由于这一特性，全局的时空Self-Attention可以近似为多个局部Self-Attention的计算，从而大大节省计算和模型规模。 Video Swin Transformer严格遵循原始Swin Transformer的层次结构，但将局部注意力计算的范围从空间域扩展到时空域。由于局部注意力是在非重叠窗口上计算的，因此原始Swin Transformer的滑动窗口机制也被重新定义了，以适应时间和空间两个域的信息。 

- 参考实现：

  ```
  url= https://github.com/SwinTransformer/Video-Swin-Transformer.git
  commit_id= db018fb8896251711791386bbd2127562fd8d6a6
  model_name= Video-Swin-Transformer
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                                  | 数据排布格式 |
  | -------- |-------------------------------------| ------------------------- | ------------ |
  | input    | FLOAT32 | batchsize x 12 x 3 x 32 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小       | 数据排布格式 |
  | -------- |----------| -------- | ------------ |
  | output  | FLOAT32  | 12 x 400 | ND           |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。
   ```
   git clone https://github.com/SwinTransformer/Video-Swin-Transformer.git     # 克隆仓库的代码
   cd Video-Swin-Transformer     # 切换到模型的代码仓目录
   git reset --hard db018fb8896251711791386bbd2127562fd8d6a6
   
   ```
   需要将库中代码放置在源码仓目录下，目录结构如下
      ```
   Video-Swin-Transformer
   ├── data           
   ├─ config
   ├─ mmaction   
   ├─   ...
   ├─ video_swin_transformer.patch
   ├─ video_swin_transformer_inference.py
   ├─ video_swin_transformer_modify.py
   ├─ video_swin_transformer_preprocess.py
   └─ video_swin_transformer_postprocess.pyl          
   ```
   将patch应用到当前源码
   ```
   git apply --check video_swin_transformer.patch # 检查patch是否可用
   git apply video_swin_transformer.patch # 将patch应用到当前源码 
   ```

2. 安装依赖。
   
   ```shell
   #安装1.11.0版本pytorch
   pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu
   #安装对应pytorch版本的mmcv库
   pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.5/index.html
   #安装mmaction2库
   pip install -v -e .
   #安装其他所需依赖
   pip install -r requirements.txt
   #参考下方链接仓库dev分支安装magiconnx
   https://gitee.com/Ronnie_zheng/MagicONNX
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
   
   本模型支持kinetics400验证集。用户需自行获取数据集，将数据集解压并上传数据集到源码包路径下。目录结构如下：
   ```
   Video-Swin-Transformer
   ├── data           
        └── kinetics400             
                ├─ annotations      //标注信息文件夹
                └─ val              // 验证集文件夹
                    ├─ abseiling
                          ├─ 0wR5jVB-WPk.mp4
                          ├─ 3caPS4FHFF8.mp4
                          ├─ ...
                          └─ Zv09PB3YQAs.mp4
                    ├─ air_drumming
                    ├─ answering_questions
                    ├─ ...
                    └─ zumba
   ```

2. 生成数据集filelist文件。

   数据集中val/crossing_river/ZVdAl-yh2m0.mp4文件size为0，需删除。
   ```
   cd ./val/crossing_river/
   rm -rf ZVdAl-yh2m0.mp4
   ```
   删除后运行如下命令
   ```
   cd tools/data/kinetics
   bash generate_videos_filelist.sh kinetics400
   ```
   运行成功后将会在Video-Swin-Transformer/data/kinetics400路径下生成kinetics400_val_list.txt文件

3. 数据预处理，将原始数据集转换为模型输入的数据。

   执行video_swin_transformer_preprocess.py脚本，完成预处理。

   ```
   python3 video_swin_transformer_preprocess.py configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py --save_path=./bin1 
   ```
   - 参数说明：
     - -- save_path: 预处理后生成bin文件的存储路径
     
## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
      [swin_small_patch244_window877_kinetics400_1k.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_small_patch244_window877_kinetics400_1k.pth)
      将下载好的权重文件放入源码根目录
   2. 导出onnx文件。

      1. 使用pytorch2onnx.py导出onnx文件。

         运行pytorch2onnx.py脚本。

         ```
         python3 tools/deployment/pytorch2onnx.py configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py swin_small_patch244_window877_kinetics400_1k.pth --output-file video_swin.onnx --shape 1 12 3 32 224 224 
         ```

         获得video_swin.onnx文件。

      2. 优化ONNX文件。

         ```
         python3 -m onnxsim video_swin.onnx video_swin_sim.onnx
         ```

         获得video_swin_sim.onnx文件。
      3. 对onnx文件进行图修改
         ```
         python3 video_swin_transformer_modify.py video_swin_sim.onnx video_swin_mod.onnx
         ```

         获得video_swin_mod.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         回显如下：
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 15.8         42                0    / 0              |
         | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
         +===================+=================+======================================================+
         | 1       310P3     | OK              | 15.4         43                0    / 0              |
         | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
         +===================+=================+======================================================+
         ```

      3. 执行ATC命令。

         ```
         atc --framework=5 \
             --model=video_swin_mod.onnx \
             --output=video_swin \
             --soc_version=Ascend${chip_name} \
             --input_shape="onnx::Reshape_0:1,12,3,32,224,224" \
             --op_precision_mode=op_precision.ini
         ```

         - 参数说明：
           - --model：为ONNX模型文件。
           - --framework：5代表ONNX模型。
           - --output：输出的OM模型。
           - --input\_shape：输入数据的shape。
           - --soc\_version：处理器型号。
           - --op\_precision\_mode: 配置文件

           运行成功后生成video_swin.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```
        mkdir ./out
        python3 -m ais_bench --model video_swin.om --input ./bin1 --output ./out --output_dir 1 --batchsize 1
        ```
        -   参数说明：

             - --model：进行推理的om文件路径。
             - --input：推理数据路径。
             - --output：推理结果的输出路径。
             - --output_dir: 输出子目录
             - --batchsize：模型输入的batch大小

        推理后的输出在当前目录out下。


   3. 精度验证。

      调用video_swin_transformer_postprocess.py脚本，可以获得Accuracy数据。

      ```
       python3 video_swin_transformer_postprocess.py configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py --data_path ./out --eval top_k_accuracy
      ```

      - 参数说明：
        - --data_path：推理输出数据
        - --eval：结果评估方式

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 -m ais_bench --model=${om_model_path} --loop=100 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型路径
        - --loop：循环次数
        - --batchsize：om模型输入的batch大小
   5. 全数据精度验证。 
   
      由于数据集较大，预处理后和aisinfer工具推理生成的输出数据可能会导致内存空间不足
      因此，可使用video_swin_transformer_inference.py脚本进行全数据集的精度验证，参考命令如下：

        ```
         python3 video_swin_transformer_inference.py configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py --model_path ${om_model_path} --eval top_k_accuracy
        ```

      - 参数说明：
        - --model_path：om模型路径
        - --eval：结果评估方式



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集         | 精度                        | 性能    |
| --------- |-----|-------------|---------------------------|-------|
|  Ascend310P3    | 1   | kinetics400 | top1:0.806<br/>top5:0.945 | 0.607 |
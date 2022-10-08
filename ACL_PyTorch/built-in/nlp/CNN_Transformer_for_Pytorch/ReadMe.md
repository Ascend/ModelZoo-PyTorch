# 参考库文

- [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)

# 参考实现

- [facebook/wav2vec2-base-960h]

# 依赖

| 依赖名称     | 版本         |
| ------------ | :----------- |
| pytorch      | 1.6.0        |
| soundfile    | 0.10.3.post1 |
| transformers | 4.3.3        |
| tqdm         | 4.49.0       |
| numpy        | 1.19.5       |
| datasets     | 1.6.2        |
| jiwer        | 2.2.0        |

# 准备数据集

运行`preprocess.py`脚本，会自动在线下载所需的分词器模型、Librispeech数据集（下载过程可能比较长），并把数据处理为bin文件，同时生成数据集的info文件。

```
python3.7 preprocess.py --pre_data_save_path=./pre_data/clean --which_dataset=clean
```

参数说明：

- --pre_data_save_path：预处理数据保存路径
- --which_dataset：指定所用的数据集
  - validation：patrickvonplaten/librispeech_asr_dummy数据集，特别小，只有70多条音频数据
  - clean：Librispeech clean数据集
  - other：Librispeech other数据集

官方提供了模型在Librispeech clean和Librispeech other数据集上的精度，本示例中仅用Librispeech clean测试精度。

# 模型推理

1. 模型转换。

   使用PyTorch将模型权重文件pth转换为onnx文件，再使用atc工具将onnx文件转为离线推理模型om文件。

   1. 导出onnx文件。

      运行`export_onnx.py`脚本，会自动在线下载pth模型，并把pth模型转换为onnx模型。

      ```
      python3.7 export_onnx.py --model_save_dir=./models
      ```

      运行完之后，会在当前目录的`models`目录下生成`wav2vec2-base-960h.onnx`模型文件。

      使用atc工具将onnx文件转换为om文件，导出onnx模型文件时需设置算子版本为11。

   2. 使用atc工具将onnx模型转om模型。

      1. 根据实际情况，修改`onnx2om.sh`脚本中的环境变量，具体的脚本示例如下：

         ```
         #!/bin/bash
         source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
         
         atc --framework=5 --model=./models/wav2vec2-base-960h.onnx --output=./models/wav2vec2-base-960h --input_format=ND --input_shape="input:1,-1" --dynamic_dims="10000;20000;30000;40000;50000;60000;70000;80000;90000;100000;110000;120000;130000;140000;150000;160000;170000;180000;190000;200000;210000;220000;230000;240000;250000;260000;270000;280000;290000;300000;310000;320000;330000;340000;350000;360000;370000;380000;390000;400000;410000;420000;430000;440000;450000;460000;470000;480000;490000;500000;510000;520000;530000;540000;550000;560000" --log=error --soc_version=$1
         ```
      
         参数说明：
         
         - --model：为ONNX模型文件。
         - --framework：5代表ONNX模型。
         - --output：输出的OM模型。
         - --input_format：输入数据的格式。
         - --input_shape：输入数据的shape。
         - --log：日志等级。
         - --soc_version：部署芯片类型。
      
      2. 执行onnx2om.sh脚本，将onnx文件转为离线推理模型文件om文件。
      
         ${chip_name}可通过`npu-smi info`指令查看

         ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)
         
         ```
         bash onnx2om.sh Ascend${chip_name} # Ascend310P3
         ```
      
         运行成功后在`models`目录下生成`wav2vec2-base-960h.om`模型文件。

2. 开始推理验证。

   1. 配置环境变量

      ```
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      ```

      `install_path`请修改为Toolkit的实际安装路径。
   
   2. 运行`pyacl_infer.py`进行推理，同时输出推理性能数据。

      ```
   python3.7 pyacl_infer.py \
        --model_path=./models/wav2vec2-base-960h.om \
        --device_id=0 \
        --cpu_run=True \
        --sync_infer=True \
        --workspace=0 \
        --input_info_file_path=./pre_data/clean/bin_file.info \
        --input_dtypes=float32 \
        --infer_res_save_path=./om_infer_res_clean \  --res_save_type=bin
      ```
   
      参数说明：
   
      - --model_path：模型路径
      - --device_id：npu id
      - --cpu_run：MeasureTime类的cpu_run参数，True or False
      - --sync_infer：推理方式：
        - True：同步推理
        - False：异步推理
      - --workspace：类似TensorRT‘workspace’参数，计算平均推理时间时排除前n次推理
      - --input_info_file_path：bin_info文件
      - --input_dtypes：模型输入的类型，用逗号分割（参考`DTYPE`变量）
        - e.g. 模型只有一个输入：--input_dtypes=float32
        - e.g. 模型有多个输入：--input_dtypes=float32,float32,float32（需要和bin_info文件多输入排列一致）
      - --infer_res_save_path：推理结果保存目录
      - --res_save_type：推理结果保存类型，bin或npy
   3. 推理数据后处理与精度统计。
   
   运行`postprocess.py`，会进行推理数据后处理，并进行精度统计。
   
   ```
   python3.7 postprocess.py \
      --bin_file_path=./om_infer_res_clean \
   --res_save_path=./om_infer_res_clean/transcriptions.txt \
      --which_dataset=clean
      ```
      
      参数说明：
      
      - --bin_file_path：pyacl推理结果存放路径
      - --res_save_path：后处理结果存放txt文件
      - --which_dataset：精度统计所用的数据集，参看preprocess.py的参数说明
   
4. 性能测试
   
   由于TensorRT无法运行`wav2vec2-base-960h.onnx`模型，所以性能测试以pyacl得到的om推理性能和pytorch在线推理性能作比较。
   
      在GPU环境上运行`pth_online_infer.py`脚本，得到pytorch在线推理性能。
   
      ```
      python pth_online_infer.py \
      --pred_res_save_path=./pth_online_infer_res/clean/transcriptions.txt \
   --which_dataset=clean
      ```
   
      参数说明：
      
      - --pred_res_save_path：pytorch在线推理结果存放路径
      - --which_dataset：参看preprocess.py的参数说明
      
      脚本执行完毕后，会在屏幕上输出pytorch在线推理平均推理时间（average infer time(ms)），假定为![Figure Name:202155144621.png CAD Name:zh-cn_formulaimage_0000001124002380.png](http://resource.idp.huawei.com/idpresource/nasshare/editor/image/34040284354/1_zh-cn_formulaimage_0000001124002380.png)，换算为单卡后pytorch在线推理的每秒推理数量为：![Figure Name:202153161710.png CAD Name:zh-cn_formulaimage_0000001166163171.png](http://resource.idp.huawei.com/idpresource/nasshare/editor/image/34040284354/3_zh-cn_formulaimage_0000001166163171.png)。
      
      上述运行pyacl_infer.py脚本会得到om平均推理时间（average infer time(ms)），假定为![Figure Name:202153161914.png CAD Name:zh-cn_formulaimage_0000001166164017.png](http://resource.idp.huawei.com/idpresource/nasshare/editor/image/34040284354/1_zh-cn_formulaimage_0000001166164017.png)，换算为单卡后om的每秒推理数量为：![Figure Name:202153161758.png CAD Name:zh-cn_formulaimage_0000001119363638.png](http://resource.idp.huawei.com/idpresource/nasshare/editor/image/34040284354/2_zh-cn_formulaimage_0000001119363638.png)。
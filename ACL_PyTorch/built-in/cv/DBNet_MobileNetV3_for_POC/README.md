# det_mv3_db_v2.5 模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [性能&精度](#ZH-CN_TOPIC_0000001172201573)



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

分割在文本检测中很流行，而二值化的后处理对于分割的检测至关重要，该检测方法将由分割方法生成的概率图转换为文本的边界框/区域。在该网络中，我们提出了一个名为可微分二值化（DB）的模块，它可以在分割网络中执行二值化过程。与DB模块一起优化的分割网络可以自适应地设置用于二值化的阈值，这不仅简化了后处理，还提高了文本检测的性能。

- 参考实现：

  ```
  url=https://github.com/PaddlePaddle/PaddleOCR.git
  branch=release/2.5
  commit_id=7f2d05cfe4e
  model_name=PPOCRV2.5_det_db_mv3 
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据
 
  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ----------------------- | ------------ |
  | x        | uint8    | 24 x 3 x 736 x 1280     | NCHW         |

- 输出数据

  | 输出数据  | 数据类型  | 大小                 | 数据排布格式 |
  | -------- | -------- | -------------------- | ----------- |
  | output  | FLOAT16   | 24 x 1 x 736 x 1280  | NCHW        |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动


| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                    | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.3.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。  | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>


1. 安装依赖。

    1. 安装基础环境
    ```bash
    pip3 install -r requirements.txt
    ```
    说明：某些库如果通过此方式安装失败，可使用pip单独进行安装。

    2. 安装ONNX改图工具
    ```bash
    git clone https://gitee.com/ascend/msadvisor.git
    cd msadvisor/auto-optimizer
    pip install --upgrade pip
    pip install wheel
    pip install .
    cd ../../
    rm -rf msadvisor
    ```

    
    3. 安装量化工具
    ```bash
    pip install protobuf==3.20.0
    pip install onnxruntime==1.8.0  # need onnxruntime-1.8.0 to build custom operators
    arch=`arch`
    wget --no-check-certificate -O amct.tar.gz https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Florence-ASL/Florence-ASL%20V100R001C30SPC703/Ascend-cann-amct_6.3.RC2.alpha003_linux-${arch}.tar.gz?response-content-type=application/octet-stream
    tar -zxvf amct.tar.gz && cd amct/amct_onnx/
    pip install amct_onnx-0.10.1-py3-none-linux_aarch64.whl
    tar -zxvf amct_onnx_op.tar.gz
    cd amct_onnx_op && python setup.py build
    cd ../../../
    rm -rf amct/ amct.tar.gz
    pip install onnxruntime==1.14.1  # need onnxruntime-1.14.1 to quantize.
    ```

    4. 安装OM推理工具
    ```bash
    arch=`arch`
    wget --no-check-certificate https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/aclruntime-0.0.2-cp37-cp37m-linux_${arch}.whl
    pip install aclruntime-0.0.2-cp37-cp37m-linux_${arch}.whl
    rm aclruntime-0.0.2-cp37-cp37m-linux_${arch}.whl
    wget --no-check-certificate https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ais_bench-0.0.2-py3-none-any.whl
    pip install ais_bench-0.0.2-py3-none-any.whl
    rm ais_bench-0.0.2-py3-none-any.whl
    ```

2. 获取源码。

    ```bash
    git clone -b release/2.5 https://github.com/PaddlePaddle/PaddleOCR.git
    cd PaddleOCR 
    git reset --hard 7f2d05cfe4e
    python3 setup.py install
    ```


## 准备数据集<a name="section183221994411"></a>

- 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

    该模型使用ICDAR2015测试集中的500张图片来验证模型精度。请参考[该链接](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/dataset/ocr_datasets.md)下载对应数据集文件并解压，本项目需要的数据目录结构如下:
    ```
    ./icdar2015/text_localization/
        ├── ch4_test_images/
        │   ├── img_1.jpg
        │   ├── ...
        │   └── img_500.jpg
        └── test_icdar2015_label.txt
    ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换。
  
    1. 获取Paddle模型。

        该模型的训练模型链接为：https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar
        ```bash
        wget -P ./checkpoint --no-check-certificate https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar
        cd ./checkpoint && tar xf det_mv3_db_v2.0_train.tar && cd ..
        ```

        以上命令执行结束后，继续执行以下命令将训练模型转换为推理模型：
        ```bash
        python3 PaddleOCR/tools/export_model.py \
            -c PaddleOCR/configs/det/det_mv3_db.yml \
            -o Global.pretrained_model=./checkpoint/det_mv3_db_v2.0_train/best_accuracy  \
                Global.save_inference_dir=./inference/ \
                Global.use_gpu=False
        ```

     2. 导出ONNX文件

        在当前目录下运行`paddle2onnx`命令，运行成功后，当前目录下会生成`db_mv3.onnx`文件。
        ```bash
        mkdir models
        paddle2onnx \
            --model_dir ./inference/ \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ./models/db_mv3.onnx \
            --opset_version 11 \
            --enable_onnx_checker True \
            --input_shape_dict="{'x':[-1,3,-1,-1]}"
        ```

        `paddle2onnx`的用法请通过`paddle2onnx -h`命令查看。
        
        生成ONNX模型后，执行以下命令，对ONNX模型做些许修改以提升性能：
        ```bash
        # usage: modify_onnx.py [-h] <input_onnx> <output_onnx>
        
        # modify onnx model.
        # positional arguments:
        #     input_onnx   path to input onnx file.
        #     output_onnx  path to save modified onnx model.

        python3 modify_onnx.py ./models/db_mv3.onnx ./models/db_mv3_opti.onnx
        ```
        说明：参数1为原始ONNX路径，参数2为修改后ONNX的保存路径。

        运行结束后，当前`./models`下将会生成修改后的ONNX模型`db_mv3_opti.onnx`。

    3. 模型量化
        在量化前，我们需要生成一份配置文件，作用是在量化时跳过下面两部分节点：
        1. Dequant节点会破坏部分Conv/ConvTranspose与后续节点融合进而导致性能劣化，所以跳过这部分Conv/ConvTranspose节点。
        2. 跳过模型尾部少量导致模型精度下降的Conv/ConvTranspose节点。
        
        执行以下命令即可自动生成配置文件：
        ```bash
        # usage: create_quant_config.py [-h] <input_onnx> <output_config>
        
        # create quantization config for AMCT.
        # positional arguments:
        #     input_onnx     path to onnx file.
        #     output_config  path to save quantization config.

        python3 create_quant_config.py ./models/db_mv3_opti.onnx ./quant.cfg
        ```

        然后生成量化时用于校准精度的数据：
        ```bash
        python3 create_quant_data.py \
        -c PaddleOCR/configs/det/det_mv3_db.yml \
        -o Global.use_gpu=False \
           data_dir=./icdar2015/text_localization/ \
           batch_size=24 \
           save_dir=./quant_data/
        ```

        最后使用`amct`工具，对ONNX模型进行量化，以进一步提升模型性能：
        ```bash
        amct_onnx calibration \
            --model ./models/db_mv3_opti.onnx \
            --save_path ./models/quanted \
            --input_shape "x:24,3,736,1280" \
            --data_dir "./quant_data/" \
            --data_types "float32" \
            --calibration_config ./quant.cfg
        ```
        量化后的模型存放路径为 `models/quanted_deploy_model.onnx`。


    4. 使用ATC工具将ONNX模型转OM模型。

        （1）配置环境变量。
        ```bash
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        ```

        > **说明：** 
        >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

        （2）执行命令查看芯片名称（$\{chip\_name\}）。
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

        （3）执行ATC命令。
      
        在`PaddleOCR`目录执行
        ```bash
        atc --framework=5 \
            --model=models/quanted_deploy_model.onnx \
            --output=models/db_mv3 \
            --input_format=NCHW \
            --input_shape="x:24,3,736,1280" \
            --insert_op_conf=./aipp.cfg \
            --enable_small_channel=1 \
            --output_type=FP16 \
            --op_select_implmode=high_performance \
            --optypelist_for_implmode=Sigmoid \
            --soc_version=Ascend310P3 \
            --log=error
        ```

        运行成功后，当前`models`下将会生成`db_mv3.om`模型文件。`atc`各参数的含义请参考：

        -    --framework：5代表ONNX模型
        -    --model：为ONNX模型文件
        -    --output：输出的OM模型
        -    --input_format：输入数据的格式
        -    --input_shape：输入数据的shape
        -    --insert_op_conf：aipp配置文件
        -    --enable_small_channel: 使能C0特效 
        -    --output_type：输出数据类型
        -    --op_select_implmode: 算子模式的选择
        -    --optypelist_for_implmode：具体哪个算子
        -    --soc_version：处理器型号
        -    --log：日志级别  



2. 推理验证

    该模型使用`ais_bench`工具进行推理，其安装、用法等详细信息请参考[ais_bench工具Gitee主页](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)


    1. 执行推理并验证精度
        ```bash
        python3 eval_npu.py \
        -c PaddleOCR/configs/det/det_mv3_db.yml \
        -o Global.use_gpu=False \
           data_dir=./icdar2015/text_localization/ \
           om_path=models/db_mv3.om \
           device_id=0
        ```

        参数说明：
        -    -c：配置文件
        -    -o Global.use_gpu：是否使能gpu
        -    -o data_dir: 测试集路径
        -    -o om_path：OM模型路径
        -    -o device_id：选择NPU的device_id

        推理完成后， 模型的精度指标会打屏显示。
   
    2. 纯推理验证性能
        ```bash
        python3 -m ais_bench --model models/db_mv3.om --loop 100 --batchsize 24
        ```
        参数说明：
        - --model: OM模型路径
        - --loop: 循环次数
        - --batchsize: 模型输入的batchsize


# 性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

| 芯片型号   | Batch Size   | 数据集    | 精度         | 性能   |
| --------- | ------------ | --------- | ----------- |------- |
|310P3      | 24           | ICDAR2015 | hmean=74.3% | 225fps |

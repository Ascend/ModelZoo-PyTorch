# Resnet50模型PyTorch离线推理指导

## 1 准备数据集

1. 获取原始数据集
   
   本模型使用ImageNet 50000张图片的验证集，请前往ImageNet官网下载数据集

    ```
    ├── ImageNet
    |   ├── val
    |   |    ├── ILSVRC2012_val_00000001.JPEG
    │   |    ├── ILSVRC2012_val_00000002.JPEG
    │   |    ├── ......
    |   ├── val_label.txt
    ```

2. 数据预处理

    1. 将原始数据集转换为模型输入的数据

        ```
        python3.7.5 imagenet_torch_preprocess.py resnet ./ImageNet/val ./prep_dataset
        ```

        每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成prep_dataset二进制文件夹

    2. 获取benchmark推理工具所需的数据集info文件

        ```
        python3.7.5 gen_dataset_info.py bin ./prep_dataset ./resnet50_prep_bin.info 256 256
        ```

        运行成功后，在当前目录下生成resnet50_prep_bin.info文件


## 3 模型生成

1. 生成onnx模型

   1. 获取权重文件

        前往[Pytorch官方文档](https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet50)下载
    
   2. 导出onnx文件

        ```
        python3.7.5 pth2onnx.py ./resnet50-0676ba61.pth
        ```

2. 将onnx转为om模型

   1. 配置环境变量
        ```
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        ```

   2. 使用ATC工具将onnx模型转om模型

        ${chip_name}可通过`npu-smi info`指令查看

        ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

        ```
        # Ascend310 or Ascend310P[1-4]
        atc --model=resnet50_official.onnx --framework=5 --output=resnet50_bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --enable_small_channel=1 --log=error --soc_version=Ascend${chip_name} --insert_op_conf=aipp_resnet50.aippconfig
        ```

        运行成功后生成resnet50_bs16.om模型文件

## 4 离线推理 

1. 安装benchmark工具

    参考[benchmark工具源码地址](https://gitee.com/ascend/cann-benchmark/tree/master/infer)安装
   
2. 模型推理

    ```
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ./benchmark.x86_64 -model_type=vision -batch_size=16 -device_id=0 -input_text_path=resnet50_prep_bin.info -input_width=256 -input_height=256 -om_path=./resnet50_bs16.om -useDvpp=False -output_binary=False
    ```

    运行成功后会在result/dumpOutput_device0下生成推理输出的txt文件

3. 精度验证

    统计推理输出的Top 1-5 Accuracy

    ```
    python3.7.5 vision_metric_ImageNet.py result/dumpOutput_device0/ ./ImageNet/val_label.txt ./ result_prep.json
    ```

    运行成功后再当前目录下生成记录精度结果的result_prep.json

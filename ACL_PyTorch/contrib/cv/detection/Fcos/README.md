# Fcos 模型-推理指导


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
   
> Fcos提出了一个简单、灵活、通用的目检测算法。该算法是一种基于FCN的逐像素目标检测算法，实现了无锚点（anchor-free）、无提议（proposal free）的解决方案，并且提出了中心度（Center—ness）的思想，同时在召回率等方面表现接近甚至超过目前很多先进主流的基于锚框目标检测算法。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection
  commit_id=dd0e8ede1f6aa2b65e8ce69826314b76751d4151
  model_name=contrib/cv/detection/Fcos
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 800 x 1333 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | int64  | batchsize x 100 | ND           |
  | output2  | FLOAT32  | batchsize x 100 x 5 | ND           |





# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.6.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   - 安装mmcv

      ```shell
      git clone https://github.com/open-mmlab/mmcv
      cd mmcv
      MMCV_WITH_OPS=1 pip3.7 install -e .
      cd ..
      ```
   - 安装mmdetection并安装补丁
      ```shell
      git clone https://github.com/open-mmlab/mmdetection -b master
      cd mmdetection
      git reset --hard dd0e8ede1f6aa2b65e8ce69826314b76751d4151
      cp ../fcos_r50_caffe_fpn_4x4_1x_coco.py ./configs/fcos  
      patch -p1 < ../fcos.diff
      pip3 install -r requirements/build.txt
      python3 setup.py develop
      cd ..
      ```
2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）



   本模型支持coco2017验证集。用户需自行获取数据集（或给出明确下载链接），val2017目录存放coco数据集的验证集图片，annotations目录存放coco数据集的instances_val2017.json。目录结构如下：

   ```
   coco
   ├── annotations    //验证集标注信息       
   └── val2017        // 验证集文件夹
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   - 执行`fcos_pth_preprocess.py.py`脚本，完成预处理。

   ```
   python3 fcos_pth_preprocess.py --image_src_path=./coco/val2017 --bin_file_path=val2017_bin --model_input_height=800 --model_input_width=1333
   ``` 

   - 图片的info文件生成
   ```
   python get_info.py jpg ./coco/val2017 fcos_jpeg.info
   ```

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       ```shell
       wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Fcos/PTH/fcos_r50_caffe_fpn_1x_4gpu_20190516-a7cac5ff.pth
       ```

   2. 导出onnx文件。

      1. 使用`pytorch2onnx`导出onnx文件。

         运行`pytorch2onnx`脚本。

         ```shell
         python3 mmdetection/tools/deployment/pytorch2onnx.py mmdetection/configs/fcos/fcos_r50_caffe_fpn_4x4_1x_coco.py ./fcos_r50_caffe_fpn_1x_4gpu_20190516-a7cac5ff.pth --output-file fcos.onnx --input-img mmdetection/demo/demo.jpg --test-img mmdetection/tests/data/color.jpg --shape 800 1333 --dynamic-export
         
         ```
         获得`fcos.onnx`文件。



   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         会显如下：
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

         ```shell
         atc --framework=5 --model=./fcos.onnx --output=fcos_bs1 --input_format=NCHW --input_shape="input:1,3,800,1333" --log=error --soc_version=Ascend${chip_name} --out_node="labels;dets"
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --out_node：输出节点的顺序

           运行成功后生成`fcos_bs1.om`模型文件。

2. 开始推理验证。
   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais-bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

   2. 执行推理。

        ```
        python3 -m ais_bench --model fcos_bs1.om --input val2017_bin --output ./ --output_dirname result  
        ```

        -   参数说明：

             -   model：om模型
             -   input：输入文件
             -   output：输出路径
             -   output_dirname：输出文件夹
                  	

        推理后的输出默认在当前目录`result`下。

        >**说明：** 
        >执行ais-bench工具请选择与运行环境架构相同的命令

   3. 精度验证。

      调用fcos_pth_postprocess.py评测map精度


      ```
       python fcos_pth_postprocess.py --bin_data_path=./result --test_annotation=fcos_jpeg.info --det_results_path=./ret_npuinfer/ --net_out_num=3 --net_input_height=800 --net_input_width=1333 --annotations_path=./coco/annotations
      ```

      - 参数说明：

        - bin_data_path：为生成推理结果所在路径  
        - test_annotation：图片info文件
        - det_results_path：保存结果路径
        - net_out_num：模型输出顺序数
        - net_input_height：输入图片高
        - net_input_width：输入图片宽
        - annotations_path：原图地址

   4. 性能验证。

      可使用ais-bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型
        - --batchsize：模型batchsize
        - --loop: 循环次数



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度 | 性能 |
| -------- | ---------- | ------ | ---- | ---- |
|    310P3      |     1       |    coco2017    |   MAP:35.9%   |   58  |
|    310P3      |     4       |    coco2017    |   MAP:35.9%   |   65  |
|    310P3      |     8       |    coco2017    |   MAP:35.9%   |   58  |
|    310P3      |     16       |    coco2017    |   MAP:35.9%   |   58  |
|    310P3      |     32       |    coco2017    |   MAP:35.9%   |   58  |
|    310P3      |     64       |    coco2017    |   MAP:35.9%   |   45  |
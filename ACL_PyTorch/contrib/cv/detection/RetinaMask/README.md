# RetinaMask模型-推理指导


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

在RetinaMask中，采用了ResNet和MobileNet。neck是backbone和heads的中间部分，可以增强或细化原始特征图（backbone输出）,在RetinaMask中使用FPN，可以提取高级语义信息，然后利用系数计算进行信息融合。



- 参考实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/contrib/cv/detection/RetinaMask	
  code_path=contrib/cv/detection/RetinaMask
  ```
  
 


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input   | RGB_FP32 | batchsize x 3 x 1344 X 1244 | NCHW        |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 1000 x 4| ND        |
  | output2  | INT32  | batchsize x 1000    | ND        |
  | output3  | FLOAT32 | batchsize x 1000   | ND        |
  | output4  | FLOAT32 | batchsize x 1000 x 1 x 28 x 28 | NCHW |    

 
# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17(NPU驱动固件版本为6.0.RC1)  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | Pytorch                                                      | 1.6.0   | -                                                            |
                    


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 安装依赖。

   ```
   pip install -r requirements.txt     
   ```
>**说明：** 
>torch1.6在arm上不支持pip直接安装，如在arm上复现请参考[官方源码编译步骤](https://github.com/pytorch/pytorch/tree/v1.7.0#installation)安装

2. 获取源码。
    1. 安装npu版retinamask源码
   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch
   cp -r  ./ModelZoo-PyTorch/PyTorch/contrib/cv/detection/RetinaMask ./
   cd RetinaMask
   patch -p1< ../RetinaMask.patch
   cd ..  
   ```

  
## 准备数据集<a name="section183221994411"></a>


1. 数据预处理。

本模型已在coco 2017数据集上验证过精度。推理数据集采用coco_val_2017，请用户自行获取coco_val_2017数据集。将instances_val2017.json文件和val2017文件夹按照如下目录结构上传。
   ```
   ├──datasets 
     └── coco 
       ├──annotations 
           └──instances_val2017.json        
       └── val2017 
   ```                    



   执行RetinaMask_preprocess.py脚本，完成预处理。

   ```
   python ./RetinaMask_preprocess.py --image_src_path=./coco/val2017 --bin_file_path=./bins --bin_info_name="retinamask_coco2017.info"
   ```
   - 参数说明：
     -	--image_src_path：原始数据验证集（.jpg）所在路径。
     -  --bin_file_path：输出的二进制文件（.bin）所在路径。
     - 	--bin_info_name：输出的二进制数据集（.info）文件。


      
运行成功后，会在当前目录下生成二进制文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       1.[下载pth文件](https://pan.baidu.com/s/1qx0WbB627AkPGgNgHHq56Q) 
         提取码：xxzz
      

   2. 导出onnx文件。

      1. 使用RetinaMask_pth2onnx.py导出onnx文件。

     

        ```
        cd RetinaMask
        python ../RetinaMask_pth2onnx.py --cfg_path="./configs/retina/retinanet_mask_R-50-FPN_2x_adjust_std011_ms.yaml" 
        --weight_path="../npu_8P_model_0020001.pth" --save_path="../retinamask.onnx" --simplify=True
        cd ..
        ```

        - 参数说明：

           -   --weight_path：PTH权重路径
           -   --save_path：ONNX文件保存路径。
           -   --simplify：是否简化ONNX模型，默认简化。
           -   --cfg_path：模型使用的YAML配置文件。
          
          

         获得retinamask.onnx文件.该模型只支持bs1.



      2. 执行cast_onnx.py脚本.

        ```
         python cast_onnx.py  --weight_path retinamask.onnx  --save_dir retinamask_cast.onnx
        ```
        >**说明：** 
        >Concat_742节点的input_0与input_1在转换om的过程中，被分别转换为float16与float32，引起报错。因此将onnx模型手动插入cast节点，均转换至float16以避免问题。


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
         atc --framework=5 --model="./retinamask_cast.onnx" --output="./retinamask" --input_format=NCHW --input_shape="input:1,3,1344,1344" 
         --log=error --soc_version=Ascend{chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc_version：处理器型号。
   

        运行成功后生成retinamask.om模型文件。

2. 开始推理验证。

   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

   2. 执行推理。

        
      ```
      python -m ais_bench --model retinamask.om --input bins --output ./ --outfmt BIN --batchsize 1 --output_dirname result
      ```

    - 参数说明：

      - --model: OM模型路径。
      - --input: 存放预处理bin文件的目录路径
      - --output: 存放推理结果的目录路径
      - --batchsize：每次输入模型的样本数
      - --outfmt: 推理结果数据的格式
      - --output_dirname: 输出结果子目录
        推理后的输出默认在当前目录result下

        >**说明：** 
        >执行ais-bench工具请选择与运行环境架构相同的命令。参数详情请参见。

   3. 精度验证。

      调用“RetinaMask_postprocess.py”评测模型的精度。

    ```
    python ./RetinaMask_postprocess.py --input_text_path="./retinamask_coco2017.info" --infer_results_path=./result --coco_path="./coco" -- 
    output_path="./evaluation_results.txt"
    ```
    - 参数说明：

      -   --input_text_path：数据集信息info文件路径。
      -	  --infer_results_path：执行推理后结果保存路径。
      -   --coco_path：coco2017数据集的根目录。
      -	  --output_path：精度结果保存。

 

     
     
    
   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

    ```
    python ais_infer.py --model ./retinamask.om --loop 100 --batchsize 1
    ```

    - 参数说明：

      - --model: om模型
      - --batchsize: 每次输入模型样本数
      - --loop: 循环次数    



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

1. 精度对比

| batch |  310P |
| ----- | ----- |
| 1     | bbox 0.279  segm 0.248 |



2. 性能对比  

| batchsize | 310P     | 
| --------- | -------- | 
| 1         |  4.3fps | 

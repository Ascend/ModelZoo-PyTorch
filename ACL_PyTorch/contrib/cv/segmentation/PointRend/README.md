# PointRend模型PyTorch离线推理指导

## 1 环境准备


1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

```
pip3.7 install -r requirements.txt  
```

2.获取，修改与安装开源模型代码

```
git clone https://github.com/facebookresearch/detectron2
cd detectron2
git reset 861b50a8894a7b3cccd5e4a927a4130e8109f7b8 --hard
patch -p1 < ../PointRend.diff
cd ..
python3.7 -m pip install -e detectron2
```

3.下载源码包到服务器任意目录并解压

目录如下
```
├── test 
|    ├── pth2om.sh                  //模型转换脚本，集成了auto tune功能，可以手动关闭  
|    ├── parse.py                   //用于解析性能数据
|    ├── eval_acc_perf.sh         //用于评估精度 
├── createTrainLabelIds.py     //用于准备数据集的脚本
├── gen_dataset_info.py         //用于获取二进制数据集信息的脚本
├── LICENSE                     
├── PointRend_pth2onnx.py       //用于pth转onnx的脚本 
├── PointRend_preprocess.py    //前处理脚本 
├── PointRend_postprocess.py   //后处理脚本
├── PointRend.diff               //补丁文件，用于修改代码
├── README.md            
├── model_final_cf6ac1.pkl        //权重文件
├── PointRend_bs1.om                //om模型
├── PointRend.onnx                   //onnx文件
└── requirements.txt             //依赖包
```


4.数据集准备

1)[cityscape数据集](https://www.cityscapes-dataset.com/)

从官网获取gtFine_trainvaltest.zip和leftImg8bit_trainvaltest.zip，将这两个压缩包解压到创建的/root/datasets/cityscapes文件夹。  


数据集目录结构如下
```
cityscapes
├── gtFine
│   ├── test
│   ├── train
│   ├── val
├── leftImg8bit
│   ├── tesy
│   ├── train
│   ├── val
```


2)[创建labelTrainIds.png](https://github.com/facebookresearch/detectron2/tree/main/datasets#expected-dataset-structure-for-cityscapes) ：
```
python3.7 createTrainIdInstanceImgs.py /root/datasets/cityscapes
```

3)执行PointRend_preprocess.py脚本，输入下载好的原始数据集，输出前处理后的bin文件
```
python3.7 PointRend_preprocess.py /root/datasets/cityscapes ./preprocess_bin
```
“/root/datasets/cityscapes”：数据集所在的路径。
“./preprocess_bin”：处理后的bin文件的输出路径。
运行成功后，即在当前目录下生成“preprocess_bin”文件夹，并在其中生成对应的前处理后的bin文件。



## 2.离线推理
```
昇腾芯片上执行，执行时使npu-smi info查看设备状态，确保device空闲
```
- **2.1 pth转ONNX**
    
    从源码包中获取权重文件“model_final_cf6ac1.pkl”[下载地址](https://dl.fbaipublicfiles.com/detectron2/PointRend/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes/202576688/model_final_cf6ac1.pkl)

    ```
    python3.7 PointRend_pth2onnx.py 'detectron2/projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml' './model_final_cf6ac1.pkl' './PointRend.onnx' 
    
    ```
    >"detectron2/projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml"：模型的配置文件。  
    >"./model_final_cf6ac1.pkl"：使用的权重文件。  
    >"./PointRend.onnx"：表示输出的onnx文件。  

- **2.2 ONNX转om**
1. 配置环境变量
    ```
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```

2. 使用ATC工具将onnx模型转om模型

    ${chip_name}可通过`npu-smi info`指令查看

    ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

    ```
    atc  --model=./PointRend.onnx  --framework=5 --output=PointRend_bs1 --input_format=NCHW  --input_shape="images:1,3,1024,2048" --log=error  --soc_version=${chip_name}
    ```
     
    参数说明：
> - --model：为ONNX模型文件。
> - --framework：5代表ONNX模型。
> - --output：输出的OM模型。
> - --input_format：输入数据的格式。
> - --input_shape：输入数据的shape。
> - --log：日志级别。
> - --soc_version：处理器型号。

- **2.3 性能验证**

    [获取ais_infer工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)  

    ```
    pip3 install aclruntime-0.01-cp37-cp37m-linux_xxx.whl
    git clone https://gitee.com/ascend/tools.git 

    ```


    ```

    推理
    python3  ais_infer.py  --model ./PointRend.om  --input ./preprocess_bin --output ./ --outfmt BIN --batchsize 1

    --model：模型地址
    --input：预处理完的数据集文件夹
    --output：推理结果保存地址
    --outfmt：推理结果保存格式
    --batchsize: batchsize的值
    ```
    |     | 310  | 310P |T4|310P/310|310P/T4|
    |-----|--------|--------|-------------------|---------|------|
    | bs1 | 0.867324 | 2.13464667180749 |2.12966|2.461187136|1.002340619|

- **2.4 精度验证**
    ```
    nohup python3.7 PointRend_postprocess.py /root/dataset/cityscapes/ result/ais_infer_resutl&

    /root/datasets/cityscapes：数据集路径。
    result/ais_infer_resutl”：ais-infer推理结果所在路径  
    ```
    |     | 310精度  | 310P精度 |
    |-----|--------|--------|
    | bs1 | 78.85439023350757 | 78.85824277897258 |

# HigherHRNet模型PyTorch离线推理指导

## 1 环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

```
pip install -r requirements.txt  
```

2.获取，修改与安装开源模型代码

```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
# Install into global site-packages
make install
cd -

git clone https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation.git
cd HigherHRNet-Human-Pose-Estimation
patch -p1 < ../HigherHRNet.patch
cd ..
```

3.[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)
将benchmark.x86_64或benchmark.aarch64放到当前目录

## 2 数据准备

1.获取coco2017数据集 ，新建data文件夹，数据文件目录格式如下：

```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

2. 数据预处理,将原始数据集转换为模型输入的数据

```
python3 HigherHRNet_preprocess.py --output ${prep_output_dir} --output_flip ${prep_output_flip_dir}
```

- 参数说明：
  -  --output：输出的二进制文件（.bin）所在路径
  - --output_flip：输出的二进制文件flip（.bin）所在路径。

3. 生成数据集info文件
```
python3 gen_dataset_info.py bin ${prep_output_dir} ./prep_bin
python3 gen_dataset_info.py bin ${prep_output_flip_dir} ./prep_bin_flip
```

- 参数说明：
  - “bin”：生成的数据集文件格式。
  - ${prep_output_dir}：预处理后的数据文件的相对路径。
  - ${prep_output_flip_dir}：预处理后的数据文件的相对路径。
  - “./prep_bin”：生成的数据集文件保存的路径。
  - “./prep_bin_flip”：生成的数据集文件保存的路径

运行成功后，在当前目录中生成“prep_bin.info”和“prep_bin_flip.info”

## 3 离线推理

1. 获取权重文件方法。

   从这里下载权重文件([GoogleDrive](https://drive.google.com/open?id=1bdXVmYrSynPLSk5lptvgyQ8fhziobD50) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW4AwKRMklXVzndJT0))。

   文件名称：pose_higher_hrnet_w32_512.pth

    ```
    mkdir models
    mv pose_higher_hrnet_w32_512.pth models
    ```
2. 导出.onnx文件。
    ```
    python3 HigherHRNet_pth2onnx.py --weights models/pose_higher_hrnet_w32_512.pth
    ```
    获得pose_higher_hrnet_w32_512_bs1_dynamic.onnx文件。

​        **注：**ATC工具转换.onnx文件到.om文件时，目前支持的**onnx算子版本为11**。

​        需将HigherHRNet_pth2onnx.py脚本中torch.onnx.export方法内的输入参数**opset_version的值需设为11**。

3. 使用ATC工具将ONNX模型转OM模型。

   a. 配置环境变量
   
   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
   
    b. ${chip_name}可通过`npu-smi info`指令查看，例：310P3
   
   ![image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)
   
    ```
    atc --framework=5 --model=models/pose_higher_hrnet_w32_512_bs1_dynamic.onnx  --output=models/pose_higher_hrnet_w32_512_bs1_dynamic --input_format=NCHW --input_shape="input:1,3,-1,-1" --dynamic_image_size="1024,512;960,512;896,512;832,512;768,512;704,512;640,512;576,512;512,512;512,576;512,640;512,704;512,768;512,832;512,896;512,960;512,1024"  --out_nodes="Conv_1818:0;Conv_1851:0" --soc_version=Ascend${chip_name}
    ```
   
- 参数说明：
  - --model：为ONNX模型文件。
  - --framework：5代表ONNX模型。
  - --output：输出的OM模型。
  - --input_format：输入数据的格式。
  - --input_shape：输入数据的shape。
  - --log：日志级别。
  - --soc_version：处理器型号。

运行成功后生成“pose_higher_hrnet_w32_512_bs1.om”模型文件。

4. 开始验证推理

   a. 使用Benchmark工具进行推理。

   执行以下命令增加Benchmark工具可执行权限，并根据OS架构选择工具，如果是X86架构，工具选择benchmark.x86_64，如果是Arm，选择benchmark.aarch64

   ```
   chmod u+x benchmark.${arch}
   ```

   b. 使用Benchmark工具进行推理。

   ```
   python3 HigherHRNet_benchmark.py --bs 1
   ```
   
- 参数说明：
  - --bs：批大小，即1次迭代所使用的样本量，目前仅只支持bs为1
  
   c. 精度验证	
  
   ```
   python3 HigherHRNet_postprocess.py  --dump_dir './result/dumpOutput_device0_bs1' --dump_dir_flip './result/dumpOutput_device0_bs1_flip'
   ```
  
- 参数说明：
  - --dump_dir：生成推理结果所在路径。
  - --dump_dir_flip：生成推理结果所在路径。
  
   后处理输出的结果，日志保存在“output”目录下。
  
   d. 性能验证
  
   ```
   python3 HigherHRNet_performance.py
   ```
  
   GPU机器上执行，执行前使用nvidia-smi查看设备状态，确保device空闲
  
   ```
   trtexec --onnx=models/pose_higher_hrnet_w32_512_bs1_dynamic.onnx --fp16 --shapes=image:1x3x512x512
   ```

**评测结果：**

1. 精度

|     | 310     |310P    |
|-----|---------|---------|
| bs1 | AP:67.1 | AP:67.1 |

目前仅支持bs为1的情况

2. 性能

| Input Shape | 310        | 310P(未优化)   | 310P/310(未优化)   | 310P(优化后）    | 310P/310(优化后）   | T4       | 310P/T4         |
|-------------|------------|------------|----------------|-------------|----------------|----------|----------------|
| (512,512)   | 196        | 172        | 0.88           | 262         | 1.33           | 39       | 6.71           |
| (512,1024)  | 87         | 102        | 1.17           | 126         | 1.45           | 20       | 6.29           |
| (1024,512)  | 90         | 111        | 1.23           | 143         | 1.59           | 20       | 7.14           |
| (512,576)   | 156        | 108        | 0.69           | 229         | 1.47           | 33       | 6.95          |
| (512,640)   | 144        | 99         | 0.69           | 215         | 1.49           | 31       | 6.94           |
| (512,704)   | 140       | 90         | 0.64           | 188         | 1.34           | 27       | 6.96           |
| (512,768)   | 117      | 86        | 0.74           | 172        | 1.47      | 26     | 6.63        |
|(512,832)   |	110	|110	|1.00	|161	|1.46	|24	|6.71   |
|(512,896)   |	100	|111	|1.11	|148	|1.48	|23	|6.45|
|(512,960)   |	101	|100	|0.99	|143	|1.41	|22	|6.48|
|(576,512)   |	156	|108	|0.69	|237	|1.52	|35	|6.76|
|(640,512)   |	142	|98	|0.69	|206	|1.45	|33	|6.23|
|(704,512)   |	136	|89	|0.65	|188	|1.38	|28	|6.72|
|(768,512)   |	116	|86	|0.74	|178	|1.53	|26	|6.84|
|(832,512)   |	113	|110	|0.97	|157	|1.39	|25	|6.27|
|(896,512)   |	102	|112	|1.10	|153	|1.50	|23	|6.63|
|(960,512)   |	101	|102	|1.01	|140	|1.38	|22	|6.35|
| avg/min/max | 124/87/196 | 106/86/172 | 0.88/0.64/1.23 | 179/126/262 | 1.45/1.33/1.59 | 27/20/39 | 6.65/6.23/7.14 |

注：若310P性能太低，可使用以下操作进行调优

在原 ATC 模型转换命令中添加 --auto_tune_mode="RL,GA" 即可实现 autotune 调优。autotune的作用是控制TBE算子编译时能在昇腾AI处理器上寻找最好的性能配置。[参考链接](https://gitee.com/ascend/docs-openmind/tree/master/guide/modelzoo/onnx_model/tutorials/%E4%B8%93%E9%A2%98%E6%A1%88%E4%BE%8B/%E6%80%A7%E8%83%BD%E8%B0%83%E4%BC%98)

```
pip install absl-py sympy decorator  
atc --framework=5 --model=models/pose_higher_hrnet_w32_512_bs1_dynamic.onnx  --output=models/pose_higher_hrnet_w32_512_bs1_dynamic --input_format=NCHW --input_shape="input:1,3,-1,-1" --dynamic_image_size="1024,512;960,512;896,512;832,512;768,512;704,512;640,512;576,512;512,512;512,576;512,640;512,704;512,768;512,832;512,896;512,960;512,1024"  --out_nodes="Conv_1818:0;Conv_1851:0" --auto_tune_mode="RL,GA" --soc_version=Ascend${chip_name}
```

${chip_name}可通过`npu-smi info`指令查看，例：310P3

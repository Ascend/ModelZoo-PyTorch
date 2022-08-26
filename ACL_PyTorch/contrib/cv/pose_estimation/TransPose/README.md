# TransPose模型离线推理指导

## 1. 模型概述

### 1.1 论文地址

[Transpose: Keypoint localization via transformer](https://arxiv.org/pdf/2012.14214.pdf)

### 1.2 代码地址

开源仓：[https://github.com/yangsenius/TransPose](https://github.com/yangsenius/TransPose)<br>
branch：main<br>
commit-id：dab9007b6f61c9c8dce04d61669a04922bbcd148<br>
model-name: TransPose-R-A3


## 2. 搭建环境
本样例配套的CANN版本为[5.1.RC1](https://www.hiascend.com/software/cann/commercial?version=5.1.RC1)。  
硬件环境、开发环境和运行环境准备请参见《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/504/envdeployment/instg/instg_000002.html)》。  
该模型需要以下依赖。

### 2.1 安装依赖
```shell
torch==1.5.0
torchvision==0.6.0
onnx==1.8.1
numpy==1.21.4
Pillow==8.4.0
yacs==0.1.8
opencv-python==4.5.4.60
opencv-contrib-python==3.4.11.45
munkres==1.1.4
EasyDict==1.7
scipy==1.7.3
pandas==1.3.5
pyyaml==6.0
json_tricks==3.15.5
pycocotools==2.0.1
cffi==1.15.0
scikit-image==0.19.1
Cpython==0.0.6
onnx-simplifier==0.3.6
decorator==5.1.1
mpmath==1.2.1
sympy==1.10.1
```

### 2.2 获取开源仓代码
- 在已下载的源码包根目录下，执行如下命令。
    ```shell
    # 获取源码
    git clone https://github.com/yangsenius/TransPose.git
    cd TransPose
    git reset dab9007b6f61c9c8dce04d61669a04922bbcd148 --hard
    patch -p1 < ../TransPose.patch 
    cd ..
    
    # 将开源代码仓加入环境变量
    export xxxx=xxxx
    ```
说明： 请根据实际情况，将开源代码仓路径加入环境变量。

## 3. 模型转换

### 3.1 准备工作
- 获取权重文件“tp_r_256x192_enc3_d256_h1024_mh8.pth”，将文件放入models文件夹内。
    ```shell
    mkdir models
    wget https://github.com/yangsenius/TransPose/releases/download/Hub/tp_r_256x192_enc3_d256_h1024_mh8.pth -P models
    ```

### 3.2 Pytorch模型转ONNX模型


- step1 使用从源码包中获取权重文件“tp_r_256x192_enc3_d256_h1024_mh8.pth”导出onnx文件。  
运行“TransPose_pth2onnx.py”脚本。
    ```shell
    python3 TransPose_pth2onnx.py --weights models/tp_r_256x192_enc3_d256_h1024_mh8.pth
    ```
     获得“tp_r_256x192_enc3_d256_h1024_mh8_bs1.onnx”文件。
- step2 优化Onnx模型
    ```shell
    python3 -m onnxsim models/tp_r_256x192_enc3_d256_h1024_mh8_bs1.onnx models/tp_r_256x192_enc3_d256_h1024_mh8_bs1_sim.onnx
    ```
    获得“tp_r_256x192_enc3_d256_h1024_mh8_bs1_sim.onnx”文件。  



### 3.3 ONNX模型转OM模型

- step1 设置环境变量
    ```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/Ascend/driver/lib64/driver
    ```
    说明：该命令中使用 CANN 默认安装路径(/usr/local/Ascend/ascend-toolkit)中的环境变量，使用过程中请按照实际安装路径设置环境变量。

- step2 执行命令查看芯片名称（${chip_name}）
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
- step3 生成OM模型
    ATC 工具的使用请参考 [ATC模型转换](https://www.hiascend.com/document/detail/zh/canncommercial/51RC2/inferapplicationdev/atctool)
    ```shell
    atc --framework=5 \
        --model=models/tp_r_256x192_enc3_d256_h1024_mh8_bs1_sim.onnx \
        --output=models/tp_r_256x192_enc3_d256_h1024_mh8_bs1 \
        --input_format=NCHW \
        --input_shape="input:1,3,256,192" \
        --fusion_switch_file=fusion_switch.cfg \
        --log=error \
        --soc_version=Ascend${chip_name}
    ```
- 参数说明：
    - --framework：5代表ONNX模型。
    - --model：为ONNX模型文件。  
    - --output：输出的OM模型。
    - --input_format：输入数据的格式。
    - --input_shape：输入数据的shape。
    - --fusion_switch_file：自定义融合规则配置文件的路径。
    - --log：日志级别。
    - --soc_version：昇腾AI处理器型号。

运行成功后生成“tp_r_256x192_enc3_d256_h1024_mh8_bs1.om”模型文件。

## 4. 数据处理

### 4.1 准备原始数据集

获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）
本模型支持COCO2017 4952张图片的验证集。请用户需自行获取COCO2017数据集，上传数据集到服务器并解压到TransPose主目录下，并按所给目录结构放置。  
```    
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
```

数据目录结构请参考： 
```
├── coco
│    ├── images
|    |    ├──val2017
│    ├── annotations
```

### 4.2 数据集预处理
数据预处理将原始数据集转换为模型输入的数据。

执行“TransPose_preprocess.py”脚本，完成预处理。
```shell
python3 TransPose_preprocess.py --output ./prep_data --output_flip ./prep_data_flip
```
参数说明：
- --output：输出的二进制文件（.bin）所在路径。
- --output_flip：输出的二进制文件flip（.bin）所在路径。

每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成prep_data与prep_data_flip目录，用于保存生成的bin文件。
## 5. 离线推理

### 5.1 准备推理工具
推理工具使用ais_infer，须自己拉取源码，打包并安装
```shell
# 指定CANN包的安装路径
export CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest

# 获取源码
git clone https://gitee.com/ascend/tools.git
cd tools/ais-bench_workload/tool/ais_infer/backend/

# 打包，会在当前目录下生成 aclruntime-xxx.whl
pip3.7 wheel ./

# 安装,具体名称视服务器芯片而定
pip3 install --force-reinstall aclruntime-xxx.whl

# 在~/TransPose/tools/ais-bench_workload/tool目录下将ais_infer复制到~/Transpose主目录下
cd ../..
cp -r ais_infer/ ../../..
cd ../../../ais_infer
```
参考：[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer#%E4%BB%8B%E7%BB%8D)



### 5.2 离线推理

- step1 准备工作

    ```shell
    # 设置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/Ascend/driver/lib64/driver
    
    # 在Transpose主目录下提前创建推理结果的保存目录
    mkdir ../prep_data_result
    mkdir ../prep_data_result_flip
    ```

- step2 对预处理后的数据进行推理

    ```shell
    python3.7 ais_infer.py --model ../models/tp_r_256x192_enc3_d256_h1024_mh8_bs{batchsize}.om --input "../prep_data/" --output ../prep_data_result/
    python3.7 ais_infer.py --model ../models/tp_r_256x192_enc3_d256_h1024_mh8_bs{batchsize}.om --input "../prep_data_flip" --output ../prep_data_result_flip/
    ```
    参数说明:
    - --model, OM模型路径
    - --input, 存放预处理 bin 文件的目录路径
    - --output, 推理输出文件夹

### 5.3 精度验证
根据5.2 生成的推理结果，可计算出 OM 模型的离线推理精度。xxx为推理时生成文件夹名称，命名规则为推理时的时间，如2022_08_26-06_44_54。
```shell
    cd ..
    python3.7 TransPose_postprocess.py  --dump_dir './prep_data_result/xxx/' --dump_dir_flip './prep_data_flip/xxx'
```
参数说明：
- --dump_dir：生成推理结果所在路径。
- --dump_dir_flip：生成推理结果所在路径。


### 5.4 性能验证
用 ais_infer 工具进行纯推理100次，然后根据平均耗时计算出吞吐率
```shell
cd tools/ais-bench_workload/tool/ais_infer/
mkdir tmp_out   # 提前创建临时目录用于存放纯推理输出
python3.7 ais_infer.py --model /path/to/model --output ./tmp_out --outfmt BIN  --batchsize ${bs} --loop 100
rm -r tmp_out   # 删除临时目录
```
说明：
1. **性能测试前使用`npu-smi info`命令查看 NPU 设备的状态，确认空闲后再进行测试。否则测出来性能会低于模型真实性能。**
2. 执行上述脚本后，日志中 **Performance Summary** 一栏会记录性能相关的指标，找到以关键字 **throughput** 开头的一行，即为 OM 模型的吞吐率。


## 6. 指标对比
总结：
 1. 310P上离线推理的精度(73.7%)与 PyTorch 在线推理精度(73.8%)持平；
 2. 在310P设备上，batchsize 为 4 时模型性能最优，达 512.22 fps。

指标详情如下：
<table>
<tr>
    <th>模型</th>
    <th>batch_size</th>
    <th>Pytorch精度</th>
    <th>310P精度</th>
    <th>310性能</th>
    <th>T4性能</th>
    <th>310P性能</th>
    <th>310P/310</th>
    <th>310P/T4</th>
</tr>
<tr>
    <td rowspan="7">Transpose</td>
    <td>1</td>
    <td rowspan="7">AP: 73.8%</td>
    <td rowspan="7">AP: 73.7%</td>
    <td>210.43 fps</td>
    <td>497.47 fps</td>
    <td>449.29 fps</td>
    <td>2.14</td>
    <td>0.90</td>
</tr>
<tr>
    <td>4</td>
    <td>206.03 fps</td>
    <td>405.03 fps</td>
    <td>512.22 fps</td>
    <td>2.49</td>
    <td>1.27</td>
</tr>
<tr>
    <td>8</td>
    <td>189.11 fps</td>
    <td>396.20 fps</td>
    <td>494.21 fps</td>
    <td>2.60</td>
    <td>1.25</td>
</tr>
<tr>
    <td>16</td>
    <td>177.46 fps</td>
    <td>263.57 fps</td>
    <td>472.23 fps</td>
    <td>2.66</td>
    <td>1.79</td>
</tr>
<tr>
    <td>32</td>
    <td>164.02 fps</td>
    <td>294.83 fps</td>
    <td>445.44 fps</td>
    <td>2.72</td>
    <td>1.51</td>
</tr>
<tr>
    <td>64</td>
    <td>161.40 fps</td>
    <td>464.43 fps</td>
    <td>481.09 fps</td>
    <td>2.98</td>
    <td>1.04</td>
</tr>
<tr>
    <td>性能最优bs</td>
    <td>210.43 fps</td>
    <td>497.47 fps</td>
    <td>512.22 fps</td>
    <td>2.43</td>
    <td>1.03</td>
</tr>
</table>


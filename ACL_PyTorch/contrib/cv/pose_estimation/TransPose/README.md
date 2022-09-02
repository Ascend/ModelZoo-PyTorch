# TransPose模型离线推理指导

## 1. 模型概述

### 1.1 论文地址

[Transpose: Keypoint localization via transformer](https://arxiv.org/pdf/2012.14214.pdf)

### 1.2 代码地址

开源仓：[https://github.com/yangsenius/TransPose.git](https://github.com/yangsenius/TransPose.git)  
branch：main  
commit-id：dab9007b6f61c9c8dce04d61669a04922bbcd148  
model-name: TransPose-R-A3  


## 2. 推理环境准备

### 2.1 配套软件

1. 该模型离线推理使用 Atlas 300I Pro 推理卡，推理所需配套的软件如下：

    | 配套      | 版本    | 环境准备指导 |
    | --------- | ------- | ---------- |
    | firmware | 1.82.22.2.220 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | driver | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN      | 5.1.RC2 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | Python    | 3.7.5   | -          |
    | PyTorch   | 1.5.0   | -          |
    
2. 安装python依赖：
    ```shell
    conda create -n transpose python=3.7.5
    conda activate transpose
    pip3 install -r requirements.txt
    ```

### 2.2 获取开源仓代码
- 在本仓目录下执行以下命令拉取开源仓代码
    ```shell
    # 获取源码
    git clone https://github.com/yangsenius/TransPose.git
    cd TransPose
    git reset dab9007b6f61c9c8dce04d61669a04922bbcd148 --hard
    patch -p1 < ../TransPose.patch 
    cd ..
    ```

## 3. 模型转换

### 3.1 准备工作
- 获取权重文件“tp_r_256x192_enc3_d256_h1024_mh8.pth”，将文件放入models文件夹内。
    ```shell
    mkdir models
    wget https://github.com/yangsenius/TransPose/releases/download/Hub/tp_r_256x192_enc3_d256_h1024_mh8.pth -P models
    ```

### 3.2 Pytorch模型转ONNX模型


- step1 使用从源码包中获取权重文件“tp_r_256x192_enc3_d256_h1024_mh8.pth”导出onnx文件。  
    ```shell
    python3 TransPose_pth2onnx.py --weights models/tp_r_256x192_enc3_d256_h1024_mh8.pth --bs 1
    ```
     获得batchsize为 1 的 ONNX 模型：tp_r_256x192_enc3_d256_h1024_mh8_bs1.onnx 
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

- 本模型使用 COCO2017 的 val 集来验证模型精度。请自行下载，在当前目录下，按照以下目录结构解压与放置数据：
    ```
    |-- data
        |-- coco
            |-- images
            |   |-- val2017
            |       |-- 000000000139.jpg
            |       |-- 000000000285.jpg
            |       |-- 000000000632.jpg
            |       |-- ... 
            |-- annotations
                |-- person_keypoints_train2017.json
                |-- person_keypoints_val2017.json
    
    ```
- 下载与解压，请参考： 
    ```shell
    mkdir -p data/coco/
    cd data/coco/
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    mkdir images
    unzip val2017.zip -d images
    unzip annotations_trainval2017.zip
    cd ../../
    ```

### 4.2 数据集预处理

- 执行数据预处理脚本将原始数据集转换为模型输入需要的bin文件。
    ```shell
    python3 TransPose_preprocess.py --output ./prep_data --output_flip ./prep_data_flip
    ```
    参数说明：
    - --output：输出的二进制文件（.bin）所在路径。
    - --output_flip：输出的二进制文件flip（.bin）所在路径。

    运行成功后，会在当前目录下生成 prep_data 与 prep_data_flip 目录，用于保存生成的bin文件。

## 5. 离线推理

### 5.1 准备推理工具

- 推理工具使用ais_infer，须自己拉取源码，打包并安装

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
    
    # 在~/TransPose/tools/ais-bench_workload/tool目录下将ais_infer复制到之前的工作目录
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
    
    # 创建推理结果的保存目录
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
- 根据5.2 生成的推理结果，可计算出 OM 模型的离线推理精度。xxx为执行推理时根据开始时间命名的的子目录，如2022_08_26-06_44_54。
    ```shell
    cd ..
    python3.7 TransPose_postprocess.py  --dump_dir './prep_data_result/xxx/' --dump_dir_flip './prep_data_flip_result/xxx'
    ```
    参数说明：
    - --dump_dir：生成推理结果所在路径。
    - --dump_dir_flip：生成推理结果所在路径。


### 5.4 性能验证
- 用 ais_infer 工具进行纯推理100次，然后根据平均耗时计算出吞吐率
    ```shell
    cd ais_infer/
    mkdir tmp_out   # 提前创建临时目录用于存放纯推理输出
    python3.7 ais_infer.py --model /path/to/model --output ./tmp_out --outfmt BIN  --batchsize ${bs} --loop 100
    rm -r tmp_out   # 删除临时目录
    ```
    说明：
    1. **性能测试前使用`npu-smi info`命令查看 NPU 设备的状态，确认空闲后再进行测试。否则测出来性能会低于模型真实性能。**
    2. 运行结束后，日志中 **Performance Summary** 一栏会记录性能相关指标，找到以关键字 **throughput** 开头的一行，行位的数字即为 OM 模型的吞吐率。
    

## 6. 指标对比
- 总结：
    1. 310P上离线推理的精度(73.7%)与开源仓精度(73.8%)持平；
    2. 在310P设备上，batchsize 为 4 时模型性能最优，达 512.22 fps。

- 指标详情如下：
    <table>
    <tr>
        <th>模型</th>
        <th>batch_size</th>
        <th>目标精度</th>
        <th>310P精度</th>
        <th>310性能</th>
        <th>T4性能</th>
        <th>310P性能</th>
        <th>310P/310</th>
        <th>310P/T4</th>
    </tr>
    <tr>
        <td rowspan="7">TransPose-R-A3</td>
        <td>1</td>
        <td rowspan="7"><a href="https://github.com/yangsenius/TransPose#model-zoo">AP: 73.8%</a></td>
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


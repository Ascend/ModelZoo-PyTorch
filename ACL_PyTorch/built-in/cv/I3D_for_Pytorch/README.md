## I3D模型推理指导：

### 文件说明
    1. acl_net.py：pyacl推理工具代码
    2. i3d_inference.py：i3d模型推理脚本
    3. i3d_atc.sh：i3d模型转换脚本
    4. loading.py: 数据异常处理，基于mmaction2源码修改


### 环境依赖
    pip3 install torch==1.5.0
    pip3 install numpy==1.19.5
    pip3 install onnx==1.7.0


### 环境准备
    下载i3d源码或解压文件
    1. 下载链接：https://github.com/open-mmlab/mmaction2/releases/tag/v0.15.0
        下载源码包解压到指定目录下

    2. 下载权重文件
       https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb/i3d_r50_32x2x1_100e_kinetics400_rgb_20200614-c25ef9a4.pth

    3. 拷贝loading.py到mmaction/datasets/pipelines目录下，替换原有文件
   
    4. 拷贝i3d_inference.py到mmaction2源码tools目录下


### 推理端到端步骤

1. pth导出onnx
    ```
    python3.7 tools/pytorch2onnx.py configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py checkpoints/i3d_r50_32x2x1_100e_kinetics400_rgb_20200614-c25ef9a4.pth --shape 1 30 3 32 256 256 --verify --show --output i3d.onnx --opset-version 11
    ```

2. 利用ATC工具转换为om
    ```
    bash i3d_atc.sh
    ```   

3. 数据预处理
    参考链接：https://github.com/open-mmlab/mmaction2/blob/master/docs/data_preparation.md

4. pyACL推理
    设置pyACL环境变量：
    ```
    export PYTHONUNBUFFERED=1
    export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/pyACL/python/site-packages/acl:$PYTHONPATH
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascendtoolkit/latest/acllib/lib64/:$LD_LIBRARY_PAT
    ```

    ```
    python3.7 tools/i3d_inference.py configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py --eval top_k_accuracy mean_class_accuracy --out result.json -bs 1 --model i3d_bs1.om --device_id 1
    ```


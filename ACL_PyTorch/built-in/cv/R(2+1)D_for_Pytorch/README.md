# R(2+1)D模型PyTorch离线推理指导

##  1  环境准备

- **1.1 安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装**

```
pip3.7 install -r requirements.txt   
```

- **1.2 获取，修改与安装开源模型代码**

```
git clone https://github.com/open-mmlab/mmcv -b master 
cd mmcv
git reset --hard 6cb534b775b7502f0dcc59331236e619d3ae5b9f
MMCV_WITH_OPS=1 pip3.7 install -e .

cd ..
git clone https://github.com/open-mmlab/mmaction2 -b master
cd mmaction2 
git reset --hard acce52d21a2545d9351b1060853c3bcd171b7158
python3.7 setup.py develop

```
注：若上述命令不能下载源码，则将https替换为git（如：git clone git://github.com/open-mmlab/mmcv -b master ）

将mmaction2/tools/deployment/目录下的pytorch2onnx.py中的torch.onnx.export添加一个参数：

` dynamic_axes={'0':{0:'-1'}}, `

将r2plus1d_r34_8x8x1_180e_ucf101_rgb2.py文件放在mmaction2/configs/recognition/r2plus1d文件夹下

- **1.3 [获取权重文件](https://www.aliyundrive.com/drive/folder/6130e24c1b56461015b44659bdc650a9d3cd8e71)**

- **1.4 [数据集UCF-101](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)**

将UCF101.rar文件解压，重命名为ucf101，放在 /root/datasets/文件夹下

```
在当前目录创建videos文件夹
mkdir -p ./data/ucf101/videos

将/root/datasets/ucf101文件夹下的视频文件夹复制到videos下
cp -r /root/datasets/ucf101/*  ./data/ucf101/videos

python3.7 ./mmaction2/tools/data/build_rawframes.py ./data/ucf101/videos/ ./data/ucf101/rawframes/ --task rgb --level 2 --ext avi --use-opencv

DATA_DIR_AN="./data/ucf101/annotations"

wget https://www.crcv.ucf.edu/wp-content/uploads/2019/03/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate

unzip -j UCF101TrainTestSplits-RecognitionTask.zip -d ${DATA_DIR_AN}/
rm UCF101TrainTestSplits-RecognitionTask.zip

PYTHONPATH=. python3.7 ./mmaction2/tools/data/build_file_list.py ucf101 data/ucf101/rawframes/ --level 2 --format rawframes --shuffle
```
- **1.5 安装ais_bench推理工具**

请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

## 2 离线推理 

```
昇腾芯片上执行，执行时使npu-smi info查看设备状态，确保device空闲
```
- **2.1 pth转ONNX**
```
python3.7 ./mmaction2/tools/deployment/pytorch2onnx.py ./mmaction2/configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_ucf101_rgb2.py best_top1_acc_epoch_35.pth --verify  --output-file=r2plus1d.onnx --shape 1 3 3 8 256 256

python3.7 -m onnxsim --input-shape="1,3,3,8,256,256" --dynamic-input-shape r2plus1d.onnx r2plus1d_sim.onnx
```
- **2.2 ONNX转om**
   1. 配置环境变量
        ```
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        ```

   2. 使用ATC工具将onnx模型转om模型

        ${chip_name}可通过`npu-smi info`指令查看

        ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

        ```
        # Ascend310 or Ascend310P[1-4]
        atc --framework=5 --model=./r2plus1d_sim.onnx --output=r2plus1d_bs4 --input_format=NCHW --input_shape="0:4,3,3,8,256,256" --log=debug --soc_version=Ascend${chip_name}
        ```

        运行成功后生成r2plus1d_bs4.om模型文件，需要生成其他bs的模型，请直接修改--input_shape="0:4,3,3,8,256,256"对应的batchsize

- **2.3 数据预处理**
```
python3.7 r2plus1d_preprocess.py --config=./mmaction2/configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_ucf101_rgb2.py --bts=1 --output_path=./predata_bts1/
```

- **2.4 模型性能测试**
```
python3.7.5 -m ais_bench --model ./r2plus1d_bs4.om --loop 50
```

- **2.5 模型精度测试**
模型推理数据集
```
python3.7.5 -m ais_bench --model ./r2plus1d_bs4.om --input ./predata_bts1/ --output ./lcmout/ --outfmt NPY
--model：模型地址
--input：预处理完的数据集文件夹
--output：推理结果保存地址
--outfmt：推理结果保存格式
```
精度验证
```
python3.7 r2plus1d_postprocess.py --result_path=./lcmout/2022_xx_xx-xx_xx_xx/sumary.json
--result_path：推理结果中的json文件
```

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

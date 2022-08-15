# SimCLR模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```
说明：PyTorch选用开源1.8.0版本



2.获取，修改与安装开源模型代码  
安装SimCLR
```shell
git clone https://github.com/sthalles/SimCLR
cd SimCLR
conda create -n simclr python=3.7

conda activate simclr
pip install -r requirements.txt 
pip install decorator
pip install sympy
pip install 
python run.py
```

3.获取权重文件  
  https://pan.baidu.com/s/18sZVnLoQpgIj_nuRpG-XnQ
  提取码：irpw 

4.数据集     
1. 获取CIFAR-10数据集
```
#Version：CIFAR-10 python version
```
2. 对压缩包进行解压到/root/datasets文件夹(执行命令：tar -zxvf cifar-10-python.tar.gz -C /root/datasets)，test_batch存放cifar10数据集的测试集图片，文件目录结构如下：
```
root
├── datasets
│   ├── cifar-10-batch-py
│   │   ├── batches.meta
│   │   ├── data_batch_1
|   |   ├── data_batch_2
|   |   ├── data_batch_3
|   |   ├── data_batch_4
|   |   ├── data_batch_5
|   |   ├── test_batch
|   |   ├── readme.html
```



## 2 离线推理 

310p上执行，执行时使npu-smi info查看设备状态，确保device空闲  
1.数据预处理
数据预处理将原始数据集转换为模型输入的数据。
执行“Simclr_preprocess.py”脚本，完成预处理。
示例：
```
python3.7 Simclr_preprocess.py ./cifar-10-batches-py/test_batch ./prep_data
```
每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成“prep_data”二进制文件夹。
“./prep_data”：输出的二进制文件（.bin）所在路径。
2.生成数据集info文件
使用“gen_dataset_info.py” 脚本，输入已经获得的图片文件，输出生成图片数据集的info文件。运行“gen_dataset_info.py” 脚本。
```
python3.7 gen_dataset_info.py bin ./prep_data ./Simclr_model.info 32 32
```
bin：生成的数据集文件格式。
./prep_data：预处理后的数据文件的相对路径。

./Simclr_model.info”：生成的数据集文件保存的路径。
32：图片的宽与高。
运行成功后，在当前目录中生成“Simclr_model.info”。

3.模型转换
使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。
a.获取权重文件
从源码包中获取权重文件：simclr.pth。
b.导出onnx文件
运行 Simclr_pth2onnx.py脚本。
```
python3.7 Simclr_pth2onnx.py ./simclr.pth Simclr_model.onnx
```
 获得Simclr_model.onnx文件。
c.使用ATC工具将ONNX模型转OM模型
配置环境变量：
```
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:${LD_LIBRARY_PATH}
```
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
获取om文件，执行ATC命令:
```
atc --framework=5 --model=Simclr_model1.onnx --output=Simclr_model_bs1 --input_format=NCHW --input_shape="input:1,3,32,32" --log=info --soc_version=Ascend310p --insert_op_conf=aipp.cfg --enable_small_channel=1 --keep_dtype=execeptionlist.cfg
```
运行成功后生成“Simclr_model_bs1.om”模型文件。
·参数说明：
·--model：为ONNX模型文件。
·--framework：5代表ONNX模型。
--output：输出的OM模型。
·--input_format：输入数据的格式。
·--input_shape：输入数据的shape。
·--log：日志级别。
·--soc_version：处理器型号。

4.开始推理验证
a.安装工具
```
pip3 install aclruntime-0.0.1-cp37-cp37m-linux_x86_64.whl
```
b.安装工具所需库
```
cd ais_infer
mkdir result
pip install -r requirements.txt
```
c.纯推理场景（仅获取性能）
```
python3.7 ais_infer.py  --model ../Simclr_model_bs1.om --output ./result/ --outfmt BIN --loop 5
```

d.文件夹输入进行推理
```
python3.7 ais_infer.py --model ../Simclr_model_bs1.om --input "../prep_data/" --output ./result/
```
throughput: 吞吐率。吞吐率计算公式：1000 *batchsize/npu_compute_time.mean

e.获取精度数据（写入log文件）
python3.7 Simclr_postprocess.py  ./ais_infer/re/2022_07_25-10_41_40/ > result_bs1.log
    


 **评测结果：**   
| 模型        |  在线推理精度  | 310离线推理精度 | 基准性能   | 310性能    | 310p离线推理精度| 310p性能   |
| :------:    | :------:      | :------:       | :------:   | :------:   | :------:     | :------:   |
| SimCLR bs1  |   65.625%    |    65.014%     | 2486.69fps | 4210.00fps |  65.870%      | 3333.33fps |
| SimCLR bs4  |   65.625%    |    65.334%     | 39876.3fps | 7920.84fps |  65.199%      | 11764.4fps |
| SimCLR bs8  |   65.625%    |    65.194%     | 2486.69fps | 11859.0fps |  65.154%      | 16666.6fps |
| SimCLR bs16 |   65.625%    |    65.099%     | 39876.3fps | 11015.4fps |  65.329%      | 22792.1fps |
| SimCLR bs32 |   65.625%    |    65.424%     | 2486.69fps | 11594.0fps |  65.559%      | 28318.3fps |
| SimCLR bs64 |   65.625%    |    65.409%     | 39876.3fps | 11884.1fps |  66.080%      | 28571.9fps |

310p最优batch为:bs64。




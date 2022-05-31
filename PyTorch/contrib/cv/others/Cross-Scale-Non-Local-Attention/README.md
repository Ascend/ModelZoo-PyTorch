# CSNLA

This is for CSNLA. The code is built on [EDSR(PyTorch)](https://github.com/sanghyun-son/EDSR-PyTorch).


## CSNLA Detail

Details, see src/model/csnln.py


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the DIV2K and Set5 datasets, and pretrained_models by referring to [Cross-Scale-Non-Local-Attention](https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention)

## Training and Testing

```
# switch to the dir src
cd src
```


To train a model, run 'main.py' with the desired model architecture and the path of the DIV2K dataset. To test a trained model, run 'main.py' with pretained model and the path of the Set5 dataset.

```bash
# xxx is the decompressed directory of datasets.zip, such as /home/CSNLA
# 1p train perf
bash ../test/train_performance_1p.sh --data_path=xxx

# 8p train perf
bash ../test/train_performance_8p.sh --data_path=xxx

# 8p train full
# Remarks: Target accuracy 37.12; test accuracy 36.969
bash ../test/train_full_8p.sh --data_path=xxx 
```


## CSNLA training result
|   名称   | 精度     | 性能   | AMP_Type |
| :----: | ------ | ---- | -------- |
| NPU-1p | -      | 0.67 | O2       |
| NPU-8p | 36.979 | 4.5  | O2       |

## CSNLA Testing

CANN版本: 5.1.RC1

###  获取源码

1. 单击“立即下载”，下载源码包。

2. 上传源码包到服务器任意目录并解压（如：/home/HwHiAiUser）。

   ```
   ├── benchmark.aarch64            //离线推理工具（适用ARM架构） 
   ├── benchmark.x86_64             //离线推理工具（适用x86架构）
   ├── CSNLN_postprocess.py     //数据后处理脚本（包含输出精度）  
   ├── CSNLN_preprocess.py      //数据预处理脚本  
   ├── csnln_x4.onnx            //权重文件转化的onnx文件
   ├── csnln_x4_fix.onnx         //fix后的onnx文件
   ├── csnln_x4_sim.onnx         //优化后的onnx文件
   ├── CSNLN_pth2onnx.py       //pth转onnx脚本
   ├── perf_softmax_transpose.py   //用于优化softmax的性能
   ├── csnln_x4_perf.onnx           //优化softmax后的onnx文件
   ├── csnln_x4_bs1.om              //onnx转化的om文件
   ├── fix_onnx_prelu.py           //修改
   ├── model_x4.pt              //权重文件 
   ├── env.sh                         //环境变量  
   ├── get_info.py                   //用于获取二进制数据集信息的脚本 
   ```

   ​

3. 安装开源仓代码

   “Cross-Scale-Non-Local-Attention”

   ```
   cd contrib/ACL_PyTorch/Research/cv/image_process/Cross-Scale-Non-Local-Attention
   git clone https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention.git -b master
   cd Cross-Scale-Non-Local-Attention/
   git reset af168f99afad2d9a04a3e4038522987a5a975e86 --hard
   cd ../
   ```

### 准备数据集

1. 获取原始数据集。

   ​

   获取Set5数据集，放入“Cross-Scale-Non-Local-Attention”路径下。文件目录结构如下*：*

   ```
   Cross-Scale-Non-Local-Attention
   ├── Set5
   │   ├── HR
   │   ├── LR_bicubic
   │   │   ├── X2
   │   │   ├── X3
   │   │   ├── X4
   ```

2. 数据预处理。

   ```
   python3.7.5 CSNLN_preprocess.py --s  ./Set5/LR_bicubic/X4/ --d prep_dataset
   ```

   “CSNLN_preprocess.py”：预处理脚本文件。

   “./Set5/LR_bicubic/X4/”：数据集路径。

   “prep_dataset”：数据预处理之后存放的路径。

3. 生成数据集info文件

   “prep_bin.info”

   ```
   python3.7.5 get_info.py bin prep_dataset/bin_56 prep_bin.info 56 56
   ```

   “get_info.py”：脚本文件。

   “./prep_dataset”：预处理后的数据文件的**相对路径。**

   “./prep_bin.info”：生成的数据集文件保存的路径。

   ### 模型推理

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   #### 模型转换

   1. 获取权重文件。

      从源码包中获取权重文件“model_x4.pt”。

   2. 导出onnx文件。

      1. 使用

         “model_x4.pt”

         导出onnx文件。

         “CSNLN_pth2onnx.py”

         ```
         python3.7.5 CSNLN_pth2onnx.py --n_feats 128 --pre_train model_x4.pt --save csnln_x4.onnx
         ```

         “CSNLN_pth2onnx.py”为执行脚本。

         --pre_train：权重文件。

         --save：生成的onnx文件。

         获得“csnln_x4.onnx”文件。

   3. 编辑环境变量

      ```
      source /usr/local/Ascend/.../set_env.sh
      ```

   4. 执行ATC命令，生成om文件

      ```
      atc --framework=5 --model=csnln_x4_perf.onnx --output=csnln_x4_bs1 --input_format=NCHW --input_shape="input.1:1,3,56,56" --log=debug --soc_version=Ascend310
      ```

   #### 推理验证

   使用Benchmark工具进行推理。

   1. 执行以下命令增加Benchmark工具可执行权限，并根据OS架构选择工具，如果是X86架构，工具选择benchmark.x86_64，如果是Arm，选择benchmark.aarch64 。

      ```
      chmod u+x benchmark.x86_64
      ```

      - 二进制输入

        ```
        ./benchmark.x86_64 -model_type=vision -batch_size=1 -device_id=0 -om_path=csnln_x4_bs1.om -input_text_path=prep_bin.info -input_width=56 -input_height=56 -useDvpp=false -output_binary=true 
        ```

        参数说明：

      - -model_type：模型类型

      - -om_path：om文件路径

      - -device_id：NPU设备编号

      - -batch_size：参数规模

      - -input_text_path：图片二进制信息

      - -input_width：输入图片宽度

      - -input_height：输入图片高度

      - -useDvpp：是否使用Dvpp

      - -output_binary：输出二进制形式

      推理后的输出默认在当前目录“result”下。

   2. 精度验证。

      ```
      python3.7.5 CSNLN_postprocess.py --hr  ./Set5/HR/ --res result/dumpOutput_device0 --save_path res_png
      ```

      --hr：生成推理结果所在路径。

      --res：标签数据。

      --save_path：生成结果文件。

      “datasets_path”：Set5的路径。

      “result/dumpOutput_device0”：推理结果目录。

      “res_png”：保存推理完成之后的图片路径

   #### 性能调优

   使用AOE工具进行性能调优

   ```
   aoe --framework=5 --model=csnln_x4.onnx --job_type=2 --output=./test_perf  --input_format=NCHW --input_shape="image:1,3,56,56"
   ```

   进行纯推理，验证优化性能

   ```
   ./benchmark.x86_64 -round=20 -om_path=test_perf.om -device_id=0 -batch_size=1
   ```

   ​

|      | 精度    | 性能       | AOE优化性能  |
| ---- | ----- | -------- | -------- |
| bs1  | 32.57 | 0.184669 | 0.314836 |

备注：由于内存限制，离线模型不支持多batch
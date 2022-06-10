### Cross-Scale-Non-Local-Attention模型PyTorch离线推理指导

#### 1 环境准备

1.安装必要的依赖

`pip3.7 install -r requirements.txt` 

2.上传源码包到服务器任意目录并解压（如：/home/HwHiAiUser）。

```
├── benchmark.aarch64            //离线推理工具（适用ARM架构） 
├── benchmark.x86_64             //离线推理工具（适用x86架构）
├── CSNLN_postprocess.py     //数据后处理脚本（包含输出精度）  
├── CSNLN_preprocess.py      //数据预处理脚本  
├── CSNLN_pth2onnx.py       //pth转onnx脚本
├── csnln_x4_bs1.om              //onnx转化的om文件
├── model_x4.pt              //权重文件 
├── env.sh                         //环境变量  
├── get_info.py                   //用于获取二进制数据集信息的脚本 
```

3.获取开源模型代码


```
git clone https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention.git -b master
cd Cross-Scale-Non-Local-Attention/
git reset af168f99afad2d9a04a3e4038522987a5a975e86 --hard
cd ../
```

4.获取权重文件 Cross-Scale-Non-Local-Attention预训练pth权重文件

model_x4.pt放在当前目录

5.获取Set5数据集

`wget https://cv.snu.ac.kr/research/EDSR/benchmark.tar`

解压并获取benchmark目录下的Set5数据集，放在当前目录,文件目录结构如下

```
Cross-Scale-Non-Local-Attention
├── Set5
│   ├── HR
│   ├── LR_bicubic
│   │   ├── X2
│   │   ├── X3
│   │   ├── X4
```

- 数据预处理。

```
python3.7.5 CSNLN_preprocess.py --s  ./Set5/LR_bicubic/X4/ --d prep_dataset
```

“CSNLN_preprocess.py”：预处理脚本文件。

“./Set5/LR_bicubic/X4/”：数据集路径。

“prep_dataset”：数据预处理之后存放的路径。

- 生成数据集info文件

“prep_bin.info”

```
python3.7.5 get_info.py bin prep_dataset/bin_56 prep_bin.info 56 56
```

“get_info.py”：脚本文件。

“./prep_dataset”：预处理后的数据文件的**相对路径。**

“./prep_bin.info”：生成的数据集文件保存的路径。

6.获取benchmark工具

```
https://gitee.com/ascend/cann-benchmark/tree/master/infer
```

将benchmark.x86_64或benchmark.aarch64放到当前目录

#### 2 离线推理

310p上执行，执行时使npu-smi info查看设备状态，确保device空闲

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

|      | 精度    | 性能       |
| ---- | ----- | -------- |
| bs1  | 32.57 | 0.314836 |


备注：由于内存限制，离线模型不支持多batch

# 3DUNet模型PyTorch离线推理指导

## 1 环境准备 

1. 安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3.7 install -r requirements.txt  
```

2. 获取，修改与安装开源模型代码  

```
git clone --recurse-submodules https://github.com/mlcommons/inference.git
cd inference
git reset 74353e3118356600c1c0f42c514e06da7247f4e8 --hard
cd vision/medical_imaging/3d-unet
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet/
git reset b38c69b345b2f60cd0d053039669e8f988b0c0af --hard
cd ../
mv nnUNet nnUnet
```

3. 编译环境，进入inference/loadgen目录，执行以下命令。

   ```
   CFLAGS="-std=c++14 -O3" python3 setup.py develop
   ```

   若失败，则升级gcc版本

4. 下载网络权重文件并导出onnx

   下载链接：https://zenodo.org/record/3903982#.YL9Ky_n7SUk下载fold_1.zip

   在3d-unet目录下创建build/result目录，并将下载的fold_1.zip文件解压，将nnUNet目录放在result目录下，文件目录为：

   ```
   3d-unet/build/result/nnUNet/3d_fullres/Task043_BraTS2019/nnUNetTrainerV2__nnUNetPlansv2.mlperf.1/
   ```

   运行脚本导出onnx，onnx默认保存在build/model下，

```
python3 unet_pytorch_to_onnx.py
```

5. 运行脚本将onnx转为om模型，该框架和应用场景都是单batch，导出单batch om模型即可

```
bash atc.sh
```

## 2 离线推理 

1. 修改运行脚本Task043_BraTS_2019.py，在main函数中添加以下内容

   ```
   nnUNet_raw_data="./build/raw_data/nnUNet_raw_data"
   maybe_mkdir_p(nnUNet_raw_data)
   ```
   
2. 修改onnxruntime_SUT.py

   import头文件

   ```
   from acl_net import Net 
   ```

   __init__函数中增加

   ```
   self.model = Net(device_id = 0, model_path = model_path)
   ```

   注释self.sess:

   ```
   #self.sess = onnxruntime.InferenceSession(model_path)
   ```

   issue_queries函数中修改output

   ```
   output = self.model(data[np.newaxis, ...])[0].squeeze(0).astype(np.float16)
   ```
   
   或者直接通过patch文件进行修改：
   
   ```
   cd inference
   patch -p1 < ../3DUnet.patch
   ```

3. 运行脚本

   手动创建build/postprocessed_data/目录

   ```
   mkdir build/postprocessed_data/
   ```

```
python3 Task043_BraTS_2019.py
python3 preprocess.py
```

3. 获取精度

   将acl_net.py拷贝到3d-unet目录下，运行source /usr/local/Ascend/ascend-toolkit/set_env.sh设置环境变量。运行如下命令获取精度：

   ```
   python3 run.py --accuracy --backend onnxruntime --model ./build/model/3DUnet.om
   ```

**评测结果：**   

|    模型    |    官网pth精度    | 310P/310离线推理精度 | gpu性能 |         310P性能         | 310性能 |
| :--------: | :---------------: | :-----------------: | :-----: | :---------------------: | ------- |
| 3DUNet bs1 | mean tumor:0.8530 |  mean tumor:0.8530  | 0.5fps  | ~~4.4fps~~<br />6.26fps | 0.78fps |




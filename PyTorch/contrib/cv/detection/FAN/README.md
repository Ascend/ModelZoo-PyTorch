# FAN

本项目实现了FAN (Face Alignment Network) 从GPU到NPU上在线推理的迁移，源开源代码仓[face-alignment](https://github.com/1adrianb/face-alignment)

## FAN Detail

本项目对于face-alignment做出了如下更改：

1. 将设备从Nvidia GPU迁移到Huawei NPU上。
2. 根据源码提供的接口增加了测试脚本。
3. 添加了FAN模型的python文件。
4. 将源码中使用jit.load离线加载的模型改为torch.load加载，权重和结构没变。
5. 对于一些操作，使用 NPU 算子优化性能，同时将一些操作转移到 CPU 上进行。

## Requirements

```bash
pip install -r requirements.txt
```

- NPU 配套的run包安装
- Python3.7.5
- PyTorch（NPU版本）



### 准备数据集
数据集下载参考源代码仓

1. 下载[300-W](http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz)数据集，将图片放在这个目录下（数据集包含4个子数据集共7674张图片）：

   ```bash
   $FAN_ROOT/dataset/
   ```

   最终数据集目录结构如下：

   ```
    FAN
    ├── face_ailgnment
    ├── dataset
    │   └── ibug_300W_large_face_landmark_dataset
    │       ├── aww
    │       ├── bug
    │       ├── Helen
    │       │   ├── trainset
    │       │   └── testset
    │       └── lfpw
    │           ├── trainset
    │           └── testset
    ├── eval.py
    ├── eval.sh
    ├── utils.py
    ```

### Test

```bash
#在npu上测试FAN效果，生成人脸关键点图像和关键点坐标
bash ./test/train_full_1p.sh --data_path=数据集路径 --landmarks_type=关键点类型"2D"或"3D"
#2D关键点坐标保存路径如下
$FAN_ROOT/result/points/2D_npu.npy 
#3D关键点坐标保存路径如下
$FAN_ROOT/result/points/3D_npu.npy 
#2D关键点图像保存路径如下
$FAN_ROOT/result/images/2D/ 
#3D关键点图像保存路径如下
$FAN_ROOT/result/images/3D/ 
```

评估FAN在NPU上2D或3D人脸关键点检测运行的效率，打印平均FPS
```bash
#仅测试部分图片
bash train_performance_1p.sh --data_path=数据集路径 --landmarks_type=关键点类型"2D"或"3D"
#log保存路径
$FAN_ROOT/test/output/ 
```
```bash
#测试全部图片
bash train_full_1p.sh --data_path=数据集路径 --landmarks_type=关键点类型"2D"或"3D"
#log保存路径
$FAN_ROOT/test/output/ 
```


## Performance
NPU需要先编译一次才会得到正常效率，FPS的测试都是编译过之后测试的。

<table align="center">
	<tr>
	    <th> </th>
	    <th>landmarks_type</th>
	    <th>FPS_perf</th>  
	</tr >
	<tr >
	    <td align="center" rowspan="2">GPU</td>
	    <td align="center">2D</td>
	    <td align="center">0.62</td>
	</tr>
	<tr>
	    <td align="center">3D</td>
	    <td align="center">0.49</td>
	</tr>
        <tr >
	    <td align="center" rowspan="2">NPU</td>
	    <td align="center">2D</td>
	    <td align="center">0.59</td>
	</tr>
	<tr>
	    <td align="center">3D</td>
	    <td align="center">0.47</td>
	</tr>
	
</table>


### GPU2DFAN人脸关键点图像结果

![输入图片说明](https://arx-2022.obs.cn-north-4.myhuaweicloud.com/pic/demo2DGPU.jpg)

### NPU2DFAN人脸关键点图像结果

![输入图片说明](https://arx-2022.obs.cn-north-4.myhuaweicloud.com/pic/demo2DNPU.png)

### GPU3DFAN人脸关键点图像结果

![输入图片说明](https://arx-2022.obs.cn-north-4.myhuaweicloud.com/pic/demo3DGPU.png)

### NPU3DFAN人脸关键点图像结果

![输入图片说明](https://arx-2022.obs.cn-north-4.myhuaweicloud.com/pic/demo3DNPU.png)

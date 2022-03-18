# TransPose模型PyTorch离线推理指导

## 环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

```
pip install -r requirements.txt  
```

2.获取，修改与安装开源模型代码

```
git clone https://github.com/yangsenius/TransPose.git
cd TransPose
git reset dab9007b6f61c9c8dce04d61669a04922bbcd148 --hard
patch -p1 < ../TransPose.patch
cd ..
```

3.获取权重文件

Download pretrained models from the releases. 
> (tp_r_256x192_enc3_d256_h1024_mh8.pth)

```
${POSE_ROOT}
 `-- models
     `-- pytorch
         |-- imagenet
         |   |-- hrnet_w32-36af842e.pth
         |   |-- hrnet_w48-8ef0771d.pth
         |   |-- resnet50-19c8e357.pth
         |-- transpose_coco
         |   |-- tp_r_256x192_enc3_d256_h1024_mh8.pth
         |   |-- tp_r_256x192_enc4_d256_h1024_mh8.pth
         |   |-- tp_h_32_256x192_enc4_d64_h128_mh1.pth
         |   |-- tp_h_48_256x192_enc4_d96_h192_mh1.pth
         |   |-- tp_h_48_256x192_enc6_d96_h192_mh1.pth    
```

```
mkdir -p models
mv tp_r_256x192_enc3_d256_h1024_mh8.pth models
```

4.数据集 

获取coco_val2017，新建data文件夹，数据文件目录格式如下：

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

5.[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)
将benchmark.x86_64或benchmark.aarch64放到当前目录

## 离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲

```
bash test/pth2om.sh
bash test/eval_acc_perf.sh --datasets_path=./data/coco
```

GPU机器上执行，执行时使用nvidia-smi查看设备状态，确保device空闲

```
bash test/perf_g.sh
```

**评测结果：**

| 模型           | pth精度  | 310离线推理精度 | 基准性能 | 310性能 |
| -------------- | -------- | --------------- | -------- | ------- |
| TransPose bs1  | mAP:73.8 | mAP:73.7        | 507fps   | 30fps   |
| TransPose bs4  |  |        | 619fps   | 96fps   |
| TransPose bs8 |  |         | 444fps   | 152fps  |
| TransPose bs16 | mAP:73.8 | mAP:73.7        | 444fps   | 231fps  |
| TransPose bs32  |  |        | 438fps   | 296fps   |

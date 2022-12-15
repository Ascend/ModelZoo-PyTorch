# DeepRemaster

-   [概述](概述.md)
-   [准备测试环境](准备测试环境.md)
-   [开始测试](开始测试.md)
-   [测试结果展示](测试结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述
DeepRemaster是一个能够对老电影进行修复和上色的半交互式框架，主要特点是采用了源-参考注意网络，允许模型处理任意数量的参考彩色图像来为长视频上色，而不需要分割，同时能够保持时间一致性。源-参考注意网络使用参考图的颜色信息来间接控制原图的着色，并且加入了自注意力机制来利用非局部时间信息以增强序列着色的一致性。该框架优于现有的系列方法，与其他方法相比，其性能随着视频的变长和参考彩色图像的增加而提高。

# 准备测试环境

## 准备环境
- pytorch 1.8.1+ascend.rc2.20221121

- pytorch-npu 1.8.1rc3.post20221121

- ffmeg 4.3.2(配置为 --enable-libx264)

- opencv(3.4.1+)

- scikit-image

- tqdm



## 准备数据集
测试集下载地址：链接：https://pan.baidu.com/s/1or0HTBqOBvcEFnapJjDF-g 
提取码：HWDR

注：下载后请保存在源码包目录下的example文件夹中，如下所示。

```
├── example
    ├── a-bomb_blast_effects_part.mp4
    ├── references
        ├── 0000874_out.png
        ├── 0000968_out.png
        ├── 0001063_out.png
        ├── 0001220_out.png
```

# 开始测试

## 测试模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 下载训练完的模型参数。
   ```
   bash download_model.sh 
   ```

3. 运行代码
   ```
   python remaster.py --input example/a-bomb_blast_effects_part.mp4 --reference_dir example/references --npu
   ```
   代码运行结束后，将保存三个视频，分别为：1. 原视频：a-bomb_blast_effects_part_in.mp4；2. 修复上色后的视频：a-bomb_blast_effects_part_out.mp4；3. 上述两者的拼接视频：a-bomb_blast_effects_part_comp.mp4
# 测试结果展示
该部分只展示拼接视频中部分帧的图片，其中左侧为原视频中的图片，右侧为利用DeepRemaster模型完成修复上色任务后得到的图片。

![](https://s2.loli.net/2022/12/15/1B9mEtGAhYIqaQH.png)

![](https://s2.loli.net/2022/12/15/tg6epGTXQE9DNMO.png)

![](https://s2.loli.net/2022/12/15/XiUeRBb314Z2p7A.png)

![](https://s2.loli.net/2022/12/15/6DkXB7cPtVzglIu.png)



该视频在GPU和NPU上完成修复和上色任务耗时如表1所示。

**表1**  测试耗时展示表

| Name | Time     |
| ---- | -------- |
| GPU  | 73.02 s  |
| NPU  | 126.94 s |



# 高级参考
运行代码的基础用法如下所示：
   ```
   python remaster.py --input "视频路径" --reference_dir "参考图片文件夹路径"
   ```
脚本参数：
   ```
   --input: 视频路径
   --reference_dir: 参考图片文件夹路径
   --npu: 是否在npu上进行计算，默认设置为false。
   --disable_colorization: 只完成修复任务而不进行上色任务，默认为false。
   ```












#  vit-small 模型PyTorch离线推理指导

## 1. 环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

~~~shell
pip3.7 install -r requirements.txt
~~~

2.获取，修改与安装开源模型代码

~~~shell
git clone https://github.com/rwightman/pytorch-image-models.git -b master
cd pytorch-image-models/
patch -p1 < ../vit_small_patch16_224.patch
cd ..
~~~

3.获取权重文件

将权重文件放到当前工作目录，可以通过以下命令下载：

~~~shell
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/vit-small/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz
~~~

4.数据集

该模型使用 `ImageNet` 官网的5万张验证集进行测试，可从 [ImageNet官网](http://www.image-net.org/) 获取 `val` 数据集与标签，分别存放在 `/home/datasets/imagenet/val` 与 `/home/datasets/imagenet/val_label.txt`

最终目录结构应为

~~~txt
imagenet/
|-- val/
|-- val_label.txt
~~~

5.[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)

将 `benchmark.x86_64` 或 `benchmark.aarch64` 放到当前工作目录

## 2. 离线推理

t4上执行，确保卡空闲时执行测试

~~~shell
bash perf_g.sh
# 脚本3-11行是对gpu bs1性能测试， 13-21行是对gpu bs16性能测试
~~~

310上执行，执行时使npu-smi info查看设备状态，确保device空闲

~~~shell
bash ./test/pth2om.sh  
# 脚本中2-7行是pth2onnx，9-16行是对onnx进行onnxsim优化, 18-27行是利用onnxsim优化后的onnx转om

bash ./test/eval_acc_perf.sh --datasets_path=real_data_path
# 如不指定--datasets_path，则采用默认数据集路径：/home/datasets/imagenet/
# 脚本中12-19行为数据集前处理，同时生成bin文件信息，20-30行是推理过程，32-44行获取精度数据，45-56行获取性能数据
~~~

**评测结果**

| 模型           | 仓库pth精度 | 310离线推理精度 | 基准性能 | 310性能  |
| -------------- | ----------- | --------------- | -------- | -------- |
| vit-small bs1  | top1:81.388 | top1:81.1       | 222.4976 | 200.9196 |
| vit-small bs16 | top1:81.388 | top1:81.1       | 657.9001 | 204.0776 |


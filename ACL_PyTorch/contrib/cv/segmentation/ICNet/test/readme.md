# 环境准备：

## 1.数据集路径  

数据集统一放在/root/datasets/或/opt/npu/  
本模型数据集放在/opt/npu/cityscapes/  

## 2.进入工作目录  

cd ICNet  

## 3.安装必要的依赖 ,测试环境可能已经安装其中的一些不同版本的库，故手动测试时不推荐使用该命令安装

pip3.7 install -r requirements.txt  

## 4.下载开源模型代码，并修改模型代码后安装  

```
git clone https://github.com/liminn/ICNet-pytorch.git
cd ICNet-pytorch
patch -p1 < ../icnet.diff
cd ..
cp -rf ./ICNet-pytorch/utils ./
```

## 5.获取权重文件  

```  
wget https://ascend-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/instance_segmentation/ICNet/%E6%8E%A8%E7%90%86/rankid0_icnet_resnet50_192_0.687_best_model.pth
```

## 6.获取benchmark工具  

将benchmark.x86_64 benckmark.aarch64放在当前目录  

## 7.310上执行，执行时确保device空闲  

bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/opt/npu/
